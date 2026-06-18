//! Cooperative "carousel" scan — a continuously-cycling shared binary scan that
//! queries hop onto, instead of fixed query-tiling that must wait for a batch.
//!
//! Motivation: tiling (history 016/038) amortizes the 122 MB base read across a
//! tile of queries, but assumes the tile is full. Real traffic is bursty: waiting
//! to fill a tile adds latency ("the car waits at the station"), and scanning a
//! partial tile wastes the base read on empty seats. The carousel keeps each
//! worker streaming the base in a loop; an arriving query attaches at the current
//! cursor, rides exactly one full revolution (sees all N docs), then leaves with
//! its top-C. Each doc read is shared by every query currently aboard — dynamic
//! tiling with the tile = whoever's in flight. This is classic scan-sharing
//! (Crescando / IBM Blink).
//!
//! This binary is a self-contained benchmark: it generates Poisson (bursty)
//! arrivals at a target rate and measures user-facing latency + throughput, for
//! either the carousel or the per-query baseline (the 017 serving model).

use std::collections::BinaryHeap;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use vector_search::{fvecs, quant};

#[derive(Parser)]
#[command(about = "Cooperative carousel scan vs per-query serving, under bursty load")]
struct Args {
    #[arg(long, default_value = "data/sift")]
    data: std::path::PathBuf,
    #[arg(long, default_value = "sift")]
    prefix: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 1000)]
    rerank: usize,
    /// "carousel" or "perquery"
    #[arg(long, default_value = "carousel")]
    mode: String,
    /// Offered load (arrivals/sec, Poisson).
    #[arg(long, default_value_t = 200.0)]
    rate: f64,
    /// Measurement duration (seconds), after warmup.
    #[arg(long, default_value_t = 6.0)]
    duration: f64,
    #[arg(long, default_value_t = 1.0)]
    warmup: f64,
    /// Max queries aboard one worker's carousel (the "seats").
    #[arg(long, default_value_t = 8)]
    seats: usize,
    /// Docs processed per carousel step before re-checking for arrivals.
    #[arg(long, default_value_t = 4096)]
    chunk: usize,
    /// Worker threads (0 = CPU cores).
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// Fan-out F for --mode grouped: F workers cooperate on each query (revolution
    /// = N/F), in G = workers/F independent groups. F=1 packs (max throughput);
    /// F=workers fully shards (min latency).
    #[arg(long, default_value_t = 1)]
    fan: usize,
}

struct Job {
    qbin: Arc<Vec<u64>>,
    qf: Arc<Vec<f32>>,
    arrival: Instant,
}

struct Aq {
    qbin: Arc<Vec<u64>>,
    qf: Arc<Vec<f32>>,
    arrival: Instant,
    remaining: isize,
    heap: BinaryHeap<(u32, u32)>,
}

/// Tiny LCG for Poisson inter-arrival times (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        // (0,1)
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
}

fn finalize(aq: Aq, basef: &fvecs::Vectors, k: usize) -> f64 {
    let cands: Vec<u32> = aq.heap.into_iter().map(|(_, i)| i).collect();
    let _ = quant::rerank(basef, &aq.qf, &cands, k);
    aq.arrival.elapsed().as_secs_f64() * 1000.0
}

fn carousel_worker(
    bbase: Arc<quant::QuantBinary>,
    basef: Arc<fvecs::Vectors>,
    rx: Arc<Mutex<Receiver<Job>>>,
    seats: usize,
    chunk: usize,
    k: usize,
    rerank_c: usize,
    out: mpsc::Sender<f64>,
) {
    let n = bbase.len();
    let want = rerank_c.max(k);
    let mut active: Vec<Aq> = Vec::with_capacity(seats);
    let mut cursor = 0usize;
    let mut disconnected = false;
    loop {
        // Admit new arrivals (up to seats) without blocking.
        while active.len() < seats {
            let got = { rx.lock().unwrap().try_recv() };
            match got {
                Ok(j) => active.push(Aq {
                    qbin: j.qbin,
                    qf: j.qf,
                    arrival: j.arrival,
                    remaining: n as isize,
                    heap: BinaryHeap::with_capacity(want + 1),
                }),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if active.is_empty() {
            if disconnected {
                return;
            }
            // Idle: block briefly for the next arrival rather than spin.
            let got = { rx.lock().unwrap().recv_timeout(Duration::from_millis(2)) };
            match got {
                Ok(j) => active.push(Aq {
                    qbin: j.qbin,
                    qf: j.qf,
                    arrival: j.arrival,
                    remaining: n as isize,
                    heap: BinaryHeap::with_capacity(want + 1),
                }),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }
        // Process one chunk of docs against everyone aboard (shared read).
        let end = (cursor + chunk).min(n);
        for d in cursor..end {
            let doc = bbase.row(d);
            for aq in active.iter_mut() {
                let h = quant::hamming(&aq.qbin, doc);
                if aq.heap.len() < want {
                    aq.heap.push((h, d as u32));
                } else if h < aq.heap.peek().unwrap().0 {
                    aq.heap.pop();
                    aq.heap.push((h, d as u32));
                }
            }
        }
        let processed = (end - cursor) as isize;
        cursor = if end >= n { 0 } else { end };
        // Complete any query that has now ridden a full revolution.
        let mut i = 0;
        while i < active.len() {
            active[i].remaining -= processed;
            if active[i].remaining <= 0 {
                let aq = active.swap_remove(i);
                let lat = finalize(aq, &basef, k);
                out.send(lat).ok();
            } else {
                i += 1;
            }
        }
    }
}

fn perquery_worker(
    bbase: Arc<quant::QuantBinary>,
    basef: Arc<fvecs::Vectors>,
    rx: Arc<Mutex<Receiver<Job>>>,
    k: usize,
    rerank_c: usize,
    out: mpsc::Sender<f64>,
) {
    loop {
        let got = { rx.lock().unwrap().recv() };
        match got {
            Ok(j) => {
                let cands = quant::knn_binary(&bbase, &j.qbin, rerank_c.max(k));
                let _ = quant::rerank(&basef, &j.qf, &cands, k);
                out.send(j.arrival.elapsed().as_secs_f64() * 1000.0).ok();
            }
            Err(_) => return,
        }
    }
}

// ---- Sharded carousel: docs split across workers; a query rides ALL shards in
// parallel (revolution = N/workers docs), partial heaps merged on completion. ----

struct Comp {
    partials: Mutex<Vec<Vec<(u32, u32)>>>,
    remaining: std::sync::atomic::AtomicUsize,
    arrival: Instant,
    qf: Arc<Vec<f32>>,
}

enum CoordMsg {
    Job(Job),
    Done,
    Stop,
}

struct ShardAq {
    comp: Arc<Comp>,
    qbin: Arc<Vec<u64>>,
    heap: BinaryHeap<(u32, u32)>,
    remaining: isize,
}

#[allow(clippy::too_many_arguments)]
fn shard_worker(
    bbase: Arc<quant::QuantBinary>,
    basef: Arc<fvecs::Vectors>,
    lo: usize,
    hi: usize,
    rx: Receiver<(Arc<Comp>, Arc<Vec<u64>>)>,
    coord: mpsc::Sender<CoordMsg>,
    seats: usize,
    chunk: usize,
    k: usize,
    rerank_c: usize,
    out: mpsc::Sender<f64>,
) {
    let want = rerank_c.max(k);
    let span = hi - lo;
    let mut active: Vec<ShardAq> = Vec::with_capacity(seats);
    let mut cursor = lo;
    let mut disconnected = false;
    loop {
        while active.len() < seats {
            match rx.try_recv() {
                Ok((comp, qbin)) => active.push(ShardAq {
                    comp,
                    qbin,
                    heap: BinaryHeap::with_capacity(want + 1),
                    remaining: span as isize,
                }),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if active.is_empty() {
            if disconnected {
                return;
            }
            match rx.recv_timeout(Duration::from_millis(2)) {
                Ok((comp, qbin)) => active.push(ShardAq {
                    comp,
                    qbin,
                    heap: BinaryHeap::with_capacity(want + 1),
                    remaining: span as isize,
                }),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }
        let end = (cursor + chunk).min(hi);
        for d in cursor..end {
            let doc = bbase.row(d);
            for aq in active.iter_mut() {
                let h = quant::hamming(&aq.qbin, doc);
                if aq.heap.len() < want {
                    aq.heap.push((h, d as u32));
                } else if h < aq.heap.peek().unwrap().0 {
                    aq.heap.pop();
                    aq.heap.push((h, d as u32));
                }
            }
        }
        let processed = (end - cursor) as isize;
        cursor = if end >= hi { lo } else { end };
        let mut i = 0;
        while i < active.len() {
            active[i].remaining -= processed;
            if active[i].remaining <= 0 {
                let aq = active.swap_remove(i);
                let part: Vec<(u32, u32)> = aq.heap.into_iter().collect();
                let comp = aq.comp;
                comp.partials.lock().unwrap().push(part);
                // last shard to finish merges + reranks
                if comp.remaining.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                    let mut all: Vec<(u32, u32)> = comp.partials.lock().unwrap().drain(..).flatten().collect();
                    all.sort_unstable();
                    let cands: Vec<u32> = all.into_iter().take(want).map(|(_, id)| id).collect();
                    let _ = quant::rerank(&basef, &comp.qf, &cands, k);
                    out.send(comp.arrival.elapsed().as_secs_f64() * 1000.0).ok();
                    coord.send(CoordMsg::Done).ok();
                }
            } else {
                i += 1;
            }
        }
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[allow(clippy::too_many_arguments)]
fn run_sharded(
    args: &Args,
    bbase: Arc<quant::QuantBinary>,
    basef: Arc<fvecs::Vectors>,
    pool: Vec<(Arc<Vec<u64>>, Arc<Vec<f32>>)>,
    workers: usize,
) {
    use std::collections::VecDeque;
    let n = bbase.len();
    let span = n / workers;
    let (coord_tx, coord_rx) = mpsc::channel::<CoordMsg>();
    let (otx, orx) = mpsc::channel::<f64>();

    // Per-shard worker channels.
    let mut worker_txs = Vec::with_capacity(workers);
    let mut whandles = Vec::new();
    for w in 0..workers {
        let lo = w * span;
        let hi = if w == workers - 1 { n } else { (w + 1) * span };
        let (wtx, wrx) = mpsc::channel::<(Arc<Comp>, Arc<Vec<u64>>)>();
        worker_txs.push(wtx);
        let (bb, bf, ct, ot) = (bbase.clone(), basef.clone(), coord_tx.clone(), otx.clone());
        let (seats, chunk, k, rc) = (args.seats, args.chunk, args.k, args.rerank);
        whandles.push(thread::spawn(move || {
            shard_worker(bb, bf, lo, hi, wrx, ct, seats, chunk, k, rc, ot);
        }));
    }
    drop(otx);

    // Coordinator: global seat control, fan-out to all shards, drain on Stop.
    let seats = args.seats;
    let chandle = thread::spawn(move || {
        let mut inflight = 0usize;
        let mut pending: VecDeque<Job> = VecDeque::new();
        let mut draining = false;
        loop {
            while inflight < seats {
                if let Some(j) = pending.pop_front() {
                    let comp = Arc::new(Comp {
                        partials: Mutex::new(Vec::new()),
                        remaining: std::sync::atomic::AtomicUsize::new(workers),
                        arrival: j.arrival,
                        qf: j.qf.clone(),
                    });
                    for ws in &worker_txs {
                        ws.send((comp.clone(), j.qbin.clone())).ok();
                    }
                    inflight += 1;
                } else {
                    break;
                }
            }
            if draining && pending.is_empty() && inflight == 0 {
                break;
            }
            match coord_rx.recv() {
                Ok(CoordMsg::Job(j)) => pending.push_back(j),
                Ok(CoordMsg::Done) => inflight -= 1,
                Ok(CoordMsg::Stop) => draining = true,
                Err(_) => break,
            }
        }
        drop(worker_txs);
    });

    // Dispatcher: Poisson arrivals.
    let start = Instant::now();
    let total = args.warmup + args.duration;
    let warmup_cutoff = Instant::now() + Duration::from_secs_f64(args.warmup);
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let np = pool.len();
    let mut sent = 0usize;
    let mut warmup_sent = 0usize;
    while start.elapsed().as_secs_f64() < total {
        let dt = -rng.unit().ln() / args.rate;
        thread::sleep(Duration::from_secs_f64(dt));
        let qi = (rng.next_u64() as usize) % np;
        let (qb, qf) = (pool[qi].0.clone(), pool[qi].1.clone());
        if Instant::now() < warmup_cutoff {
            warmup_sent += 1;
        }
        if coord_tx.send(CoordMsg::Job(Job { qbin: qb, qf, arrival: Instant::now() })).is_err() {
            break;
        }
        sent += 1;
    }
    coord_tx.send(CoordMsg::Stop).ok();
    drop(coord_tx);

    let mut lats: Vec<f64> = orx.iter().collect();
    chandle.join().ok();
    for h in whandles {
        h.join().ok();
    }
    let drop_n = warmup_sent.min(lats.len());
    let measured = lats.split_off(drop_n);
    let mut s = measured;
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let done = s.len();
    let thru = done as f64 / args.duration;
    let mean = if done > 0 { s.iter().sum::<f64>() / done as f64 } else { f64::NAN };
    println!(
        "{{\"mode\":\"sharded\",\"rate\":{:.0},\"seats\":{},\"chunk\":{},\"workers\":{},\"offered\":{},\"completed\":{},\"throughput\":{:.1},\"lat_ms\":{{\"mean\":{:.2},\"p50\":{:.2},\"p95\":{:.2},\"p99\":{:.2},\"max\":{:.2}}}}}",
        args.rate, args.seats, args.chunk, workers, sent, done, thru,
        mean, pct(&s, 50.0), pct(&s, 95.0), pct(&s, 99.0), pct(&s, 100.0)
    );
}

fn group_coordinator(
    coord_rx: Receiver<CoordMsg>,
    worker_txs: Vec<mpsc::Sender<(Arc<Comp>, Arc<Vec<u64>>)>>,
    seats: usize,
    fan: usize,
) {
    use std::collections::VecDeque;
    let mut inflight = 0usize;
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut draining = false;
    loop {
        while inflight < seats {
            if let Some(j) = pending.pop_front() {
                let comp = Arc::new(Comp {
                    partials: Mutex::new(Vec::new()),
                    remaining: std::sync::atomic::AtomicUsize::new(fan),
                    arrival: j.arrival,
                    qf: j.qf.clone(),
                });
                for ws in &worker_txs {
                    ws.send((comp.clone(), j.qbin.clone())).ok();
                }
                inflight += 1;
            } else {
                break;
            }
        }
        if draining && pending.is_empty() && inflight == 0 {
            break;
        }
        match coord_rx.recv() {
            Ok(CoordMsg::Job(j)) => pending.push_back(j),
            Ok(CoordMsg::Done) => inflight -= 1,
            Ok(CoordMsg::Stop) => draining = true,
            Err(_) => break,
        }
    }
    drop(worker_txs);
}

/// Generalized carousel: G = workers/fan independent groups, each group shards the
/// base across `fan` workers (revolution = N/fan) and serves its own queries.
fn run_grouped(
    args: &Args,
    bbase: Arc<quant::QuantBinary>,
    basef: Arc<fvecs::Vectors>,
    pool: Vec<(Arc<Vec<u64>>, Arc<Vec<f32>>)>,
    workers: usize,
) {
    let fan = args.fan.clamp(1, workers);
    let g = (workers / fan).max(1);
    let n = bbase.len();
    let span = n / fan;
    let (otx, orx) = mpsc::channel::<f64>();
    let mut disp_txs = Vec::with_capacity(g);
    let mut whandles = Vec::new();
    let mut chandles = Vec::new();
    for _grp in 0..g {
        let (coord_tx, coord_rx) = mpsc::channel::<CoordMsg>();
        let mut worker_txs = Vec::with_capacity(fan);
        for j in 0..fan {
            let lo = j * span;
            let hi = if j == fan - 1 { n } else { (j + 1) * span };
            let (wtx, wrx) = mpsc::channel::<(Arc<Comp>, Arc<Vec<u64>>)>();
            worker_txs.push(wtx);
            let (bb, bf, ct, ot) = (bbase.clone(), basef.clone(), coord_tx.clone(), otx.clone());
            let (seats, chunk, k, rc) = (args.seats, args.chunk, args.k, args.rerank);
            whandles.push(thread::spawn(move || {
                shard_worker(bb, bf, lo, hi, wrx, ct, seats, chunk, k, rc, ot);
            }));
        }
        let seats = args.seats;
        chandles.push(thread::spawn(move || group_coordinator(coord_rx, worker_txs, seats, fan)));
        disp_txs.push(coord_tx);
    }
    drop(otx);

    let start = Instant::now();
    let total = args.warmup + args.duration;
    let warmup_cutoff = Instant::now() + Duration::from_secs_f64(args.warmup);
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let np = pool.len();
    let mut sent = 0usize;
    let mut warmup_sent = 0usize;
    let mut rr = 0usize;
    while start.elapsed().as_secs_f64() < total {
        let dt = -rng.unit().ln() / args.rate;
        thread::sleep(Duration::from_secs_f64(dt));
        let qi = (rng.next_u64() as usize) % np;
        let (qb, qf) = (pool[qi].0.clone(), pool[qi].1.clone());
        if Instant::now() < warmup_cutoff {
            warmup_sent += 1;
        }
        let tx = &disp_txs[rr % g];
        rr += 1;
        if tx.send(CoordMsg::Job(Job { qbin: qb, qf, arrival: Instant::now() })).is_err() {
            break;
        }
        sent += 1;
    }
    for tx in &disp_txs {
        tx.send(CoordMsg::Stop).ok();
    }
    drop(disp_txs);

    let mut lats: Vec<f64> = orx.iter().collect();
    for h in chandles {
        h.join().ok();
    }
    for h in whandles {
        h.join().ok();
    }
    let drop_n = warmup_sent.min(lats.len());
    let measured = lats.split_off(drop_n);
    let mut s = measured;
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let done = s.len();
    let thru = done as f64 / args.duration;
    let mean = if done > 0 { s.iter().sum::<f64>() / done as f64 } else { f64::NAN };
    println!(
        "{{\"mode\":\"grouped\",\"fan\":{},\"groups\":{},\"rate\":{:.0},\"seats\":{},\"chunk\":{},\"workers\":{},\"offered\":{},\"completed\":{},\"throughput\":{:.1},\"lat_ms\":{{\"mean\":{:.2},\"p50\":{:.2},\"p95\":{:.2},\"p99\":{:.2},\"max\":{:.2}}}}}",
        fan, g, args.rate, args.seats, args.chunk, workers, sent, done, thru,
        mean, pct(&s, 50.0), pct(&s, 95.0), pct(&s, 99.0), pct(&s, 100.0)
    );
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let base = fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?;
    let queries = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let bbase = Arc::new(quant::QuantBinary::from_f32(&base));
    let basef = Arc::new(base);

    // Query pool: binary code (for scan) + f32 (for rerank).
    let pool: Vec<(Arc<Vec<u64>>, Arc<Vec<f32>>)> = (0..queries.len())
        .map(|i| {
            (
                Arc::new(quant::binarize_query(queries.row(i), 0)),
                Arc::new(queries.row(i).to_vec()),
            )
        })
        .collect();

    let workers = if args.workers == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        args.workers
    };

    if args.mode == "sharded" {
        run_sharded(&args, bbase, basef, pool, workers);
        return Ok(());
    }
    if args.mode == "grouped" {
        run_grouped(&args, bbase, basef, pool, workers);
        return Ok(());
    }

    let (tx, rx) = mpsc::channel::<Job>();
    let rx = Arc::new(Mutex::new(rx));
    let (otx, orx) = mpsc::channel::<f64>();

    let mut handles = Vec::new();
    for _ in 0..workers {
        let (bb, bf, rxc, otc) = (bbase.clone(), basef.clone(), rx.clone(), otx.clone());
        let (mode, seats, chunk, k, rc) =
            (args.mode.clone(), args.seats, args.chunk, args.k, args.rerank);
        handles.push(thread::spawn(move || {
            if mode == "carousel" {
                carousel_worker(bb, bf, rxc, seats, chunk, k, rc, otc);
            } else {
                perquery_worker(bb, bf, rxc, k, rc, otc);
            }
        }));
    }
    drop(otx);

    // Dispatcher: Poisson arrivals for warmup+duration. Tag each completion with
    // its arrival offset so we can keep only steady-state samples.
    let start = Instant::now();
    let total = args.warmup + args.duration;
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let np = pool.len();
    let mut sent = 0usize;
    // Record arrival offsets in a side map via the latency value alone won't carry
    // offset; instead gate by wall clock: collect all, then drop the first warmup
    // fraction by completion order is imprecise — so we timestamp arrivals here and
    // only *send* jobs during [0,total], and filter latencies by a warmup count.
    let warmup_cutoff = Instant::now() + Duration::from_secs_f64(args.warmup);
    let mut warmup_sent = 0usize;
    while start.elapsed().as_secs_f64() < total {
        let dt = -rng.unit().ln() / args.rate;
        thread::sleep(Duration::from_secs_f64(dt));
        let qi = (rng.next_u64() as usize) % np;
        let (qb, qf) = (pool[qi].0.clone(), pool[qi].1.clone());
        if Instant::now() < warmup_cutoff {
            warmup_sent += 1;
        }
        if tx.send(Job { qbin: qb, qf, arrival: Instant::now() }).is_err() {
            break;
        }
        sent += 1;
    }
    drop(tx); // workers drain then exit

    let mut lats: Vec<f64> = orx.iter().collect();
    for h in handles {
        let _ = h.join();
    }

    // Drop warmup samples (first ~warmup_sent completions).
    let drop_n = warmup_sent.min(lats.len());
    let measured: Vec<f64> = lats.split_off(drop_n);
    let mut s = measured.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let done = s.len();
    let thru = done as f64 / args.duration;
    let mean = if done > 0 { s.iter().sum::<f64>() / done as f64 } else { f64::NAN };

    println!(
        "{{\"mode\":\"{}\",\"rate\":{:.0},\"seats\":{},\"chunk\":{},\"workers\":{},\"offered\":{},\"completed\":{},\"throughput\":{:.1},\"lat_ms\":{{\"mean\":{:.2},\"p50\":{:.2},\"p95\":{:.2},\"p99\":{:.2},\"max\":{:.2}}}}}",
        args.mode, args.rate, args.seats, args.chunk, workers, sent, done, thru,
        mean, pct(&s, 50.0), pct(&s, 95.0), pct(&s, 99.0), pct(&s, 100.0)
    );
    Ok(())
}
