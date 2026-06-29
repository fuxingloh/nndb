//! disk.rs — economics of disk-resident vectors: how much RAM do we actually need,
//! and how slow is it when the vectors are NOT in RAM?
//!
//! The motivation is cost, not speed: RAM is the expensive resource. This measures
//! the spectrum from "everything in RAM" to "nothing in RAM":
//!
//!   exact-ram     full f32 in RAM, brute force                 (the matrix-search baseline)
//!   exact-disk    full f32 mmap'd from disk, brute force       (the "nothing in RAM" case)
//!   funnel-ram    1-bit codes + f32 both in RAM                (our in-memory winner)
//!   funnel-hybrid 1-bit codes in RAM, f32 on disk (mmap)       (DiskANN-style split)
//!   funnel-disk   codes AND f32 mmap'd from disk               (nothing in RAM, funnel)
//!
//! Exact search touches every vector per query, so from disk it re-streams the whole
//! dataset every query → disk-bandwidth bound (seconds). The funnel only streams the
//! tiny codes for the scan and reads C random vectors for rerank — so the big f32
//! can live on cheap SSD. "cold" = page cache dropped by the driver before the run
//! (sudo); "warm" = run again with the file cached. Run one mode per process so the
//! driver controls the cache state.

use std::collections::BinaryHeap;
use std::fs::File;
use std::time::Instant;

use clap::Parser;
use memmap2::{Advice, Mmap};

use nndb::fvecs::{self, Vectors};
use nndb::quant::{self, hamming, QuantBinary, Rotation};
use nndb::search::l2_sq;

#[derive(Parser)]
#[command(about = "Economics of disk-resident vectors: RAM footprint vs latency")]
struct Args {
    #[arg(long, default_value = "data/cohere")]
    data: std::path::PathBuf,
    #[arg(long, default_value = "cohere")]
    prefix: String,
    #[arg(long)]
    mode: String, // exact-ram | exact-disk | funnel-ram | funnel-hybrid | funnel-disk
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// queries to time (keep small for cold exact — each is seconds)
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    /// funnel rerank width
    #[arg(long, default_value_t = 200)]
    c: usize,
    #[arg(long, default_value_t = 2)]
    rotate: usize,
}

/// Memory-map a raw little-endian f32 file as &[f32] (row-major, `dim` per row).
struct DiskVectors {
    _mmap: Mmap,
    ptr: *const f32,
    n: usize,
    dim: usize,
}
unsafe impl Sync for DiskVectors {}
impl DiskVectors {
    fn open(path: &std::path::Path, dim: usize, sequential: bool) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let _ = mmap.advise(if sequential { Advice::Sequential } else { Advice::Random });
        let bytes: &[u8] = &mmap;
        let ptr = bytes.as_ptr() as *const f32;
        let n = bytes.len() / 4 / dim;
        Ok(DiskVectors { _mmap: mmap, ptr, n, dim })
    }
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr.add(i * self.dim), self.dim) }
    }
}

/// Write a raw flat f32 file (no per-record headers) so row i is at i*dim*4.
fn ensure_flat(base: &Vectors, path: &std::path::Path) -> std::io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(base.data.as_ptr() as *const u8, base.data.len() * 4)
    };
    std::fs::write(path, bytes)
}

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let queries = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let nq = if args.queries == 0 { queries.len() } else { args.queries.min(queries.len()) };

    // peek dim/n from the fvecs header without loading (cheap: read first 4 bytes + size)
    let base_path = args.data.join(format!("{p}_base.fvecs"));
    let flat_path = args.data.join(format!("{p}.f32raw"));

    // We need the base loaded for: building codes, the RAM modes, and to create the
    // flat file the first time. Load it (this is RAM; the disk modes drop it after).
    let base = fvecs::read_fvecs(&base_path)?;
    let (n, dim) = (base.len(), base.dim);
    ensure_flat(&base, &flat_path)?;

    let codes_mb = (n * dim.div_ceil(64) * 8) as f64 / 1e6;
    let f32_mb = (n * dim * 4) as f64 / 1e6;
    eprintln!(
        "mode={} n={n} dim={dim} nq={nq} k={} c={} | codes={codes_mb:.0}MB f32={f32_mb:.0}MB",
        args.mode, args.k, args.c
    );

    let t0 = Instant::now();
    let (ram_mb, mut lats): (f64, Vec<f64>) = match args.mode.as_str() {
        // ---------- EXACT (touches every vector per query) ----------
        "exact-ram" => {
            let mut lats = Vec::with_capacity(nq);
            for q in 0..nq {
                let qv = queries.row(q);
                let t = Instant::now();
                let mut heap: BinaryHeap<(F, u32)> = BinaryHeap::with_capacity(args.k + 1);
                for i in 0..n {
                    let d = F(l2_sq(qv, base.row(i)));
                    if heap.len() < args.k {
                        heap.push((d, i as u32));
                    } else if d < heap.peek().unwrap().0 {
                        heap.pop();
                        heap.push((d, i as u32));
                    }
                }
                std::hint::black_box(&heap);
                lats.push(t.elapsed().as_secs_f64() * 1e3);
            }
            (f32_mb, lats)
        }
        "exact-disk" => {
            drop(base); // free the RAM copy; read from disk only
            let disk = DiskVectors::open(&flat_path, dim, true)?;
            let mut lats = Vec::with_capacity(nq);
            for q in 0..nq {
                let qv = queries.row(q);
                let t = Instant::now();
                let mut heap: BinaryHeap<(F, u32)> = BinaryHeap::with_capacity(args.k + 1);
                for i in 0..disk.n {
                    let d = F(l2_sq(qv, disk.row(i)));
                    if heap.len() < args.k {
                        heap.push((d, i as u32));
                    } else if d < heap.peek().unwrap().0 {
                        heap.pop();
                        heap.push((d, i as u32));
                    }
                }
                std::hint::black_box(&heap);
                lats.push(t.elapsed().as_secs_f64() * 1e3);
            }
            (0.0, lats)
        }
        // ---------- FUNNEL (scan codes, rerank C vectors) ----------
        "funnel-ram" => {
            let rot = Rotation::new(dim, args.rotate, 0xC0FFEE);
            let codes = QuantBinary::from_f32_rotated(&base, &rot, 0);
            let qbins: Vec<Vec<u64>> =
                (0..nq).map(|q| quant::binarize_query_rotated(queries.row(q), &rot, 0)).collect();
            let mut lats = Vec::with_capacity(nq);
            for q in 0..nq {
                let t = Instant::now();
                let cands = scan_topc(&codes, &qbins[q], args.c.max(args.k));
                let _ = rerank_ram(&base, queries.row(q), &cands, args.k);
                lats.push(t.elapsed().as_secs_f64() * 1e3);
            }
            (codes_mb + f32_mb, lats)
        }
        "funnel-hybrid" => {
            let rot = Rotation::new(dim, args.rotate, 0xC0FFEE);
            let codes = QuantBinary::from_f32_rotated(&base, &rot, 0); // codes stay in RAM
            let qbins: Vec<Vec<u64>> =
                (0..nq).map(|q| quant::binarize_query_rotated(queries.row(q), &rot, 0)).collect();
            drop(base); // f32 comes from disk now
            let disk = DiskVectors::open(&flat_path, dim, false)?; // random advice
            let mut lats = Vec::with_capacity(nq);
            for q in 0..nq {
                let t = Instant::now();
                let cands = scan_topc(&codes, &qbins[q], args.c.max(args.k));
                let _ = rerank_disk(&disk, queries.row(q), &cands, args.k);
                lats.push(t.elapsed().as_secs_f64() * 1e3);
            }
            (codes_mb, lats)
        }
        "funnel-disk" => {
            // codes AND f32 on disk: write codes flat too, mmap both. Nothing in RAM.
            let rot = Rotation::new(dim, args.rotate, 0xC0FFEE);
            let codes_path = args.data.join(format!("{p}.codesraw"));
            if !codes_path.exists() {
                let codes = QuantBinary::from_f32_rotated(&base, &rot, 0);
                let bytes = unsafe {
                    std::slice::from_raw_parts(codes.data.as_ptr() as *const u8, codes.data.len() * 8)
                };
                std::fs::write(&codes_path, bytes)?;
            }
            let words = dim.div_ceil(64);
            let qbins: Vec<Vec<u64>> =
                (0..nq).map(|q| quant::binarize_query_rotated(queries.row(q), &rot, 0)).collect();
            drop(base);
            let cfile = File::open(&codes_path)?;
            let cmap = unsafe { Mmap::map(&cfile)? };
            let _ = cmap.advise(Advice::Sequential);
            let cptr = cmap.as_ptr() as *const u64;
            let disk = DiskVectors::open(&flat_path, dim, false)?;
            let mut lats = Vec::with_capacity(nq);
            for q in 0..nq {
                let t = Instant::now();
                // scan codes from disk
                let qb = &qbins[q];
                let want = args.c.max(args.k);
                let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(want + 1);
                for i in 0..n {
                    let row = unsafe { std::slice::from_raw_parts(cptr.add(i * words), words) };
                    let h = hamming(qb, row);
                    if heap.len() < want {
                        heap.push((h, i as u32));
                    } else if h < heap.peek().unwrap().0 {
                        heap.pop();
                        heap.push((h, i as u32));
                    }
                }
                let cands: Vec<u32> = heap.into_iter().map(|(_, i)| i).collect();
                let _ = rerank_disk(&disk, queries.row(q), &cands, args.k);
                lats.push(t.elapsed().as_secs_f64() * 1e3);
            }
            (0.0, lats)
        }
        m => {
            eprintln!("unknown mode {m}");
            std::process::exit(2);
        }
    };

    let wall = t0.elapsed().as_secs_f64();
    let first_ms = *lats.first().unwrap(); // coldest: nothing cached yet
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = pct(lats.clone(), 0.50);
    let p99 = pct(lats.clone(), 0.99);
    let mean = lats.iter().sum::<f64>() / lats.len() as f64;
    let qps = nq as f64 / wall;
    eprintln!(
        "  ram_resident={ram_mb:.0}MB  first={first_ms:.3}ms p50={p50:.3}ms p99={p99:.3}ms mean={mean:.3}ms  serial_qps={qps:.1}"
    );
    println!(
        "{{\"mode\":\"{}\",\"n\":{n},\"dim\":{dim},\"nq\":{nq},\"k\":{},\"c\":{},\
         \"ram_mb\":{ram_mb:.0},\"codes_mb\":{codes_mb:.0},\"f32_mb\":{f32_mb:.0},\
         \"first_ms\":{first_ms:.3},\"p50_ms\":{p50:.3},\"p99_ms\":{p99:.3},\"mean_ms\":{mean:.3},\"serial_qps\":{qps:.1}}}",
        args.mode, args.k, args.c
    );
    Ok(())
}

// total-order f32 wrapper for the exact heap
#[derive(PartialEq)]
struct F(f32);
impl Eq for F {}
impl PartialOrd for F {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for F {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn scan_topc(codes: &QuantBinary, qb: &[u64], want: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(want + 1);
    for i in 0..codes.len() {
        let h = hamming(qb, codes.row(i));
        if heap.len() < want {
            heap.push((h, i as u32));
        } else if h < heap.peek().unwrap().0 {
            heap.pop();
            heap.push((h, i as u32));
        }
    }
    heap.into_iter().map(|(_, i)| i).collect()
}

fn rerank_ram(base: &Vectors, q: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
    let mut s: Vec<(f32, u32)> =
        cands.iter().map(|&c| (l2_sq(q, base.row(c as usize)), c)).collect();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    s.into_iter().take(k).map(|(_, i)| i).collect()
}

fn rerank_disk(disk: &DiskVectors, q: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
    let mut s: Vec<(f32, u32)> =
        cands.iter().map(|&c| (l2_sq(q, disk.row(c as usize)), c)).collect();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    s.into_iter().take(k).map(|(_, i)| i).collect()
}
