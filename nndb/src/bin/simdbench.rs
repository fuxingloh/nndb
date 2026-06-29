//! simdbench.rs — SIMD exploration, step 1: how far do *safe, stable* Rust
//! autovectorization HINTS push the binary-scan Hamming kernel, vs the current
//! `count_ones` loop? (Before reaching for nightly `core::simd` or unsafe `std::arch`.)
//!
//! The hot path of the engine is Hamming over u64 words. The current kernel is a
//! naive `for i { d += (a[i]^b[i]).count_ones() }`, which the compiler already turns
//! into VPOPCNTDQ on AVX-512. History 012/023 showed hand-restructuring (multi-acc,
//! register-tiling) LOSES by defeating that autovectorization. This bench isolates
//! the KERNEL (no heap, data resident in cache) so we measure popcount throughput,
//! not memory or selection overhead. Single-thread Gcmp/s per variant.

use std::hint::black_box;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

#[derive(Parser)]
#[command(about = "Hamming kernel SIMD-hint microbenchmark (stable, safe)")]
struct Args {
    #[arg(long, default_value_t = 1024)]
    bits: usize,
    /// doc count — keep small so the set stays cache-resident (isolates compute)
    #[arg(long, default_value_t = 4096)]
    docs: usize,
    #[arg(long, default_value_t = 0.5)]
    secs: f64,
}

fn fill(buf: &mut [u64], seed: u64) {
    let mut z = seed;
    for x in buf.iter_mut() {
        z = z.wrapping_add(0x9E3779B97F4A7C15);
        let mut t = z;
        t = (t ^ (t >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        t = (t ^ (t >> 27)).wrapping_mul(0x94D049BB133111EB);
        *x = t ^ (t >> 31);
    }
}

// ---- per-doc Hamming kernels ----
#[inline(always)]
fn ham_baseline(a: &[u64], b: &[u64]) -> u32 {
    let mut d = 0u32;
    for i in 0..a.len() {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

#[inline(always)]
fn ham_iter(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

// hint: fixed 8-wide chunks (matches AVX-512 8×u64 VPOPCNTDQ), no remainder branch
// inside the hot loop.
#[inline(always)]
fn ham_chunks8(a: &[u64], b: &[u64]) -> u32 {
    let mut d = 0u32;
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        let mut acc = 0u32;
        for i in 0..8 {
            acc += (x[i] ^ y[i]).count_ones();
        }
        d += acc;
    }
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        d += (x ^ y).count_ones();
    }
    d
}

// the 012/023 "loser" reconfirmed on AVX-512: 4 manual accumulators over words.
#[inline(always)]
fn ham_acc4(a: &[u64], b: &[u64]) -> u32 {
    let mut acc = [0u32; 4];
    let n = a.len();
    let blocks = n / 4;
    for blk in 0..blocks {
        let o = blk * 4;
        for l in 0..4 {
            acc[l] += (a[o + l] ^ b[o + l]).count_ones();
        }
    }
    let mut tail = 0u32;
    for i in (blocks * 4)..n {
        tail += (a[i] ^ b[i]).count_ones();
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

// scan one query across all docs with the given kernel, summing (sink defeats DCE).
fn scan(q: &[u64], docs: &[u64], words: usize, n: usize, kern: fn(&[u64], &[u64]) -> u32) -> u64 {
    let mut sink = 0u64;
    for j in 0..n {
        sink += kern(q, &docs[j * words..(j + 1) * words]) as u64;
    }
    sink
}

// doc-interleaved scan: 4 independent per-doc Hammings in flight (ILP hint), each
// still a full vectorized count_ones — distinct from 023 (which interleaved QUERIES
// per word and forced scalar popcount).
fn scan_interleave4(q: &[u64], docs: &[u64], words: usize, n: usize) -> u64 {
    let mut s = [0u64; 4];
    let mut j = 0;
    while j + 4 <= n {
        let d0 = ham_baseline(q, &docs[(j) * words..(j + 1) * words]);
        let d1 = ham_baseline(q, &docs[(j + 1) * words..(j + 2) * words]);
        let d2 = ham_baseline(q, &docs[(j + 2) * words..(j + 3) * words]);
        let d3 = ham_baseline(q, &docs[(j + 3) * words..(j + 4) * words]);
        s[0] += d0 as u64;
        s[1] += d1 as u64;
        s[2] += d2 as u64;
        s[3] += d3 as u64;
        j += 4;
    }
    while j < n {
        s[0] += ham_baseline(q, &docs[j * words..(j + 1) * words]) as u64;
        j += 1;
    }
    s[0] + s[1] + s[2] + s[3]
}

// Tiled / carousel scan: each doc loaded once, compared against T queries (scan-
// sharing). Amortizes the doc read across T riders → per-query bandwidth = base/T.
// Each query's Hamming is still a full vectorized count_ones over the doc slice.
fn scan_tiled(qs: &[u64], docs: &[u64], words: usize, n: usize, t: usize) -> u64 {
    let mut sink = 0u64;
    for j in 0..n {
        let doc = &docs[j * words..(j + 1) * words];
        for q in 0..t {
            sink += ham_baseline(&qs[q * words..(q + 1) * words], doc) as u64;
        }
    }
    sink
}

/// Sharded carousel: docs split across `cores`; each core scans ITS shard once
/// against all T queries → the base is read ONCE total (no redundant streaming).
fn par_sharded(qs: &[u64], docs: &[u64], words: usize, n: usize, t: usize, cores: usize) -> u64 {
    let chunk = n.div_ceil(cores);
    (0..cores)
        .into_par_iter()
        .map(|c| {
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n);
            let mut sink = 0u64;
            for j in lo..hi {
                let doc = &docs[j * words..(j + 1) * words];
                for q in 0..t {
                    sink += ham_baseline(&qs[q * words..(q + 1) * words], doc) as u64;
                }
            }
            sink
        })
        .sum()
}

/// Tiled batch (the 047 model): queries split across `cores`; each core scans the
/// FULL base against its query subset → the base is streamed `cores`× (redundant).
fn par_query_split(qs: &[u64], docs: &[u64], words: usize, n: usize, t: usize, cores: usize) -> u64 {
    let qchunk = t.div_ceil(cores);
    (0..cores)
        .into_par_iter()
        .map(|c| {
            let qlo = c * qchunk;
            let qhi = ((c + 1) * qchunk).min(t);
            let mut sink = 0u64;
            for j in 0..n {
                let doc = &docs[j * words..(j + 1) * words];
                for q in qlo..qhi {
                    sink += ham_baseline(&qs[q * words..(q + 1) * words], doc) as u64;
                }
            }
            sink
        })
        .sum()
}

fn time_it<F: FnMut() -> u64>(secs: f64, mut f: F) -> (f64, u64) {
    // warmup
    black_box(f());
    let mut iters = 0u64;
    let mut sink = 0u64;
    let t = Instant::now();
    loop {
        sink = sink.wrapping_add(f());
        iters += 1;
        if t.elapsed().as_secs_f64() >= secs {
            break;
        }
    }
    (t.elapsed().as_secs_f64() / iters as f64, sink)
}

fn main() {
    let args = Args::parse();
    let words = args.bits / 64;
    let n = args.docs;
    let mut docs = vec![0u64; n * words];
    let mut q = vec![0u64; words];
    fill(&mut docs, 0xABCDEF);
    fill(&mut q, 0x1234);
    eprintln!(
        "bits={} words={words} docs={n} ({} KB resident) single-thread",
        args.bits,
        n * words * 8 / 1024
    );

    let variants: [(&str, fn(&[u64], &[u64]) -> u32); 4] = [
        ("baseline(count_ones)", ham_baseline),
        ("iter_zip_sum", ham_iter),
        ("chunks_exact8", ham_chunks8),
        ("manual_acc4 (012 loser)", ham_acc4),
    ];
    let mut out = Vec::new();
    for (name, kern) in variants {
        let (secs, sink) = time_it(args.secs, || scan(&q, &docs, words, n, kern));
        let gcmp = n as f64 / secs / 1e9;
        eprintln!("  {name:<26} {gcmp:>6.3} Gcmp/s  (sink={sink})");
        out.push(format!("{{\"variant\":\"{name}\",\"gcmp_per_s\":{gcmp:.4}}}"));
    }
    let (secs, sink) = time_it(args.secs, || scan_interleave4(&q, &docs, words, n));
    let gcmp = n as f64 / secs / 1e9;
    eprintln!("  {:<26} {gcmp:>6.3} Gcmp/s  (sink={sink})", "interleave4 (ILP)");
    out.push(format!("{{\"variant\":\"interleave4\",\"gcmp_per_s\":{gcmp:.4}}}"));

    // --- scan-sharing (carousel/tiled) sweep: T queries share each doc read ---
    // If bandwidth-bound, Gcmp/s rises with T (amortization) then plateaus at the
    // compute (popcount) ceiling. The plateau height vs the in-cache ceiling tells us
    // whether the shared scan is popcount-limited (SIMD already maxed) or has headroom.
    let mut tq = vec![0u64; 64 * words];
    fill(&mut tq, 0x55AA);
    eprintln!("  -- scan-sharing T-sweep (Gcmp/s = T*n/s) --");
    let mut tout = Vec::new();
    for t in [1usize, 4, 8, 16, 32, 64] {
        let (secs, sink) = time_it(args.secs, || scan_tiled(&tq, &docs, words, n, t));
        let gcmp = (t * n) as f64 / secs / 1e9;
        eprintln!("     T={t:<3} {gcmp:>6.3} Gcmp/s  (sink={sink})");
        tout.push(format!("{{\"t\":{t},\"gcmp_per_s\":{gcmp:.4}}}"));
    }
    out.push(format!("{{\"tiled_sweep\":[{}]}}", tout.join(",")));

    // --- multi-core: sharded carousel (base read once) vs tiled batch (base read
    // cores×). At scale this is the carousel's throughput advantage. ---
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(8);
    let t = 16usize;
    eprintln!("  -- multi-core ({cores} cores), T={t}, N={n} --");
    let (secs, sink) = time_it(args.secs, || par_sharded(&tq, &docs, words, n, t, cores));
    let g_sh = (t * n) as f64 / secs / 1e9;
    eprintln!("     sharded (base x1)     {g_sh:>6.3} Gcmp/s  (sink={sink})");
    let (secs, sink) = time_it(args.secs, || par_query_split(&tq, &docs, words, n, t, cores));
    let g_qs = (t * n) as f64 / secs / 1e9;
    eprintln!("     query-split (base x{cores})  {g_qs:>6.3} Gcmp/s  (sink={sink})");
    out.push(format!(
        "{{\"multicore\":{{\"cores\":{cores},\"t\":{t},\"sharded_gcmp\":{g_sh:.3},\"query_split_gcmp\":{g_qs:.3}}}}}"
    ));

    println!(
        "{{\"bits\":{},\"words\":{words},\"docs\":{n},\"thread\":\"single\",\"variants\":[{}]}}",
        args.bits,
        out.join(",")
    );
}
