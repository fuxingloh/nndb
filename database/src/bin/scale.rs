//! scale.rs — direction 8: scale & the memory-hierarchy cliff.
//!
//! Push the binary scan to 100M vectors (codes only — an f32 store would be 400 GB)
//! and find where the working set leaves L3 for DRAM. Codes are 128 B/vector at 1024
//! bits, so L3 (480 MB here) holds ~3.75M vectors — the cliff should sit near N≈3-4M.
//! Question: does the funnel's "tiling amortizes the base read" property survive at
//! scale — i.e. once we're firmly DRAM-bound, does QPS still scale ~linearly with tile?
//!
//! Data is RANDOM binary codes (this is a SCAN-PERFORMANCE study, not recall): we build
//! QuantBinary directly so we never materialize f32.
//!
//! Measurement contract: the tiled kernel parallelizes ACROSS query-chunks (one chunk
//! of `tile` queries per rayon task), so to use all cores we need `cores` chunks. We set
//! **q = cores * tile** → exactly `cores` base-passes per config, all cores busy. Then:
//!   - bytes streamed = cores * base_bytes  (tile-INDEPENDENT) → GB/s flat across tile
//!     when memory-bound; GB/s drops from L3 BW to DRAM BW as N crosses L3 (the cliff).
//!   - QPS = q/secs scales ~linearly with tile when memory-bound (tiling wins), and
//!     saturates below tile× when compute-bound (small N in cache).

use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use vector_search::fvecs::Vectors;
use vector_search::quant::{knn_binary_funnel_tiled, QuantBinary};

#[derive(Parser)]
#[command(about = "Binary scan throughput vs N (the L3->DRAM cliff) and tiling at scale")]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "100000,1000000,3000000,10000000,30000000,100000000")]
    cells: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,8,16,32")]
    tiles: Vec<usize>,
    #[arg(long, default_value_t = 1024)]
    bits: usize,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 8)]
    cores: usize,
    /// minimum seconds to accumulate per timed config (repeats fast small-N configs)
    #[arg(long, default_value_t = 0.4)]
    min_secs: f64,
}

fn fill_random(buf: &mut [u64], seed: u64) {
    buf.par_iter_mut().enumerate().for_each(|(i, x)| {
        let mut z = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        *x = z ^ (z >> 31);
    });
}

fn main() {
    let args = Args::parse();
    let words = args.bits / 64;
    let empty = Vectors { data: vec![], dim: args.bits };
    eprintln!(
        "bits={} words={words} k={} cores={} tiles={:?}",
        args.bits, args.k, args.cores, args.tiles
    );

    let mut cell_json: Vec<String> = Vec::new();
    for &n in &args.cells {
        let base_bytes = (n * words * 8) as f64;
        eprintln!("\n===== N={n} ({:.2} GB codes) =====", base_bytes / 1e9);
        let mut data = vec![0u64; n * words];
        fill_random(&mut data, 0xABCDEF);
        let codes = QuantBinary { data, words, dim: args.bits };

        let mut pts: Vec<String> = Vec::new();
        for &tile in &args.tiles {
            let q = args.cores * tile; // q/tile == cores chunks → all cores busy
            let mut qd = vec![0u64; q * words];
            fill_random(&mut qd, 0x1234 + tile as u64);
            let qcodes = QuantBinary { data: qd, words, dim: args.bits };

            // warmup
            let _ = knn_binary_funnel_tiled(&codes, &qcodes, &empty, &empty, args.k, 0, tile, false, false);
            // measure: accumulate >= min_secs
            let mut iters = 0u32;
            let t = Instant::now();
            loop {
                let _ = knn_binary_funnel_tiled(&codes, &qcodes, &empty, &empty, args.k, 0, tile, false, false);
                iters += 1;
                if t.elapsed().as_secs_f64() >= args.min_secs {
                    break;
                }
            }
            let secs = t.elapsed().as_secs_f64() / iters as f64;

            let passes = args.cores as f64; // q/tile
            let qps = q as f64 / secs;
            let gbps = base_bytes * passes / secs / 1e9;
            let gcmp = q as f64 * n as f64 / secs / 1e9; // vector-comparisons/s
            eprintln!(
                "  tile={tile:<3} qps={qps:>10.1}  {gbps:>6.1} GB/s  {gcmp:>6.2} Gcmp/s  ({iters} it)"
            );
            pts.push(format!(
                "{{\"tile\":{tile},\"qps\":{qps:.1},\"gbps\":{gbps:.1},\"gcmp_per_s\":{gcmp:.3}}}"
            ));
        }
        let line = format!(
            "{{\"n\":{n},\"code_bytes\":{},\"gb\":{:.2},\"points\":[{}]}}",
            words * 8,
            base_bytes / 1e9,
            pts.join(",")
        );
        println!("{line}"); // emit per-N so partial results survive an OOM at the top end
        cell_json.push(line);
        // codes dropped here
    }
    eprintln!("\n=== one JSON object per N on stdout above ===");
    let _ = cell_json;
}
