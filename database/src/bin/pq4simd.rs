//! pq4simd.rs — Task #2: can a SIMD ADC scan beat popcount? (the question parked in 050/054)
//!
//! Scalar PQ-ADC lost to popcount (054) because its M table lookups are scalar gathers.
//! The FAISS PQ4 / ScaNN trick: 4-bit codes (16 centroids/subspace) + an int8 LUT that
//! fits in a register, looked up with `pshufb` — 16 lookups in ONE instruction. In safe
//! Rust that's `core::simd::Simd::swizzle_dyn` (lowers to pshufb on x86 / tbl on ARM).
//!
//! This microbench compares single-thread scan throughput (Gcmp/s) of:
//!   - popcount  : 1024-bit binary, count_ones (VPOPCNTDQ) — the funnel's scan, 128 B/vec
//!   - scalar-PQ : M=16 byte codes, scalar ADC (M lookups/doc) — 16 B/vec (the 054 loser)
//!   - pq4-simd  : M=16 nibble codes, swizzle_dyn LUT, 16 docs/instruction — 8 B/vec
//! Random data; isolates the kernel. If pq4-simd beats popcount, it's the first thing to.

#![feature(portable_simd)]
use std::hint::black_box;
use std::simd::num::SimdUint;
use std::simd::Simd;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

type U8x16 = Simd<u8, 16>;

#[derive(Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "100000,1000000,10000000")]
    n: Vec<usize>,
    #[arg(long, default_value_t = 16)]
    m: usize, // subspaces (also bytes for scalar; M/2 bytes for pq4)
    #[arg(long, default_value_t = 1024)]
    bits: usize, // for the popcount baseline
    #[arg(long, default_value_t = 0.5)]
    secs: f64,
}

fn fill_u8(buf: &mut [u8], seed: u64) {
    let mut z = seed;
    for x in buf.iter_mut() {
        z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = (z >> 33) as u8;
    }
}
fn fill_u64(buf: &mut [u64], seed: u64) {
    let mut z = seed;
    for x in buf.iter_mut() {
        z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = z;
    }
}

fn time<F: FnMut() -> u64>(secs: f64, mut f: F) -> (f64, u64) {
    black_box(f());
    let (mut it, mut sink) = (0u64, 0u64);
    let t = Instant::now();
    loop {
        sink = sink.wrapping_add(f());
        it += 1;
        if t.elapsed().as_secs_f64() >= secs { break; }
    }
    (t.elapsed().as_secs_f64() / it as f64, sink)
}

fn main() {
    let args = Args::parse();
    let words = args.bits / 64;
    let m = args.m;
    eprintln!("M={m} bits={} single-thread", args.bits);

    for &n in &args.n {
        // ---- popcount baseline ----
        let mut codes = vec![0u64; n * words];
        fill_u64(&mut codes, 0xA1);
        let mut q = vec![0u64; words];
        fill_u64(&mut q, 0xB2);
        let (s_pc, _) = time(args.secs, || {
            let mut sink = 0u64;
            for i in 0..n {
                let c = &codes[i * words..(i + 1) * words];
                let mut h = 0u32;
                for w in 0..words { h += (c[w] ^ q[w]).count_ones(); }
                sink += h as u64;
            }
            sink
        });
        let g_pc = n as f64 / s_pc / 1e9;
        drop(codes);

        // ---- scalar PQ-ADC (M byte codes, M*256 f32 table) ----
        let mut pcodes = vec![0u8; n * m];
        fill_u8(&mut pcodes, 0xC3);
        let mut tab = vec![0f32; m * 256];
        for (i, t) in tab.iter_mut().enumerate() { *t = (i % 97) as f32 * 0.013; }
        let (s_sc, _) = time(args.secs, || {
            let mut sink = 0u64;
            for i in 0..n {
                let c = &pcodes[i * m..(i + 1) * m];
                let mut d = 0f32;
                for s in 0..m { d += tab[s * 256 + c[s] as usize]; }
                sink += d as u64;
            }
            sink
        });
        let g_sc = n as f64 / s_sc / 1e9;
        drop(pcodes);

        // ---- pq4 SIMD (4-bit codes, blocks of 16 vectors, u8 LUT, swizzle_dyn) ----
        let nb = n.div_ceil(16);
        let np = nb * 16;
        // layout: codes4[blk*m*16 + s*16 + v] = nibble of vector v, subspace s
        let mut codes4 = vec![0u8; nb * m * 16];
        fill_u8(&mut codes4, 0xD4);
        for x in codes4.iter_mut() { *x &= 0x0F; } // 4-bit
        let mut lut = vec![0u8; m * 16];
        fill_u8(&mut lut, 0xE5);
        let (s_p4, _) = time(args.secs, || {
            let mut sink = 0u64;
            for b in 0..nb {
                let base = b * m * 16;
                let mut acc = U8x16::splat(0);
                for s in 0..m {
                    let cs = U8x16::from_slice(&codes4[base + s * 16..base + s * 16 + 16]);
                    let ls = U8x16::from_slice(&lut[s * 16..s * 16 + 16]);
                    acc = acc.saturating_add(ls.swizzle_dyn(cs));
                }
                sink += acc.reduce_max() as u64;
            }
            sink
        });
        let g_p4 = np as f64 / s_p4 / 1e9;
        drop(codes4);

        // ---- 8-core parallel (bandwidth regime): does pq4's 16x smaller footprint win? ----
        let mut codes_pc = vec![0u64; n * words];
        fill_u64(&mut codes_pc, 0xA1);
        let (s_pcp, _) = time(args.secs, || {
            codes_pc.par_chunks(65536 * words).map(|chunk| {
                let mut sink = 0u64;
                for c in chunk.chunks_exact(words) {
                    let mut h = 0u32;
                    for w in 0..words { h += (c[w] ^ q[w]).count_ones(); }
                    sink += h as u64;
                }
                sink
            }).sum()
        });
        let g_pcp = n as f64 / s_pcp / 1e9;
        drop(codes_pc);
        let mut codes4p = vec![0u8; nb * m * 16];
        fill_u8(&mut codes4p, 0xD4);
        for x in codes4p.iter_mut() { *x &= 0x0F; }
        let (s_p4p, _) = time(args.secs, || {
            codes4p.par_chunks(4096 * m * 16).map(|chunk| {
                let mut sink = 0u64;
                for blk in chunk.chunks_exact(m * 16) {
                    let mut acc = U8x16::splat(0);
                    for s in 0..m {
                        let cs = U8x16::from_slice(&blk[s * 16..s * 16 + 16]);
                        let ls = U8x16::from_slice(&lut[s * 16..s * 16 + 16]);
                        acc = acc.saturating_add(ls.swizzle_dyn(cs));
                    }
                    sink += acc.reduce_min() as u64;
                }
                sink
            }).sum()
        });
        let g_p4p = np as f64 / s_p4p / 1e9;
        drop(codes4p);

        eprintln!(
            "N={n:>9} | 1-thread: popcount(128B)={g_pc:.2} pq4(8B)={g_p4:.2} ({:.2}x) | 8-core: popcount={g_pcp:.2} pq4={g_p4p:.2} ({:.2}x) Gcmp/s",
            g_p4 / g_pc, g_p4p / g_pcp
        );
        println!(
            "{{\"n\":{n},\"st_popcount\":{g_pc:.3},\"st_scalarpq\":{g_sc:.3},\"st_pq4\":{g_p4:.3},\"mt_popcount\":{g_pcp:.3},\"mt_pq4\":{g_p4p:.3}}}"
        );
    }
}
