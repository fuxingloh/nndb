//! How many of the base's binary codes are unique? Collisions = vectors the 1-bit
//! scan cannot distinguish (identical Hamming distance to every query) — only the
//! rerank can separate them. Builds the same codes as the shipped funnel:
//! residual (centroid subtraction) → rotation ×2 (seed 0x5EED) → sign bits.
//!
//! Usage: cargo run --release --example count_unique_codes -- data/oai oai256 [--no-residual]

use std::collections::HashMap;

use nndb::fvecs;
use nndb::quant::{self, QuantBinary, Rotation};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = args.get(1).map(String::as_str).unwrap_or("data/oai");
    let prefix = args.get(2).map(String::as_str).unwrap_or("oai256");
    let residual = !args.iter().any(|a| a == "--no-residual");

    let base = fvecs::read_fvecs(std::path::Path::new(data).join(format!("{prefix}_base.fvecs")))?;
    let n = base.len();
    let bits = base.dim;
    eprintln!("n={n} dim={bits} residual={residual}");

    let rot = Rotation::new(base.dim, 2, 0x5EED);
    let codes: QuantBinary = if residual {
        let c = quant::centroid(&base);
        QuantBinary::from_f32_rotated(&quant::subtract_centroid(&base, &c), &rot, bits)
    } else {
        QuantBinary::from_f32_rotated(&base, &rot, bits)
    };

    // count exact duplicate codes
    let mut seen: HashMap<&[u64], u32> = HashMap::with_capacity(n);
    for i in 0..n {
        *seen.entry(codes.row(i)).or_insert(0) += 1;
    }
    let unique = seen.len();
    let dup_groups = seen.values().filter(|&&c| c > 1).count();
    let dup_vectors: u64 = seen.values().filter(|&&c| c > 1).map(|&c| c as u64).sum();
    let max_group = seen.values().copied().max().unwrap_or(0);

    println!(
        "{{\"n\":{n},\"bits\":{bits},\"residual\":{residual},\"unique_codes\":{unique},\
         \"unique_frac\":{:.6},\"collision_groups\":{dup_groups},\
         \"vectors_in_collisions\":{dup_vectors},\"largest_group\":{max_group}}}",
        unique as f64 / n as f64
    );
    Ok(())
}
