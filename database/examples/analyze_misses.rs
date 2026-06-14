//! Diagnostic: why is exact-search recall not 1.0 on SIFT1M?
//! Hypothesis: distances are exact integers, and every "miss" is a boundary tie.
//! Run: cargo run --release --example analyze_misses

use std::collections::HashSet;
use vector_search::{fvecs, search};

fn main() {
    let dir = "data/sift";
    let base = fvecs::read_fvecs(format!("{dir}/sift_base.fvecs")).unwrap();
    let mut query = fvecs::read_fvecs(format!("{dir}/sift_query.fvecs")).unwrap();
    let gt = fvecs::read_ivecs(format!("{dir}/sift_groundtruth.ivecs")).unwrap();
    let k = 10;

    // Use a subset so the brute-force pass is quick.
    let nq = 1000;
    query.data.truncate(nq * query.dim);

    // 1) Are the stored vector components actually integers (uint8 descriptors)?
    let non_integer = base
        .data
        .iter()
        .take(nq * base.dim)
        .filter(|v| v.fract() != 0.0)
        .count();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in base.data.iter().take(nq * base.dim) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    println!("component check (first {nq} vectors):");
    println!("  non-integer components: {non_integer}");
    println!("  value range: [{lo}, {hi}]  -> distances are sums of squared ints, exact in f32\n");

    let found = search::knn_batch(&base, &query, k);

    let mut total_miss = 0usize;
    let mut tie_explained = 0usize;
    let mut shown = 0;

    for q in 0..nq {
        let qv = query.row(q);
        let truth: Vec<u32> = gt.row(q)[..k].iter().map(|&x| x as u32).collect();
        let truth_set: HashSet<u32> = truth.iter().copied().collect();
        let got_set: HashSet<u32> = found[q].iter().copied().collect();

        for &t in &truth {
            if got_set.contains(&t) {
                continue;
            }
            total_miss += 1;
            let dist_missed = search::l2_sq(qv, base.row(t as usize));

            // Did we return a *different* point at the exact same distance?
            // If so, the miss is a tie at the k-boundary, not a wrong answer.
            let tie_partner = found[q].iter().find(|&&g| {
                !truth_set.contains(&g) && search::l2_sq(qv, base.row(g as usize)) == dist_missed
            });
            if let Some(&g) = tie_partner {
                tie_explained += 1;
                if shown < 5 {
                    shown += 1;
                    println!(
                        "q{q}: GT id {t} and our id {g} BOTH at distance {dist_missed} (tie at slot k)"
                    );
                }
            }
        }
    }

    let slots = nq * k;
    println!("\nmissed neighbor slots: {total_miss} / {slots}");
    println!("explained by exact-distance ties: {tie_explained} / {total_miss}");
    println!("recall@{k} = {:.6}", 1.0 - total_miss as f64 / slots as f64);
}
