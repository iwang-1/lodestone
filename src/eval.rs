//! Recall measurement against the exact oracle.
//!
//! `recall@k` for a single query is `|approx_topk ∩ exact_topk| / k`. The
//! reported recall is the mean over a query set. This is the standard
//! ann-benchmarks definition and is self-verifying: the ground truth is the
//! brute-force result on the same corpus, so the number cannot be inflated.

use std::collections::HashSet;

/// Mean recall@k of `approx` result lists against `exact` result lists.
/// Both are slices of (id, distance); only ids are compared.
pub fn recall_at_k(exact: &[Vec<(u32, f32)>], approx: &[Vec<(u32, f32)>], k: usize) -> f64 {
    assert_eq!(exact.len(), approx.len());
    if exact.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (e, a) in exact.iter().zip(approx.iter()) {
        let truth: HashSet<u32> = e.iter().take(k).map(|x| x.0).collect();
        let hit = a.iter().take(k).filter(|x| truth.contains(&x.0)).count();
        total += hit as f64 / k as f64;
    }
    total / exact.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_recall() {
        let e = vec![vec![(1, 0.0), (2, 1.0), (3, 2.0)]];
        let a = vec![vec![(1, 0.0), (2, 1.0), (3, 2.0)]];
        assert!((recall_at_k(&e, &a, 3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn half_recall() {
        let e = vec![vec![(1, 0.0), (2, 1.0)]];
        let a = vec![vec![(1, 0.0), (9, 1.0)]];
        assert!((recall_at_k(&e, &a, 2) - 0.5).abs() < 1e-9);
    }
}
