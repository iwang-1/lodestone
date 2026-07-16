//! Exact brute-force index. This is the recall ground-truth oracle: HNSW and
//! IVF-PQ recall@k is defined against the exact top-k this returns.

use crate::distance::{DistanceFn, Metric};
use rayon::prelude::*;

pub struct BruteForce {
    dim: usize,
    dist: DistanceFn,
    vectors: Vec<f32>,
    n: usize,
}

impl BruteForce {
    pub fn new(dim: usize, metric: Metric) -> Self {
        BruteForce {
            dim,
            dist: DistanceFn::new(metric),
            vectors: Vec::new(),
            n: 0,
        }
    }

    pub fn add(&mut self, vector: &[f32]) -> u32 {
        assert_eq!(vector.len(), self.dim);
        let id = self.n as u32;
        self.vectors.extend_from_slice(vector);
        self.n += 1;
        id
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Exact top-k, computed in parallel across the corpus.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim);
        let dim = self.dim;
        let dist = self.dist;
        let mut scored: Vec<(u32, f32)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let v = &self.vectors[i * dim..(i + 1) * dim];
                (i as u32, dist.distance(v, query))
            })
            .collect();
        scored.par_sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }
}
