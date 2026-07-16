//! IVF-PQ index: an inverted file over a coarse k-means quantizer, with each
//! posting list storing PQ codes. Query scans the `nprobe` nearest coarse
//! cells and scores their codes via ADC, giving large memory savings (m bytes
//! per vector instead of 4*dim) at a tunable recall cost.

use crate::distance::{DistanceFn, Metric};
use crate::quant::pq::ProductQuantizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64;

pub struct IvfPq {
    dim: usize,
    metric: Metric,
    dist: DistanceFn,
    /// Coarse centroids: `nlist * dim`.
    coarse: Vec<f32>,
    nlist: usize,
    pq: ProductQuantizer,
    /// Posting lists: `lists[cell]` = (vector id, pq codes).
    lists: Vec<Vec<(u32, Vec<u8>)>>,
    /// Full-precision vectors kept for optional exact re-ranking of the ADC
    /// shortlist (the standard IVFPQ+R recall recovery). Row `id` is
    /// `raw[id*dim..]`.
    raw: Vec<f32>,
    /// Multiplier on k for the ADC shortlist before exact re-ranking.
    rerank_factor: usize,
    n: usize,
}

impl IvfPq {
    /// Trains the coarse quantizer and PQ codebooks on `training`.
    pub fn train(
        dim: usize,
        nlist: usize,
        m: usize,
        training: &[f32],
        metric: Metric,
        seed: u64,
    ) -> Self {
        let n = training.len() / dim;
        assert!(n >= nlist, "need >= nlist training vectors");
        let mut rng = Pcg64::seed_from_u64(seed);

        // Coarse quantizer: random-sample seeded k-means over full vectors.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);
        let mut coarse = vec![0.0f32; nlist * dim];
        for c in 0..nlist {
            let src = idx[c % n] * dim;
            coarse[c * dim..(c + 1) * dim].copy_from_slice(&training[src..src + dim]);
        }
        let dist = DistanceFn::new(metric);
        lloyd_full(&training[..n * dim], dim, &mut coarse, nlist, 10, &dist);

        let pq = ProductQuantizer::train(dim, m, training, metric, seed ^ 0xABCD);

        IvfPq {
            dim,
            metric,
            dist,
            coarse,
            nlist,
            pq,
            lists: vec![Vec::new(); nlist],
            raw: Vec::new(),
            rerank_factor: 8,
            n: 0,
        }
    }

    /// Sets how many ADC candidates (as a multiple of k) are exact-re-ranked.
    /// `0` disables re-ranking (pure ADC). Default 8.
    pub fn set_rerank_factor(&mut self, f: usize) {
        self.rerank_factor = f;
    }

    #[inline]
    fn raw_of(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.raw[i..i + self.dim]
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn compression_ratio(&self) -> f32 {
        self.pq.compression_ratio()
    }

    fn nearest_cell(&self, v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..self.nlist {
            let cent = &self.coarse[c * self.dim..(c + 1) * self.dim];
            let d = self.dist.distance(cent, v);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }

    /// Adds a vector: assign to nearest coarse cell, store its PQ code, and
    /// (for re-ranking) keep the full-precision vector by id.
    pub fn add(&mut self, id: u32, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        let cell = self.nearest_cell(v);
        let codes = self.pq.encode(v);
        self.lists[cell].push((id, codes));
        // raw store is dense by id; ids are added in order in the bench/tests.
        debug_assert_eq!(self.raw.len(), id as usize * self.dim);
        self.raw.extend_from_slice(v);
        self.n += 1;
    }

    /// Searches the `nprobe` nearest cells. ADC produces a shortlist of
    /// `rerank_factor * k` candidates cheaply; those are then re-scored with
    /// exact full-precision distances so the returned top-k recovers the recall
    /// that raw product quantization loses. `rerank_factor = 0` returns the
    /// pure-ADC ranking.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim);
        // Rank coarse cells by distance to the query.
        let mut cells: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.coarse[c * self.dim..(c + 1) * self.dim];
                (c, self.dist.distance(cent, query))
            })
            .collect();
        cells.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let probe = nprobe.min(self.nlist);

        let table = self.pq.adc_table(query);
        let mut scored: Vec<(u32, f32)> = Vec::new();
        for &(cell, _) in cells.iter().take(probe) {
            for (id, codes) in &self.lists[cell] {
                scored.push((*id, self.pq.adc_distance(&table, codes)));
            }
        }
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        if self.rerank_factor == 0 {
            scored.truncate(k);
            return scored;
        }

        // Exact re-rank of the ADC shortlist.
        let shortlist = (self.rerank_factor * k).min(scored.len());
        let mut exact: Vec<(u32, f32)> = scored[..shortlist]
            .iter()
            .map(|&(id, _)| (id, self.dist.distance(self.raw_of(id), query)))
            .collect();
        exact.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        exact.truncate(k);
        exact
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }
}

/// Lloyd's k-means over full `dim`-vectors, refining `centroids` in place.
fn lloyd_full(
    data: &[f32],
    dim: usize,
    centroids: &mut [f32],
    k: usize,
    iters: usize,
    dist: &DistanceFn,
) {
    let n = data.len() / dim;
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let d = dist.distance(cent, v);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if best != assign[i] {
                assign[i] = best;
                changed = true;
            }
        }
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            let v = &data[i * dim..(i + 1) * dim];
            for j in 0..dim {
                sums[c * dim + j] += v[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..dim {
                    centroids[c * dim + j] = sums[c * dim + j] * inv;
                }
            }
        }
        if !changed {
            break;
        }
    }
}
