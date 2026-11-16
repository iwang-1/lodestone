//! Ivf-pq index: an inverted file over a coarse k-means quantizer, with each posting list storing PQ codes.

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
    /// Full-precision vectors kept for optional exact re-ranking of the ADC shortlist (the standard ivfpq+R
    /// recall recovery).
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

    /// Searches the `nprobe` nearest cells.
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

// ---- Persistence ----

const IVFPQ_MAGIC: &[u8; 8] = b"LDSTIVPQ";
const IVFPQ_FORMAT_VERSION: u32 = 1;

impl IvfPq {
    /// Write the index to `path` (atomically: temp file, fsync, rename): the coarse centroids, the PQ
    /// codebooks, every posting list, and the full-precision vectors used for re-ranking, so
    /// [`IvfPq::load`] restores an index that answers every query identically.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> crate::persist::Result<()> {
        let mut e = crate::persist::Encoder::new(IVFPQ_MAGIC, IVFPQ_FORMAT_VERSION);
        e.u64(self.dim as u64);
        e.u8(crate::persist::metric_tag(self.metric));
        e.u64(self.nlist as u64);
        e.u64(self.rerank_factor as u64);
        e.u64(self.n as u64);
        e.f32s(&self.coarse);
        self.pq.encode_into(&mut e);
        for list in &self.lists {
            let ids: Vec<u32> = list.iter().map(|(id, _)| *id).collect();
            e.u32s(&ids);
            let codes: Vec<u8> = list.iter().flat_map(|(_, c)| c.iter().copied()).collect();
            e.bytes(&codes);
        }
        e.f32s(&self.raw);
        e.write_atomic(path.as_ref())
    }

    /// Read an index written by [`IvfPq::save`]. Verifies the magic, format version, and checksum, then
    /// the structure search relies on: centroid and codebook shapes, one PQ code of `m` bytes per posting,
    /// every vector id in `0..n` appearing in exactly one posting list, and a full-precision row for each.
    pub fn load(path: impl AsRef<std::path::Path>) -> crate::persist::Result<IvfPq> {
        use crate::persist::{Decoder, PersistError::Corrupt};
        let file = std::fs::read(path.as_ref())?;
        let mut d = Decoder::open(&file, IVFPQ_MAGIC, IVFPQ_FORMAT_VERSION)?;
        let dim = d.u64()? as usize;
        let metric = crate::persist::metric_from_tag(d.u8()?)?;
        let nlist = d.u64()? as usize;
        let rerank_factor = d.u64()? as usize;
        let n = d.u64()? as usize;
        if dim == 0 || nlist == 0 {
            return Err(Corrupt(format!("dim={dim} nlist={nlist}")));
        }
        let coarse = d.f32s()?;
        if coarse.len() != nlist * dim {
            return Err(Corrupt(format!(
                "{} centroid floats for nlist={nlist} dim={dim}",
                coarse.len()
            )));
        }
        let pq = ProductQuantizer::decode_from(&mut d)?;
        if pq.dim() != dim {
            return Err(Corrupt(format!("PQ dim {} for index dim {dim}", pq.dim())));
        }
        let m = pq.m();
        let mut seen = vec![false; n];
        let mut lists = Vec::with_capacity(nlist);
        for cell in 0..nlist {
            let ids = d.u32s()?;
            let codes = d.bytes()?;
            if codes.len() != ids.len() * m {
                return Err(Corrupt(format!(
                    "cell {cell}: {} code bytes for {} postings of {m} bytes",
                    codes.len(),
                    ids.len()
                )));
            }
            let mut list = Vec::with_capacity(ids.len());
            for (i, &id) in ids.iter().enumerate() {
                let slot = seen
                    .get_mut(id as usize)
                    .ok_or_else(|| Corrupt(format!("cell {cell}: id {id} of {n}")))?;
                if *slot {
                    return Err(Corrupt(format!("id {id} appears in more than one posting")));
                }
                *slot = true;
                list.push((id, codes[i * m..(i + 1) * m].to_vec()));
            }
            lists.push(list);
        }
        if let Some(missing) = seen.iter().position(|s| !s) {
            return Err(Corrupt(format!("id {missing} is in no posting list")));
        }
        let raw = d.f32s()?;
        if raw.len() != n * dim {
            return Err(Corrupt(format!(
                "{} raw floats for {n} vectors of dim {dim}",
                raw.len()
            )));
        }
        d.finish()?;
        Ok(IvfPq {
            dim,
            metric,
            dist: DistanceFn::new(metric),
            coarse,
            nlist,
            pq,
            lists,
            raw,
            rerank_factor,
            n,
        })
    }
}

#[cfg(test)]
mod persist_tests {
    use super::*;
    use crate::persist::PersistError;

    fn small() -> IvfPq {
        let dim = 8;
        let data: Vec<f32> = (0..400 * dim)
            .map(|i| ((i * 37) % 101) as f32 / 10.0)
            .collect();
        let mut idx = IvfPq::train(dim, 8, 4, &data, Metric::L2, 3);
        for i in 0..400 {
            idx.add(i as u32, &data[i * dim..(i + 1) * dim]);
        }
        idx
    }

    fn assert_rejected(idx: &IvfPq, name: &str) {
        let path = std::env::temp_dir().join(format!(
            "lodestone-ivf-unit-{}-{name}.ivfpq",
            std::process::id()
        ));
        idx.save(&path).unwrap();
        match IvfPq::load(&path) {
            Err(PersistError::Corrupt(_)) => {}
            other => panic!("{name}: expected Corrupt, got ok={}", other.is_ok()),
        }
    }

    #[test]
    fn duplicate_posting_is_rejected() {
        let mut idx = small();
        let dup = idx.lists.iter().flatten().next().unwrap().clone();
        idx.lists[0].push(dup);
        assert_rejected(&idx, "dup");
    }

    #[test]
    fn missing_posting_is_rejected() {
        let mut idx = small();
        let cell = idx.lists.iter().position(|l| !l.is_empty()).unwrap();
        idx.lists[cell].pop();
        assert_rejected(&idx, "missing");
    }

    #[test]
    fn wrong_code_length_is_rejected() {
        let mut idx = small();
        let cell = idx.lists.iter().position(|l| !l.is_empty()).unwrap();
        idx.lists[cell][0].1.push(0);
        assert_rejected(&idx, "codelen");
    }

    #[test]
    fn raw_vector_count_mismatch_is_rejected() {
        let mut idx = small();
        idx.raw.truncate(idx.raw.len() - idx.dim);
        assert_rejected(&idx, "raw");
    }
}
