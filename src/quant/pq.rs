//! Product Quantization (Jégou, Douze & Schmid, 2011).
//!
//! Splits each `dim`-vector into `m` contiguous subvectors and learns a
//! 256-centroid codebook per subspace via Lloyd's k-means, so a vector
//! compresses to `m` bytes (one code per subspace). Search uses Asymmetric
//! Distance Computation (ADC): the query stays full-precision and its distance
//! to every centroid is precomputed into a lookup table, so scoring a code is
//! `m` table lookups and adds.

use crate::distance::Metric;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// A trained product quantizer: `m` subspaces, 256 centroids each.
pub struct ProductQuantizer {
    dim: usize,
    m: usize,
    sub_dim: usize,
    /// codebooks[s] holds 256 * sub_dim f32 centroids for subspace s.
    codebooks: Vec<Vec<f32>>,
    metric: Metric,
}

const K: usize = 256; // centroids per subspace -> one u8 code each

impl ProductQuantizer {
    /// `m` must divide `dim`.
    pub fn train(dim: usize, m: usize, training: &[f32], metric: Metric, seed: u64) -> Self {
        assert!(dim.is_multiple_of(m), "m must divide dim");
        let sub_dim = dim / m;
        let n = training.len() / dim;
        assert!(n >= K, "need at least {K} training vectors, got {n}");
        let mut rng = Pcg64::seed_from_u64(seed);

        let mut codebooks = Vec::with_capacity(m);
        for s in 0..m {
            // Gather the s-th subvector of every training point.
            let mut sub: Vec<f32> = Vec::with_capacity(n * sub_dim);
            for i in 0..n {
                let off = i * dim + s * sub_dim;
                sub.extend_from_slice(&training[off..off + sub_dim]);
            }
            let book = kmeans(&sub, sub_dim, K, 15, &mut rng);
            codebooks.push(book);
        }
        ProductQuantizer {
            dim,
            m,
            sub_dim,
            codebooks,
            metric,
        }
    }

    pub fn m(&self) -> usize {
        self.m
    }

    /// Compression ratio vs a raw f32 vector (bytes_raw / bytes_code).
    pub fn compression_ratio(&self) -> f32 {
        (self.dim * 4) as f32 / self.m as f32
    }

    /// Encodes a full vector into `m` u8 codes by nearest centroid per subspace.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim);
        let mut codes = vec![0u8; self.m];
        for (s, code) in codes.iter_mut().enumerate() {
            let off = s * self.sub_dim;
            let sub = &v[off..off + self.sub_dim];
            *code = nearest_centroid(sub, &self.codebooks[s], self.sub_dim) as u8;
        }
        codes
    }

    /// Builds the ADC lookup table for a query: `table[s*K + c]` = squared L2
    /// (or negative inner product) between the query's s-th subvector and
    /// centroid `c` of subspace `s`.
    pub fn adc_table(&self, query: &[f32]) -> Vec<f32> {
        let mut table = vec![0.0f32; self.m * K];
        for s in 0..self.m {
            let off = s * self.sub_dim;
            let q = &query[off..off + self.sub_dim];
            let book = &self.codebooks[s];
            for c in 0..K {
                let cent = &book[c * self.sub_dim..(c + 1) * self.sub_dim];
                table[s * K + c] = match self.metric {
                    Metric::L2 => {
                        let mut acc = 0.0f32;
                        for j in 0..self.sub_dim {
                            let d = q[j] - cent[j];
                            acc += d * d;
                        }
                        acc
                    }
                    Metric::Ip => {
                        let mut acc = 0.0f32;
                        for j in 0..self.sub_dim {
                            acc += q[j] * cent[j];
                        }
                        -acc
                    }
                };
            }
        }
        table
    }

    /// Scores a code against a prebuilt ADC table: `m` lookups + adds.
    #[inline]
    pub fn adc_distance(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut acc = 0.0f32;
        for s in 0..self.m {
            acc += table[s * K + codes[s] as usize];
        }
        acc
    }
}

fn nearest_centroid(sub: &[f32], book: &[f32], sub_dim: usize) -> usize {
    let k = book.len() / sub_dim;
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..k {
        let cent = &book[c * sub_dim..(c + 1) * sub_dim];
        let mut acc = 0.0f32;
        for j in 0..sub_dim {
            let d = sub[j] - cent[j];
            acc += d * d;
        }
        if acc < best_d {
            best_d = acc;
            best = c;
        }
    }
    best
}

/// Lloyd's k-means on flat `sub_dim`-vectors; returns `k*sub_dim` centroids.
fn kmeans(data: &[f32], sub_dim: usize, k: usize, iters: usize, rng: &mut Pcg64) -> Vec<f32> {
    let n = data.len() / sub_dim;
    // k-means++ style seeding simplified to a random distinct sample.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);
    let mut centroids = vec![0.0f32; k * sub_dim];
    for c in 0..k {
        let src = idx[c % n] * sub_dim;
        centroids[c * sub_dim..(c + 1) * sub_dim].copy_from_slice(&data[src..src + sub_dim]);
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        // Assignment step.
        let mut changed = false;
        for i in 0..n {
            let v = &data[i * sub_dim..(i + 1) * sub_dim];
            let c = nearest_centroid(v, &centroids, sub_dim);
            if c != assign[i] {
                assign[i] = c;
                changed = true;
            }
        }
        // Update step.
        let mut sums = vec![0.0f32; k * sub_dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            let v = &data[i * sub_dim..(i + 1) * sub_dim];
            for j in 0..sub_dim {
                sums[c * sub_dim + j] += v[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..sub_dim {
                    centroids[c * sub_dim + j] = sums[c * sub_dim + j] * inv;
                }
            }
        }
        if !changed {
            break;
        }
    }
    centroids
}
