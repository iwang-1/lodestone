//! Reproducible synthetic corpora for benchmarking without external downloads.
//!
//! `clustered` draws vectors from a mixture of Gaussians, which mimics the
//! cluster structure of real embedding spaces far better than uniform noise —
//! and is exactly the regime where the HNSW neighbor-selection heuristic
//! matters. Everything is seeded, so a benchmark run is byte-reproducible.

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// A flat corpus: `n` vectors of dimension `dim`, row-major.
pub struct Corpus {
    pub dim: usize,
    pub n: usize,
    pub data: Vec<f32>,
}

impl Corpus {
    #[inline]
    pub fn get(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

/// Generates `n` vectors from `n_clusters` isotropic Gaussians with unit-ish
/// separation, then a set of `n_queries` held-out query vectors from the same
/// distribution. Returns `(corpus, queries)`.
pub fn clustered(
    dim: usize,
    n: usize,
    n_clusters: usize,
    n_queries: usize,
    seed: u64,
) -> (Corpus, Corpus) {
    let mut rng = Pcg64::seed_from_u64(seed);

    // Cluster centers spread across a hypercube.
    let mut centers = vec![0.0f32; n_clusters * dim];
    for v in centers.iter_mut() {
        *v = rng.gen_range(-5.0..5.0);
    }

    let sigma = 1.0f32;
    let gen_point = |rng: &mut Pcg64, centers: &[f32]| -> Vec<f32> {
        let c = rng.gen_range(0..n_clusters);
        let mut p = vec![0.0f32; dim];
        for j in 0..dim {
            // Box-Muller for a standard normal, scaled by sigma.
            let u1: f32 = rng.gen_range(f32::MIN_POSITIVE..1.0);
            let u2: f32 = rng.gen_range(0.0..1.0);
            let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            p[j] = centers[c * dim + j] + sigma * g;
        }
        p
    };

    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n {
        data.extend_from_slice(&gen_point(&mut rng, &centers));
    }
    let mut qdata = Vec::with_capacity(n_queries * dim);
    for _ in 0..n_queries {
        qdata.extend_from_slice(&gen_point(&mut rng, &centers));
    }

    (
        Corpus { dim, n, data },
        Corpus {
            dim,
            n: n_queries,
            data: qdata,
        },
    )
}
