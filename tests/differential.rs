//! Differential test: the avx-512 distance kernels must agree with the portable scalar reference across
//! many random vectors and every dimension near a 16-lane boundary (tail-handling is the classic SIMD bug).

use lodestone::distance::{DistanceFn, Metric};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Returns the worst error under a combined absolute+relative tolerance.
fn max_err(metric: Metric, dim: usize, trials: usize, rng: &mut Pcg64) -> f32 {
    let scalar = DistanceFn::scalar(metric);
    let simd = DistanceFn::new(metric);
    let atol = 1e-3f32;
    let rtol = 1e-4f32;
    let mut worst = 0.0f32;
    for _ in 0..trials {
        let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let s = scalar.distance(&a, &b);
        let v = simd.distance(&a, &b);
        let allow = atol + rtol * s.abs();
        let viol = (s - v).abs() / allow; // <= 1.0 means within tolerance
        if viol > worst {
            worst = viol;
        }
    }
    worst
}

#[test]
fn avx512_matches_scalar_l2_all_dims() {
    let mut rng = Pcg64::seed_from_u64(0xD1FF);
    // Sweep dims straddling the 16-lane boundary to exercise the masked tail.
    for dim in [1usize, 7, 15, 16, 17, 31, 33, 64, 96, 127, 128, 256, 768] {
        let err = max_err(Metric::L2, dim, 200, &mut rng);
        assert!(
            err <= 1.0,
            "L2 dim={dim} exceeded tolerance (normalized viol={err})"
        );
    }
}

#[test]
fn avx512_matches_scalar_ip_all_dims() {
    let mut rng = Pcg64::seed_from_u64(0xBEEF);
    for dim in [1usize, 7, 15, 16, 17, 31, 33, 64, 96, 127, 128, 256, 768] {
        let err = max_err(Metric::Ip, dim, 200, &mut rng);
        assert!(
            err <= 1.0,
            "IP dim={dim} exceeded tolerance (normalized viol={err})"
        );
    }
}
