//! Distance kernels with runtime SIMD dispatch.
//!
//! Every metric ships a portable scalar reference and, on `x86_64`, a
//! hand-written AVX-512 path selected once at construction via
//! `is_x86_feature_detected!`. The scalar path is the correctness oracle the
//! AVX-512 path is differential-tested against.

#[cfg(target_arch = "x86_64")]
mod avx512;

/// Distance/similarity metric over `f32` vectors of a fixed dimension.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean (L2^2). Monotone in L2, avoids the sqrt.
    L2,
    /// Negative inner product, so that "smaller is closer" holds uniformly.
    /// For normalized vectors this is order-equivalent to cosine distance.
    Ip,
}

/// A dispatched distance function bound to a metric and a backend.
#[derive(Copy, Clone, Debug)]
pub struct DistanceFn {
    metric: Metric,
    backend: Backend,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl DistanceFn {
    /// Selects the fastest available backend for this host, once.
    pub fn new(metric: Metric) -> Self {
        let backend = detect_backend();
        DistanceFn { metric, backend }
    }

    /// Forces the portable scalar backend (used as the differential oracle).
    pub fn scalar(metric: Metric) -> Self {
        DistanceFn {
            metric,
            backend: Backend::Scalar,
        }
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Returns the backend name for disclosure in benchmarks/READMEs.
    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            Backend::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => "avx512",
        }
    }

    /// Distance between two equal-length vectors. Smaller is closer for both
    /// metrics. Panics in debug builds on a length mismatch.
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self.backend {
            Backend::Scalar => scalar_distance(self.metric, a, b),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => {
                // SAFETY: backend is only set to Avx512 after runtime detection.
                unsafe { avx512::distance(self.metric, a, b) }
            }
        }
    }
}

fn detect_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return Backend::Avx512;
        }
    }
    Backend::Scalar
}

#[inline]
fn scalar_distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => {
            let mut acc = 0.0f32;
            for i in 0..a.len() {
                let d = a[i] - b[i];
                acc += d * d;
            }
            acc
        }
        Metric::Ip => {
            let mut acc = 0.0f32;
            for i in 0..a.len() {
                acc += a[i] * b[i];
            }
            -acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_l2_basic() {
        let d = DistanceFn::scalar(Metric::L2);
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 8.0];
        // 3^2 + 4^2 + 5^2 = 9 + 16 + 25 = 50
        assert!((d.distance(&a, &b) - 50.0).abs() < 1e-4);
    }

    #[test]
    fn scalar_ip_basic() {
        let d = DistanceFn::scalar(Metric::Ip);
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // -(4 + 10 + 18) = -32
        assert!((d.distance(&a, &b) + 32.0).abs() < 1e-4);
    }
}
