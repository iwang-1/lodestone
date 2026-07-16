//! Hand-written AVX-512 distance kernels (`x86_64`).
//!
//! Each kernel processes 16 `f32` lanes per iteration with a fused
//! multiply-add and handles the ragged tail with a masked load, so any
//! dimension is exact — not just multiples of 16. Correctness is pinned to the
//! scalar reference by the differential tests in `tests/`.

use super::Metric;
use std::arch::x86_64::*;

/// Dispatch entry. Caller guarantees `avx512f` is present.
#[target_feature(enable = "avx512f")]
pub unsafe fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => l2_sq(a, b),
        Metric::Ip => -dot(a, b),
    }
}

/// Horizontal sum of a 512-bit register of 16 f32 lanes.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn hsum(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// Builds a mask covering the low `rem` lanes (rem in 0..16).
#[inline]
fn tail_mask(rem: usize) -> __mmask16 {
    debug_assert!(rem < 16);
    ((1u32 << rem) - 1) as __mmask16
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc = _mm512_setzero_ps();

    let chunks = n / 16;
    let mut i = 0;
    while i < chunks * 16 {
        let va = _mm512_loadu_ps(pa.add(i));
        let vb = _mm512_loadu_ps(pb.add(i));
        let d = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc);
        i += 16;
    }

    let rem = n - i;
    if rem > 0 {
        let m = tail_mask(rem);
        let va = _mm512_maskz_loadu_ps(m, pa.add(i));
        let vb = _mm512_maskz_loadu_ps(m, pb.add(i));
        let d = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc);
    }
    hsum(acc)
}

#[target_feature(enable = "avx512f")]
unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc = _mm512_setzero_ps();

    let chunks = n / 16;
    let mut i = 0;
    while i < chunks * 16 {
        let va = _mm512_loadu_ps(pa.add(i));
        let vb = _mm512_loadu_ps(pb.add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
        i += 16;
    }

    let rem = n - i;
    if rem > 0 {
        let m = tail_mask(rem);
        let va = _mm512_maskz_loadu_ps(m, pa.add(i));
        let vb = _mm512_maskz_loadu_ps(m, pb.add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    hsum(acc)
}
