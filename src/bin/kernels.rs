//! Kernel microbenchmark: nanoseconds per distance call for the portable scalar reference and the
//! host's SIMD backend, across common embedding dimensions. Prints a markdown table.
//!
//! Each measurement computes distances from one query to a pool of vectors that fits in L1/L2 cache,
//! repeated until at least 200 ms have elapsed, and reports the best of 5 runs. It measures the
//! arithmetic of the kernel, not memory bandwidth.

use lodestone::distance::{DistanceFn, Metric};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::hint::black_box;
use std::time::{Duration, Instant};

const POOL: usize = 64;

fn ns_per_call(f: &DistanceFn, query: &[f32], pool: &[Vec<f32>]) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut calls = 0u64;
        let mut sink = 0.0f32;
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(200) {
            for v in pool {
                sink += f.distance(black_box(query), black_box(v));
            }
            calls += pool.len() as u64;
        }
        black_box(sink);
        let ns = start.elapsed().as_nanos() as f64 / calls as f64;
        best = best.min(ns);
    }
    best
}

fn main() {
    let simd_name = DistanceFn::new(Metric::L2).backend_name();
    println!("backend: {simd_name}");
    println!();
    println!("| metric | dim | scalar (ns/call) | {simd_name} (ns/call) | speedup |");
    println!("|---|---:|---:|---:|---:|");
    let mut rng = Pcg64::seed_from_u64(0x5EED);
    for metric in [Metric::L2, Metric::Ip] {
        for dim in [128usize, 384, 768, 1536] {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let pool: Vec<Vec<f32>> = (0..POOL)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect();
            let scalar = ns_per_call(&DistanceFn::scalar(metric), &query, &pool);
            let simd = ns_per_call(&DistanceFn::new(metric), &query, &pool);
            let name = match metric {
                Metric::L2 => "L2",
                Metric::Ip => "IP",
            };
            println!(
                "| {name} | {dim} | {scalar:.1} | {simd:.1} | {:.1}x |",
                scalar / simd
            );
        }
    }
}
