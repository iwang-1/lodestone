//! Benchmark harness: builds each index over a seeded clustered corpus and
//! sweeps the recall/latency knob (ef for HNSW, nprobe for IVF-PQ), printing a
//! recall@k-vs-QPS table plus the exact-oracle backend in use. Single-threaded
//! query timing so QPS reflects one core; build uses all cores.
//!
//! Usage: lodestone-bench [--n N] [--dim D] [--queries Q] [--k K]

use lodestone::dataset;
use lodestone::distance::DistanceFn;
use lodestone::eval::recall_at_k;
use lodestone::index::{BruteForce, Hnsw, HnswParams, IvfPq};
use lodestone::Metric;
use std::time::Instant;

fn arg_usize(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == name && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse() {
                return v;
            }
        }
    }
    default
}

fn main() {
    let n = arg_usize("--n", 50_000);
    let dim = arg_usize("--dim", 128);
    let n_queries = arg_usize("--queries", 1_000);
    let k = arg_usize("--k", 10);
    let metric = Metric::L2;

    eprintln!(
        "lodestone-bench: n={n} dim={dim} queries={n_queries} k={k} backend={}",
        DistanceFn::new(metric).backend_name()
    );

    let (corpus, queries) = dataset::clustered(dim, n, 64, n_queries, 0xC0FFEE);

    // ---- exact oracle ----
    let mut brute = BruteForce::new(dim, metric);
    for i in 0..corpus.n {
        brute.add(corpus.get(i));
    }
    let t = Instant::now();
    let exact: Vec<Vec<(u32, f32)>> = (0..queries.n)
        .map(|q| brute.search(queries.get(q), k))
        .collect();
    let brute_secs = t.elapsed().as_secs_f64();
    let brute_qps = queries.n as f64 / brute_secs;
    println!("# exact brute-force oracle: {brute_qps:.0} QPS (parallel, full corpus scan)");

    // ---- HNSW ----
    println!("\n## HNSW (m=16)  recall@{k} vs QPS");
    println!("| ef | recall@{k} | QPS (1 core) |");
    println!("|---:|---:|---:|");
    let params = HnswParams::new(16);
    let t = Instant::now();
    let mut hnsw = Hnsw::new(dim, metric, params);
    for i in 0..corpus.n {
        hnsw.insert(corpus.get(i));
    }
    let build_secs = t.elapsed().as_secs_f64();
    eprintln!(
        "hnsw build: {build_secs:.1}s for {n} vectors (backend={})",
        hnsw.backend_name()
    );
    for &ef in &[16usize, 32, 64, 128, 256] {
        let t = Instant::now();
        let approx: Vec<Vec<(u32, f32)>> = (0..queries.n)
            .map(|q| hnsw.search(queries.get(q), k, ef))
            .collect();
        let secs = t.elapsed().as_secs_f64();
        let qps = queries.n as f64 / secs;
        let r = recall_at_k(&exact, &approx, k);
        println!("| {ef} | {:.4} | {qps:.0} |", r);
    }

    // ---- IVF-PQ ----
    let nlist = 256usize;
    let m = 32usize; // 128/32 = 4 dims/subspace, 16x compression
    println!(
        "\n## IVF-PQ (nlist={nlist}, m={m}, {}x compression)  recall@{k} vs QPS",
        (dim * 4) / m
    );
    println!("| nprobe | recall@{k} | QPS (1 core) |");
    println!("|---:|---:|---:|");
    let train_n = n.min(20_000);
    let ivf = {
        let mut ivf = IvfPq::train(dim, nlist, m, &corpus.data[..train_n * dim], metric, 0x1234);
        // Exact re-rank of a 32x-k ADC shortlist recovers the recall raw PQ
        // loses while keeping the compressed memory footprint.
        ivf.set_rerank_factor(32);
        for i in 0..corpus.n {
            ivf.add(i as u32, corpus.get(i));
        }
        ivf
    };
    eprintln!("ivfpq compression ratio: {:.1}x", ivf.compression_ratio());
    for &nprobe in &[1usize, 4, 8, 16, 32] {
        let t = Instant::now();
        let approx: Vec<Vec<(u32, f32)>> = (0..queries.n)
            .map(|q| ivf.search(queries.get(q), k, nprobe))
            .collect();
        let secs = t.elapsed().as_secs_f64();
        let qps = queries.n as f64 / secs;
        let r = recall_at_k(&exact, &approx, k);
        println!("| {nprobe} | {:.4} | {qps:.0} |", r);
    }
}
