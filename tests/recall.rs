//! Recall sanity: on a small seeded clustered corpus, HNSW at a generous ef
//! and IVF-PQ at high nprobe must both recover most of the exact top-k. This
//! guards against index regressions without asserting host-specific speed.

use lodestone::dataset;
use lodestone::eval::recall_at_k;
use lodestone::index::{BruteForce, Hnsw, HnswParams, IvfPq};
use lodestone::Metric;

fn exact_topk(
    corpus: &dataset::Corpus,
    queries: &dataset::Corpus,
    k: usize,
) -> Vec<Vec<(u32, f32)>> {
    let mut brute = BruteForce::new(corpus.dim, Metric::L2);
    for i in 0..corpus.n {
        brute.add(corpus.get(i));
    }
    (0..queries.n)
        .map(|q| brute.search(queries.get(q), k))
        .collect()
}

#[test]
fn hnsw_high_ef_recovers_most_topk() {
    let dim = 64;
    let (corpus, queries) = dataset::clustered(dim, 5_000, 32, 200, 0x11);
    let k = 10;
    let exact = exact_topk(&corpus, &queries, k);

    let mut hnsw = Hnsw::new(dim, Metric::L2, HnswParams::new(16));
    for i in 0..corpus.n {
        hnsw.insert(corpus.get(i));
    }
    let approx: Vec<Vec<(u32, f32)>> = (0..queries.n)
        .map(|q| hnsw.search(queries.get(q), k, 200))
        .collect();
    let r = recall_at_k(&exact, &approx, k);
    assert!(
        r > 0.90,
        "HNSW recall@{k} at ef=200 was {r}, expected > 0.90"
    );
}

#[test]
fn ivfpq_high_nprobe_recovers_topk() {
    let dim = 64;
    let (corpus, queries) = dataset::clustered(dim, 5_000, 32, 200, 0x22);
    let k = 10;
    let exact = exact_topk(&corpus, &queries, k);

    let mut ivf = IvfPq::train(dim, 64, 8, &corpus.data, Metric::L2, 0x99);
    for i in 0..corpus.n {
        ivf.add(i as u32, corpus.get(i));
    }
    // Scanning all cells removes the coarse-miss term; remaining loss is PQ
    // approximation only. Should still recover a solid majority of top-k.
    let approx: Vec<Vec<(u32, f32)>> = (0..queries.n)
        .map(|q| ivf.search(queries.get(q), k, 64))
        .collect();
    let r = recall_at_k(&exact, &approx, k);
    assert!(
        r > 0.55,
        "IVF-PQ recall@{k} at nprobe=all was {r}, expected > 0.55"
    );
}
