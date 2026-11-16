//! Index persistence: a saved and reloaded HNSW index must answer exactly like the original, keep
//! growing exactly like the original, and reject any damaged file with an error instead of a panic.

use lodestone::dataset;
use lodestone::index::{Hnsw, HnswParams};
use lodestone::persist::PersistError;
use lodestone::Metric;
use std::path::PathBuf;

fn tmp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lodestone-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn build(n: usize, dim: usize, metric: Metric) -> (Hnsw, dataset::Corpus) {
    let (corpus, queries) = dataset::clustered(dim, n, 16, 100, 0xABCD);
    let mut idx = Hnsw::new(dim, metric, HnswParams::new(12));
    for i in 0..corpus.n {
        idx.insert(corpus.get(i));
    }
    (idx, queries)
}

/// Bit-exact comparison of search results (ids and distances) over several `ef` values.
fn assert_same_answers(a: &Hnsw, b: &Hnsw, queries: &dataset::Corpus) {
    for ef in [10, 40, 120] {
        for q in 0..queries.n {
            let ra = a.search(queries.get(q), 10, ef);
            let rb = b.search(queries.get(q), 10, ef);
            let ra: Vec<(u32, u32)> = ra.iter().map(|&(id, d)| (id, d.to_bits())).collect();
            let rb: Vec<(u32, u32)> = rb.iter().map(|&(id, d)| (id, d.to_bits())).collect();
            assert_eq!(ra, rb, "query {q} ef={ef}");
        }
    }
}

#[test]
fn roundtrip_answers_identically_for_both_metrics() {
    for metric in [Metric::L2, Metric::Ip] {
        let (idx, queries) = build(2_000, 32, metric);
        let path = tmp_path(&format!("roundtrip-{metric:?}.hnsw"));
        idx.save(&path).unwrap();
        let loaded = Hnsw::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_same_answers(&idx, &loaded, &queries);
    }
}

#[test]
fn inserts_after_load_build_the_same_graph() {
    let dim = 24;
    let (corpus, queries) = dataset::clustered(dim, 1_500, 12, 100, 0x5151);
    let mut original = Hnsw::new(dim, Metric::L2, HnswParams::new(10));
    for i in 0..1_000 {
        original.insert(corpus.get(i));
    }
    let path = tmp_path("grow.hnsw");
    original.save(&path).unwrap();
    let mut loaded = Hnsw::load(&path).unwrap();
    for i in 1_000..corpus.n {
        assert_eq!(original.insert(corpus.get(i)), loaded.insert(corpus.get(i)));
    }
    assert_same_answers(&original, &loaded, &queries);
    // Byte-identical files mean identical vectors, levels, and links, not just identical answers.
    let (pa, pb) = (tmp_path("grow-a.hnsw"), tmp_path("grow-b.hnsw"));
    original.save(&pa).unwrap();
    loaded.save(&pb).unwrap();
    assert_eq!(std::fs::read(pa).unwrap(), std::fs::read(pb).unwrap());
}

#[test]
fn empty_index_roundtrips() {
    let idx = Hnsw::new(8, Metric::L2, HnswParams::new(8));
    let path = tmp_path("empty.hnsw");
    idx.save(&path).unwrap();
    let loaded = Hnsw::load(&path).unwrap();
    assert!(loaded.is_empty());
    assert!(loaded.search(&[0.0; 8], 5, 10).is_empty());
}

#[test]
fn damaged_files_are_rejected_not_misread() {
    let (idx, _) = build(300, 16, Metric::L2);
    let path = tmp_path("damage.hnsw");
    idx.save(&path).unwrap();
    let good = std::fs::read(&path).unwrap();

    // Flip one bit at many positions spread through the file: header, vectors, links, trailer.
    let bad = tmp_path("damage-bad.hnsw");
    let step = (good.len() / 97).max(1);
    let mut checked = 0;
    for pos in (0..good.len()).step_by(step) {
        let mut bytes = good.clone();
        bytes[pos] ^= 0x10;
        std::fs::write(&bad, &bytes).unwrap();
        let err = Hnsw::load(&bad)
            .err()
            .unwrap_or_else(|| panic!("flip at {pos} loaded"));
        assert!(
            matches!(err, PersistError::Checksum | PersistError::BadMagic),
            "flip at {pos}: {err}"
        );
        checked += 1;
    }
    assert!(checked >= 90);

    // Truncation at several lengths.
    for len in [0, 4, 11, 12, good.len() / 2, good.len() - 1] {
        std::fs::write(&bad, &good[..len]).unwrap();
        assert!(Hnsw::load(&bad).is_err(), "truncated to {len} bytes loaded");
    }

    // A different file type.
    std::fs::write(&bad, b"not an index file at all").unwrap();
    assert!(matches!(Hnsw::load(&bad), Err(PersistError::BadMagic)));

    // A missing file is an I/O error.
    assert!(matches!(
        Hnsw::load(tmp_path("does-not-exist.hnsw")),
        Err(PersistError::Io(_))
    ));
}

// ---- IVF-PQ ----

use lodestone::index::IvfPq;

fn build_ivfpq(metric: Metric) -> (IvfPq, dataset::Corpus) {
    let dim = 32;
    let (corpus, queries) = dataset::clustered(dim, 3_000, 16, 100, 0x1F1F);
    let mut idx = IvfPq::train(dim, 32, 8, &corpus.data[..2_000 * dim], metric, 0x77);
    idx.set_rerank_factor(8);
    for i in 0..corpus.n {
        idx.add(i as u32, corpus.get(i));
    }
    (idx, queries)
}

#[test]
fn ivfpq_roundtrip_answers_identically() {
    for metric in [Metric::L2, Metric::Ip] {
        let (idx, queries) = build_ivfpq(metric);
        let path = tmp_path(&format!("ivfpq-{metric:?}.ivfpq"));
        idx.save(&path).unwrap();
        let loaded = IvfPq::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.compression_ratio(), idx.compression_ratio());
        for nprobe in [1, 4, 16] {
            for q in 0..queries.n {
                let a: Vec<(u32, u32)> = idx
                    .search(queries.get(q), 10, nprobe)
                    .iter()
                    .map(|&(id, d)| (id, d.to_bits()))
                    .collect();
                let b: Vec<(u32, u32)> = loaded
                    .search(queries.get(q), 10, nprobe)
                    .iter()
                    .map(|&(id, d)| (id, d.to_bits()))
                    .collect();
                assert_eq!(a, b, "{metric:?} query {q} nprobe={nprobe}");
            }
        }
    }
}

#[test]
fn ivfpq_damaged_files_are_rejected() {
    let (idx, _) = build_ivfpq(Metric::L2);
    let path = tmp_path("ivfpq-damage.ivfpq");
    idx.save(&path).unwrap();
    let good = std::fs::read(&path).unwrap();
    let bad = tmp_path("ivfpq-damage-bad.ivfpq");
    for pos in (0..good.len()).step_by((good.len() / 83).max(1)) {
        let mut bytes = good.clone();
        bytes[pos] ^= 0x01;
        std::fs::write(&bad, &bytes).unwrap();
        assert!(IvfPq::load(&bad).is_err(), "flip at {pos} loaded");
    }
    std::fs::write(&bad, &good[..good.len() - 3]).unwrap();
    assert!(IvfPq::load(&bad).is_err());
    // An HNSW file is not an IVF-PQ file.
    let (hnsw, _) = build(50, 8, Metric::L2);
    hnsw.save(&bad).unwrap();
    assert!(matches!(IvfPq::load(&bad), Err(PersistError::BadMagic)));
}
