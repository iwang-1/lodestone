//! HNSW (Hierarchical Navigable Small World) graph index.
//!
//! Implements Malkov & Yashunin (2018): a multi-layer proximity graph with
//! exponentially-decaying level assignment, greedy descent through the upper
//! layers, and a beam search (`ef`) on layer 0. Insertion uses the paper's
//! neighbor-selection heuristic (Algorithm 4) rather than plain nearest-M,
//! which preserves recall on clustered data.

use crate::distance::{DistanceFn, Metric};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Tunable index-construction and query parameters.
#[derive(Copy, Clone, Debug)]
pub struct HnswParams {
    /// Max out-degree on layers > 0.
    pub m: usize,
    /// Max out-degree on layer 0 (typically 2*m).
    pub m0: usize,
    /// Candidate-list width during construction.
    pub ef_construction: usize,
    /// Level-generation normalization factor (1 / ln(m)).
    pub ml: f64,
    /// PRNG seed — construction is deterministic for a fixed seed + insert order.
    pub seed: u64,
}

impl HnswParams {
    pub fn new(m: usize) -> Self {
        HnswParams {
            m,
            m0: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f64).ln(),
            seed: 0x5eed_1234,
        }
    }
}

/// A candidate (distance, node id) ordered so that a max-heap yields the
/// farthest element at the top — the standard HNSW working-set discipline.
#[derive(Copy, Clone, Debug)]
struct Candidate {
    dist: f32,
    id: u32,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap on distance; break ties by id for determinism.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}

/// Min-heap wrapper (closest at top) via reversed ordering.
#[derive(Copy, Clone)]
struct MinCand(Candidate);
impl PartialEq for MinCand {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for MinCand {}
impl PartialOrd for MinCand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinCand {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.cmp(&self.0)
    }
}

/// An in-memory HNSW index over `f32` vectors of a fixed dimension.
pub struct Hnsw {
    dim: usize,
    params: HnswParams,
    dist: DistanceFn,
    /// Flat vector store: row `i` is `vectors[i*dim .. (i+1)*dim]`.
    vectors: Vec<f32>,
    /// `links[layer][node]` = neighbor ids of `node` on `layer`.
    links: Vec<Vec<Vec<u32>>>,
    /// Per-node top layer.
    node_levels: Vec<usize>,
    entry_point: Option<u32>,
    max_layer: usize,
    rng: Pcg64,
}

impl Hnsw {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        Hnsw {
            dim,
            params,
            dist: DistanceFn::new(metric),
            vectors: Vec::new(),
            links: vec![Vec::new()],
            node_levels: Vec::new(),
            entry_point: None,
            max_layer: 0,
            rng: Pcg64::seed_from_u64(params.seed),
        }
    }

    pub fn len(&self) -> usize {
        self.node_levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn backend_name(&self) -> &'static str {
        self.dist.backend_name()
    }

    #[inline]
    fn vec_of(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.vectors[i..i + self.dim]
    }

    #[inline]
    fn d(&self, id: u32, query: &[f32]) -> f32 {
        self.dist.distance(self.vec_of(id), query)
    }

    fn random_level(&mut self) -> usize {
        // level = floor(-ln(U) * ml), U in (0,1].
        let u: f64 = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
        (-u.ln() * self.params.ml).floor() as usize
    }

    /// Inserts a vector, returning its assigned node id.
    pub fn insert(&mut self, vector: &[f32]) -> u32 {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");
        let id = self.len() as u32;
        self.vectors.extend_from_slice(vector);

        let level = self.random_level();
        self.node_levels.push(level);
        // Any brand-new upper layer must be backfilled with an (empty) slot for
        // every already-inserted node, or a later descent that indexes
        // links[layer][old_node] goes out of bounds.
        while self.links.len() <= level {
            self.links.push(vec![Vec::new(); id as usize]);
        }
        // Every layer carries exactly one slot per node id; push this node's.
        for l in 0..self.links.len() {
            debug_assert!(self.links[l].len() == id as usize);
            self.links[l].push(Vec::new());
        }

        let entry = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_layer = level;
                return id;
            }
            Some(e) => e,
        };

        // Phase 1: greedy descent from the top down to level+1, ef=1.
        let mut curr = entry;
        let mut l = self.max_layer;
        while l > level {
            curr = self.greedy_descend(curr, vector, l);
            if l == 0 {
                break;
            }
            l -= 1;
        }

        // Phase 2: at each layer from min(level, max_layer) down to 0, run a
        // full ef_construction search and connect via the heuristic.
        let start_layer = level.min(self.max_layer);
        let mut ep = vec![curr];
        for l in (0..=start_layer).rev() {
            let mut found = self.search_layer(vector, &ep, self.params.ef_construction, l);
            let max_conn = if l == 0 {
                self.params.m0
            } else {
                self.params.m
            };
            let selected = self.select_neighbors(vector, &found, max_conn);
            self.connect(id, &selected, l, max_conn);
            // carry the found set forward as the next layer's entry points
            found.truncate(self.params.ef_construction);
            ep = found.iter().map(|c| c.id).collect();
            if ep.is_empty() {
                ep = vec![curr];
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }
        id
    }

    /// ef=1 greedy hop to the local minimum on one layer.
    fn greedy_descend(&self, entry: u32, query: &[f32], layer: usize) -> u32 {
        let mut curr = entry;
        let mut curr_d = self.d(curr, query);
        loop {
            let mut improved = false;
            for &nb in &self.links[layer][curr as usize] {
                let dd = self.d(nb, query);
                if dd < curr_d {
                    curr_d = dd;
                    curr = nb;
                    improved = true;
                }
            }
            if !improved {
                return curr;
            }
        }
    }

    /// Beam search on one layer returning up to `ef` nearest, sorted ascending.
    fn search_layer(
        &self,
        query: &[f32],
        entries: &[u32],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = vec![false; self.len()];
        // candidates: min-heap (closest first); results: max-heap (farthest at top).
        let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &e in entries {
            if visited[e as usize] {
                continue;
            }
            visited[e as usize] = true;
            let de = self.d(e, query);
            candidates.push(MinCand(Candidate { dist: de, id: e }));
            results.push(Candidate { dist: de, id: e });
            if results.len() > ef {
                results.pop();
            }
        }

        while let Some(MinCand(c)) = candidates.pop() {
            let worst = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            for &nb in &self.links[layer][c.id as usize] {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let dd = self.d(nb, query);
                let worst = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if dd < worst || results.len() < ef {
                    candidates.push(MinCand(Candidate { dist: dd, id: nb }));
                    results.push(Candidate { dist: dd, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_vec();
        out.sort(); // ascending by distance
        out
    }

    /// Neighbor-selection heuristic (Malkov & Yashunin Algorithm 4): keep a
    /// candidate only if it is closer to the query than to any already-kept
    /// neighbor, which prunes redundant links within the same cluster.
    fn select_neighbors(&self, _query: &[f32], candidates: &[Candidate], m: usize) -> Vec<u32> {
        let mut result: Vec<u32> = Vec::with_capacity(m);
        // candidates are sorted ascending by distance to the query.
        for c in candidates {
            if result.len() >= m {
                break;
            }
            let mut keep = true;
            for &r in &result {
                // distance(candidate, kept) vs distance(candidate, query)
                let d_cr = self.dist.distance(self.vec_of(c.id), self.vec_of(r));
                if d_cr < c.dist {
                    keep = false;
                    break;
                }
            }
            if keep {
                result.push(c.id);
            }
        }
        // If the heuristic was too aggressive, backfill with nearest remaining.
        if result.len() < m {
            for c in candidates {
                if result.len() >= m {
                    break;
                }
                if !result.contains(&c.id) {
                    result.push(c.id);
                }
            }
        }
        result
    }

    /// Adds bidirectional links and prunes over-full neighbor lists.
    fn connect(&mut self, id: u32, neighbors: &[u32], layer: usize, max_conn: usize) {
        self.links[layer][id as usize] = neighbors.to_vec();
        for &nb in neighbors {
            if nb == id {
                continue;
            }
            self.links[layer][nb as usize].push(id);
            if self.links[layer][nb as usize].len() > max_conn {
                // Re-select the neighbor's links against itself.
                let base = nb;
                let base_vec_start = base as usize * self.dim;
                let base_vec: Vec<f32> =
                    self.vectors[base_vec_start..base_vec_start + self.dim].to_vec();
                let mut cands: Vec<Candidate> = self.links[layer][base as usize]
                    .iter()
                    .map(|&x| Candidate {
                        dist: self.dist.distance(self.vec_of(x), &base_vec),
                        id: x,
                    })
                    .collect();
                cands.sort();
                let pruned = self.select_neighbors(&base_vec, &cands, max_conn);
                self.links[layer][base as usize] = pruned;
            }
        }
    }

    /// Approximate k-nearest-neighbor search. `ef` bounds the beam width and
    /// trades recall for latency; larger `ef` => higher recall, more work.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        let entry = match self.entry_point {
            None => return Vec::new(),
            Some(e) => e,
        };
        let mut curr = entry;
        let mut l = self.max_layer;
        while l > 0 {
            curr = self.greedy_descend(curr, query, l);
            l -= 1;
        }
        let ef = ef.max(k);
        let mut found = self.search_layer(query, &[curr], ef, 0);
        found.truncate(k);
        found.into_iter().map(|c| (c.id, c.dist)).collect()
    }
}
