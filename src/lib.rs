//! # lodestone
//!
//! A from-scratch vector search engine in Rust. It implements the two index
//! families that back production retrieval and RAG systems — an HNSW proximity
//! graph and an IVF-PQ compressed inverted file — over hand-written AVX-512
//! distance kernels, and measures approximate-search quality honestly against
//! an exact brute-force oracle on a recall@k-vs-QPS curve.
//!
//! The crate is deliberately dependency-light (no FAISS, no BLAS): the graph
//! construction, product quantizer, k-means, and SIMD kernels are all in-tree
//! so every claim traces to code in this repository.
//!
//! ## Modules
//! - [`distance`]: L2 / inner-product kernels with runtime AVX-512 dispatch and
//!   a scalar oracle.
//! - [`index`]: [`index::BruteForce`] (exact), [`index::Hnsw`] (graph),
//!   [`index::IvfPq`] (compressed).
//! - [`quant`]: [`quant::ProductQuantizer`] with Asymmetric Distance Computation.
//! - [`dataset`]: seeded, reproducible clustered corpora.
//! - [`eval`]: recall@k measurement against the exact oracle.

pub mod dataset;
pub mod distance;
pub mod eval;
pub mod index;
pub mod quant;

pub use distance::Metric;
