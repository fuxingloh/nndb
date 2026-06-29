//! In-memory vector search.
//!
//! The pipeline mirrors the ANN-Benchmarks setup: a `base` set of database
//! vectors, a `query` set, and ground-truth nearest neighbors per query so we
//! can measure recall. Everything is loaded fully into RAM (not disk-bound).

pub mod eval;
pub mod fvecs;
pub mod quant;
pub mod search;
