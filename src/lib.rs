//! cachemax — OpenAI-compatible prompt-cache measurement proxy.
//!
//! A library crate so the binary and the integration tests exercise one code
//! path. The binary (`src/main.rs`) is the CLI shell around these modules.

pub mod adapters;
pub mod breakpoints;
pub mod canary;
pub mod dashboard;
pub mod export;
pub mod invariants;
pub mod ledger;
pub mod matrix;
pub mod proxy;
pub mod rates;
pub mod record;
pub mod repair;
pub mod replay;
pub mod sessions;
pub mod tokenize;
