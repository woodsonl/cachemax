//! mlx-lm adapter — macOS/Apple Silicon only, via the Python sidecar.
//! Exposes no cache truth: TTFT warm/cold discrimination only, no hit-rate
//! number. `source()` is `NoCacheTruth` and the dashboard renders `—`.

use super::{Adapter, CacheSignal};
use crate::record::SourceLabel;

#[derive(Default)]
pub struct MlxLmAdapter;

impl Adapter for MlxLmAdapter {
    fn name(&self) -> &'static str {
        "mlxlm"
    }

    fn source(&self) -> SourceLabel {
        SourceLabel::NoCacheTruth
    }

    fn cache_signal(&self, _response_body: &[u8]) -> CacheSignal {
        CacheSignal::none()
    }
}
