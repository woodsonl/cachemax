//! Token-prefix hashing and token counting.
//!
//! The tape's prefix map needs token-level hashes so continuity means the same
//! thing the provider's cache means. `tiktoken-rs` embeds the vocabulary, so no
//! external asset ships. Default encoding is `cl100k_base`; `--tokenizer`
//! overrides with another encoding name, a model name, or a vocab path.
//!
//! Cloud token counting stays provider-reported — this module never re-derives
//! a provider's token counts; it only hashes prefixes (always) and, on the
//! local path, measures the history span.

use tiktoken_rs::CoreBPE;

/// A messagelike unit whose token prefix we track. Kept dialect-neutral: each
/// adapter maps its wire messages onto these before hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Role or channel (`system`, `user`, `assistant`, `tool`).
    pub role: String,
    /// Flattened text content of the message.
    pub text: String,
}

/// The tokenizer, shared across requests. Holds a process-wide cached vocab,
/// so cloning is a pointer copy.
#[derive(Clone)]
pub struct Tokenizer {
    bpe: &'static CoreBPE,
    label: String,
}

impl Tokenizer {
    /// Default encoder: `cl100k_base`.
    pub fn default_encoder() -> Result<Self, String> {
        let bpe = tiktoken_rs::cl100k_base_singleton();
        Ok(Self {
            bpe,
            label: "cl100k_base".to_string(),
        })
    }

    /// Resolve `--tokenizer`: an encoding name (`cl100k_base`, `o200k_base`,
    /// `p50k_base`, `r50k_base`), or a model name routed through
    /// `bpe_for_model`. Every arm yields a cached singleton, so no vocab is
    /// re-parsed per call.
    pub fn resolve(spec: &str) -> Result<Self, String> {
        let bpe = match spec {
            "cl100k_base" => tiktoken_rs::cl100k_base_singleton(),
            "o200k_base" => tiktoken_rs::o200k_base_singleton(),
            "p50k_base" => tiktoken_rs::p50k_base_singleton(),
            "r50k_base" | "gpt2" => tiktoken_rs::r50k_base_singleton(),
            model => tiktoken_rs::bpe_for_model(model).map_err(|e| e.to_string())?,
        };
        Ok(Self {
            bpe,
            label: spec.to_string(),
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// Token count of one string.
    pub fn count(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    /// Hash a message list into a prefix-hash sequence. Each element is the
    /// hash of the token stream of all messages up to and including index `i`,
    /// so two requests share a prefix iff their hash sequences share one.
    pub fn prefix_hashes(&self, messages: &[Message]) -> Vec<u64> {
        use std::hash::{Hash, Hasher};
        let mut running = Vec::new();
        let mut hashes = Vec::with_capacity(messages.len());
        for m in messages {
            running.extend(self.bpe.encode_with_special_tokens(&m.role));
            running.extend(self.bpe.encode_with_special_tokens(&m.text));
            let mut h = std::collections::hash_map::DefaultHasher::new();
            running.hash(&mut h);
            hashes.push(h.finish());
        }
        hashes
    }

    /// Token count of a message list — the raw material for
    /// `resent_history_tokens` on the local path.
    pub fn count_messages(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| self.count(&m.role) + self.count(&m.text))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.into(),
            text: text.into(),
        }
    }

    #[test]
    fn default_encoder_is_cl100k() {
        let t = Tokenizer::default_encoder().unwrap();
        assert_eq!(t.label(), "cl100k_base");
    }

    #[test]
    fn shared_prefix_produces_shared_hashes() {
        let t = Tokenizer::default_encoder().unwrap();
        let a = [msg("system", "You are helpful."), msg("user", "Hi")];
        let b = [
            msg("system", "You are helpful."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
        ];
        let ha = t.prefix_hashes(&a);
        let hb = t.prefix_hashes(&b);
        assert_eq!(ha, hb[..ha.len()], "a is a prefix of b at the token level");
    }

    #[test]
    fn divergent_content_diverges_at_the_change_point() {
        let t = Tokenizer::default_encoder().unwrap();
        let a = [msg("system", "one"), msg("user", "apple")];
        let b = [msg("system", "one"), msg("user", "banana")];
        let ha = t.prefix_hashes(&a);
        let hb = t.prefix_hashes(&b);
        assert_eq!(ha[0], hb[0], "the shared system message hashes the same");
        assert_ne!(ha[1], hb[1], "the divergent user message differs");
    }

    #[test]
    fn count_is_positive_and_grows_with_text() {
        let t = Tokenizer::default_encoder().unwrap();
        assert!(t.count("hello") > 0);
        assert!(t.count("hello world, this is longer") > t.count("hello"));
    }

    #[test]
    fn resolve_named_encodings() {
        assert_eq!(Tokenizer::resolve("o200k_base").unwrap().label(), "o200k_base");
        assert_eq!(Tokenizer::resolve("p50k_base").unwrap().label(), "p50k_base");
    }

    #[test]
    fn golden_token_counts_pin_the_vocab() {
        // cl100k_base golden values. The tokenizer defines session continuity,
        // so a silent vocab swap would corrupt the tape — pin it.
        let t = Tokenizer::default_encoder().unwrap();
        assert_eq!(t.count("hello world"), 2);
        assert_eq!(t.count(""), 0);
        assert_eq!(
            t.count("The quick brown fox jumps over the lazy dog."),
            10
        );
    }
}
