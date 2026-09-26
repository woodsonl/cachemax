//! Token-prefix hashing and token counting.
//!
//! The tape's prefix map needs token-level hashes so continuity means the same
//! thing the provider's cache means. `tiktoken-rs` embeds the vocabulary, so no
//! external asset ships. Default encoding is `cl100k_base`; `--tokenizer`
//! overrides with another encoding name (`o200k_base`, `p50k_base`, `r50k_base`)
//! or a model name routed through `bpe_for_model`.
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
    ///
    /// Message and field boundaries are delimited explicitly (a fixed sentinel
    /// byte hashed between role and text and between messages). Without that,
    /// two different message lists could flatten to the same token-ID sequence
    /// and collide into one session; the sentinel makes the boundary structural
    /// rather than a tokenizer coincidence.
    ///
    /// Incremental: one running hasher is extended per message and snapshotted,
    /// so the whole thing is O(total tokens), not O(n²) over the conversation.
    pub fn prefix_hashes(&self, messages: &[Message]) -> Vec<u64> {
        use std::hash::{Hash, Hasher};
        const FIELD_SEP: u8 = 0x1f; // unit separator
        const MSG_SEP: u8 = 0x1e; // record separator
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut hashes = Vec::with_capacity(messages.len());
        for m in messages {
            // Feed this message's tokens into the single running hasher,
            // delimiting role from text so the boundary can't be forged.
            for tok in self.bpe.encode_with_special_tokens(&m.role) {
                tok.hash(&mut hasher);
            }
            FIELD_SEP.hash(&mut hasher);
            for tok in self.bpe.encode_with_special_tokens(&m.text) {
                tok.hash(&mut hasher);
            }
            MSG_SEP.hash(&mut hasher);
            // The running hasher's state after message i is the prefix hash.
            hashes.push(hasher.finish());
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
    fn message_boundaries_cannot_be_forged() {
        // Two message lists whose flattened content is identical must not
        // collide: the boundary between messages is structural, not incidental.
        // Here the second list folds the first list's two messages into one,
        // which would share a token stream without an explicit separator.
        let t = Tokenizer::default_encoder().unwrap();
        let two = [msg("user", "hello"), msg("user", "world")];
        let one = [msg("user", "hello world")];
        let h_two = t.prefix_hashes(&two);
        let h_one = t.prefix_hashes(&one);
        assert_ne!(
            h_two[0], h_one[0],
            "one message must not hash to a two-message prefix"
        );
    }

    #[test]
    fn count_is_positive_and_grows_with_text() {
        let t = Tokenizer::default_encoder().unwrap();
        assert!(t.count("hello") > 0);
        assert!(t.count("hello world, this is longer") > t.count("hello"));
    }

    #[test]
    fn resolve_named_encodings() {
        assert_eq!(
            Tokenizer::resolve("o200k_base").unwrap().label(),
            "o200k_base"
        );
        assert_eq!(
            Tokenizer::resolve("p50k_base").unwrap().label(),
            "p50k_base"
        );
    }

    #[test]
    fn golden_token_counts_pin_the_vocab() {
        // cl100k_base golden values. The tokenizer defines session continuity,
        // so a silent vocab swap would corrupt the tape — pin it.
        let t = Tokenizer::default_encoder().unwrap();
        assert_eq!(t.count("hello world"), 2);
        assert_eq!(t.count(""), 0);
        assert_eq!(t.count("The quick brown fox jumps over the lazy dog."), 10);
    }
}
