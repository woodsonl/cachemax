// Does a concurrent pair both resolve the same turn and both append?
use cachemax::proxy::plan_request;
use cachemax::sessions::SharedSessions;
use cachemax::tokenize::{Message, Tokenizer};
use std::sync::Arc;
fn main() {
    let t = Tokenizer::default_encoder().unwrap();
    let store = Arc::new(SharedSessions::new());
    let conv = vec![
        Message {
            role: "system".into(),
            text: "s".into(),
        },
        Message {
            role: "user".into(),
            text: "u".into(),
        },
    ];
    // Two "concurrent" plans for the same session before either appends.
    let mut g = store.0.lock().unwrap();
    let p1 = plan_request(&mut g, &t, &conv);
    let p2 = plan_request(&mut g, &t, &conv);
    println!("p1 turn={} session={}", p1.turn, p1.session_id);
    println!("p2 turn={} session={}", p2.turn, p2.session_id);
}
