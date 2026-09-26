use cachemax::tokenize::{Message, Tokenizer};
fn m(r: &str, t: &str) -> Message {
    Message {
        role: r.into(),
        text: t.into(),
    }
}
fn main() {
    let t = Tokenizer::default_encoder().unwrap();
    // Roles encode to single tokens: check
    for r in ["system", "user", "assistant", "tool"] {
        println!("role {:?} -> {:?}", r, t.count(r));
    }
    // If role hash is fed as tokens, a text that begins where role ends could shift.
    // A: [user "x"], [user "y"]  vs  B: [user "x"], [system "y"]? roles differ.
    // The stream includes roles, so to collide we need role sequence to match.
    // A: [user "x"], [user "y"]  stream u x u y
    // B: [user "x user y"]? no (space token).
    // Without spaces: A: [user "x"], [user "y"] -> "user"+"x"+"user"+"y"
    //                 B: [user "xuser"?] no role missing.
    // Conclusion: each message contributes role then text, both token lists appended.
    // A collision requires concatenation of two token lists to equal another's,
    // e.g. role1=[a], text1=[b] ; role2=[c], text2=[d]  vs  role=[a], text=[b c d]
    // Need a role whose tokens = [b]. "user" tokens = [u,s,e,r]? print:
    println!("user tokens = {}", t.count("user"));
    // build a text whose tokens are exactly the role tokens of a following message
    let a = vec![m("user", "I"), m("user", "am")];
    let b = vec![m("user", "I"), m("user", "am")];
    println!("{:?}", t.prefix_hashes(&a) == t.prefix_hashes(&b));
}
