// Test for the menu bar's render decision, separated from AppKit so it runs
// headless. Kept in lockstep with CacheMaxMenuBar.swift's `render`/title logic.
//
// Build/run: menubar/test.sh
import Foundation

struct ProxyState: Decodable {
    let live: Bool
    let hitRate: String
    let incomplete: Int
    enum CodingKeys: String, CodingKey {
        case live
        case hitRate = "hit_rate"
        case incomplete = "incomplete_count"
    }
}

func statusTitle(from data: Data?) -> String {
    guard let data, let s = try? JSONDecoder().decode(ProxyState.self, from: data), s.live else {
        return "○ —"
    }
    if s.incomplete > 0 { return "⚠ \(s.incomplete)" }
    return "● \(s.hitRate)"
}

var failures = 0
func check(_ name: String, _ got: String, _ want: String) {
    if got == want {
        print("ok   \(name): \(got)")
    } else {
        print("FAIL \(name): got \(got), want \(want)")
        failures += 1
    }
}

let live = #"{"live":true,"hit_rate":"723%","incomplete_count":0}"#.data(using: .utf8)
let incomplete = #"{"live":true,"hit_rate":"80%","incomplete_count":3}"#.data(using: .utf8)
let downField = #"{"live":false,"hit_rate":"—","incomplete_count":0}"#.data(using: .utf8)

check("live", statusTitle(from: live), "● 723%")
check("incomplete", statusTitle(from: incomplete), "⚠ 3")
check("live=false", statusTitle(from: downField), "○ —")
check("no response (proxy down)", statusTitle(from: nil), "○ —")
check("malformed json", statusTitle(from: Data("not json".utf8)), "○ —")

if failures > 0 {
    print("\n\(failures) failure(s)")
    exit(1)
}
print("\nall render cases pass")
