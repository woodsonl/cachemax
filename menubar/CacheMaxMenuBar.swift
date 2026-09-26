// cachemax menu bar — glanceable proxy status, click-through to the dashboard.
//
// macOS only. AppKit only, no dependencies. One status item that shows:
//   ● <hit rate>   proxy live (green)
//   ○ —            proxy not running (dim)
//   ⚠ <n>          turns incomplete (amber)
// Click opens the existing web dashboard; there are no panels here by design
// (the web dashboard is the single full UI).
//
// Build/run:  menubar/build.sh        (produces cachemax-menubar.app)
// Override the URL: CACHEMAX_URL=http://127.0.0.1:9999 menubar/build.sh && open ...

import AppKit
import Foundation

let baseURL = ProcessInfo.processInfo.environment["CACHEMAX_URL"] ?? "http://127.0.0.1:8899"
let pollSeconds: TimeInterval = 2

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

final class StatusController: NSObject {
    private let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private var timer: Timer?

    override init() {
        super.init()
        item.button?.title = "○ —"
        item.button?.toolTip = "cachemax: starting…"
        item.menu = menu(for: nil)
        poll()
        timer = Timer.scheduledTimer(withTimeInterval: pollSeconds, repeats: true) { [weak self] _ in
            self?.poll()
        }
    }

    private func poll() {
        guard let url = URL(string: "\(baseURL)/api/state") else { return }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.5
        URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
            let state = data.flatMap { try? JSONDecoder().decode(ProxyState.self, from: $0) }
            DispatchQueue.main.async { self?.render(state) }
        }.resume()
    }

    private func render(_ state: ProxyState?) {
        guard let s = state, s.live else {
            item.button?.title = "○ —"
            item.button?.toolTip = "cachemax: proxy not running (\(baseURL))"
            item.menu = menu(for: nil)
            return
        }
        if s.incomplete > 0 {
            item.button?.title = "⚠ \(s.incomplete)"
            item.button?.toolTip = "cachemax: \(s.incomplete) incomplete turn(s); hit rate \(s.hitRate)"
        } else {
            item.button?.title = "● \(s.hitRate)"
            item.button?.toolTip = "cachemax: live; hit rate \(s.hitRate)"
        }
        item.menu = menu(for: s)
    }

    private func menu(for state: ProxyState?) -> NSMenu {
        let m = NSMenu()
        let status = NSMenuItem(
            title: state == nil ? "Proxy not running" : "Live · hit rate \(state!.hitRate)",
            action: nil, keyEquivalent: "")
        status.isEnabled = false
        m.addItem(status)
        m.addItem(.separator())
        m.addItem(NSMenuItem(
            title: "Open dashboard", action: #selector(openDashboard), keyEquivalent: "d"))
        m.addItem(NSMenuItem(
            title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        for it in m.items where it.action == #selector(openDashboard) {
            it.target = self
        }
        return m
    }

    @objc private func openDashboard() {
        guard let url = URL(string: baseURL) else { return }
        NSWorkspace.shared.open(url)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory) // menu bar only, no Dock icon
let controller = StatusController()
_ = controller
app.run()
