"""
HEX BOT -- Clean Training Dashboard

Black-and-white minimalist dashboard with live game visualization,
ELO progression, loss curves, and resource monitoring.

Zero game-specific imports. Accepts data via REST API, WebSocket, or Python API.

Usage:
    Standalone:   python dashboard_clean.py
    As import:    from dashboard_clean import Dashboard
                  dash = Dashboard(port=5001)
                  dash.start()
                  dash.add_game(moves=[[0,0],[1,1]], result=1.0)
                  dash.add_metric(iteration=1, loss=0.5, elo=1050)
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from datetime import datetime as _dt
from typing import Any, Dict, List, Optional

import psutil
from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Track CPU and RAM usage over time."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._lock = threading.Lock()
        self._history: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            self.snapshot()
            time.sleep(self.interval)

    def snapshot(self) -> dict:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        snap = {
            "cpu_pct": round(cpu_pct, 1),
            "ram_pct": round(mem.percent, 1),
            "ram_used_gb": round(mem.used / (1024**3), 1),
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "cpu_count": multiprocessing.cpu_count(),
            "ts": time.time(),
        }
        with self._lock:
            self._history.append(snap)
            if len(self._history) > 600:
                self._history = self._history[-600:]
        return snap

    def get_history(self, n: int = 120) -> List[dict]:
        with self._lock:
            return list(self._history[-n:])

    @property
    def latest(self) -> dict:
        with self._lock:
            return self._history[-1] if self._history else {}


# ---------------------------------------------------------------------------
# Data stores (thread-safe, in-memory)
# ---------------------------------------------------------------------------

class DataStore:
    """Holds all dashboard state: games, metrics, resources."""

    def __init__(self):
        self._lock = threading.Lock()
        # Games
        self.games: List[dict] = []  # last 50
        self.total_games: int = 0
        # Metrics per iteration
        self.metrics: List[dict] = []
        # ELO history
        self.elo_history: List[dict] = [{"iteration": 0, "elo": 1000.0}]
        # Loss history
        self.loss_history: List[dict] = []
        # Win rate history
        self.winrate_history: List[dict] = []
        # Game length history
        self.gamelength_history: List[dict] = []
        # Speed history
        self.speed_history: List[dict] = []
        # Aggregate stats
        self.current_iteration: int = 0
        self.current_elo: float = 1000.0
        self.total_samples: int = 0
        self.workers: int = multiprocessing.cpu_count()
        # Per-iteration accumulators
        self._iter_wins = [0, 0, 0]  # p0, p1, draw
        self._iter_lengths: List[int] = []

    # -- Games --

    def add_game(self, moves: list, result: float,
                 metadata: Optional[dict] = None) -> dict:
        with self._lock:
            self.total_games += 1
            idx = self.total_games
            entry = {
                "game_idx": idx,
                "moves": moves,
                "result": result,
                "num_moves": len(moves),
                "metadata": metadata or {},
                "ts": time.time(),
            }
            self.games.append(entry)
            if len(self.games) > 50:
                self.games = self.games[-50:]
            # Accumulate win stats
            if result > 0:
                self._iter_wins[0] += 1
            elif result < 0:
                self._iter_wins[1] += 1
            else:
                self._iter_wins[2] += 1
            self._iter_lengths.append(len(moves))
            return entry

    def recent_games(self, n: int = 10) -> List[dict]:
        with self._lock:
            return list(self.games[-n:])

    # -- Metrics --

    def add_metric(self, **kw) -> dict:
        with self._lock:
            m = dict(kw)
            m.setdefault("ts", time.time())
            self.metrics.append(m)
            if len(self.metrics) > 2000:
                self.metrics = self.metrics[-2000:]
            # Update convenience fields
            if "iteration" in m:
                self.current_iteration = m["iteration"]
            if "elo" in m and m["elo"] is not None:
                self.current_elo = m["elo"]
                self.elo_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "elo": round(m["elo"], 1),
                })
            if "loss" in m and m["loss"] is not None:
                loss_entry = {"iteration": m.get("iteration", self.current_iteration)}
                if isinstance(m["loss"], dict):
                    loss_entry.update(m["loss"])
                else:
                    loss_entry["total"] = m["loss"]
                self.loss_history.append(loss_entry)
            if "wins" in m:
                w = m["wins"]
                total = sum(w) or 1
                self.winrate_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "p0": round(w[0] / total * 100, 1),
                    "p1": round(w[1] / total * 100, 1),
                    "draw": round(w[2] / total * 100, 1) if len(w) > 2 else 0,
                })
            if "avg_game_length" in m:
                self.gamelength_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "avg_game_length": m["avg_game_length"],
                })
            if "games" in m and "self_play_time" in m and m["self_play_time"] > 0:
                self.speed_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "games_per_sec": round(m["games"] / m["self_play_time"], 2),
                })
            if "workers" in m:
                self.workers = m["workers"]
            # Reset per-iteration accumulators on new iteration
            self._iter_wins = [0, 0, 0]
            self._iter_lengths = []
            return m

    def get_stats(self) -> dict:
        with self._lock:
            latest = self.metrics[-1] if self.metrics else {}
            total_w = sum(self._iter_wins) or 1
            avg_len = (
                round(sum(self._iter_lengths) / len(self._iter_lengths))
                if self._iter_lengths else 0
            )
            return {
                "iteration": self.current_iteration,
                "total_games": self.total_games,
                "current_elo": round(self.current_elo, 1),
                "workers": self.workers,
                "avg_game_length": avg_len,
                "win_p0": round(self._iter_wins[0] / total_w * 100, 1),
                "win_p1": round(self._iter_wins[1] / total_w * 100, 1),
                "latest_loss": latest.get("loss"),
                "buffer_size": latest.get("buffer_size", 0),
                "self_play_time": latest.get("self_play_time", 0),
            }

    def get_elo_history(self) -> List[dict]:
        with self._lock:
            return list(self.elo_history)

    def get_loss_history(self) -> List[dict]:
        with self._lock:
            return list(self.loss_history)

    def get_winrate_history(self) -> List[dict]:
        with self._lock:
            return list(self.winrate_history)

    def get_gamelength_history(self) -> List[dict]:
        with self._lock:
            return list(self.gamelength_history)

    def get_speed_history(self) -> List[dict]:
        with self._lock:
            return list(self.speed_history)


# ---------------------------------------------------------------------------
# Dashboard class (Python API)
# ---------------------------------------------------------------------------

class Dashboard:
    """Self-contained training dashboard server.

    Usage::

        dash = Dashboard(port=5001)
        dash.start()                          # background thread
        dash.add_game([[0,0],[1,1]], 1.0)     # push game
        dash.add_metric(iteration=1, loss=0.5, elo=1050)
        dash.update_progress(step=10, total=100, loss=0.32)
    """

    def __init__(self, port: int = 5001, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.store = DataStore()
        self.resource_monitor = ResourceMonitor()
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "hexbot-dashboard"
        self.socketio = SocketIO(
            self.app, cors_allowed_origins="*", async_mode="threading"
        )
        self._server_thread: Optional[threading.Thread] = None
        self._setup_routes()
        self._setup_socket_events()

    # -- Public API --

    def start(self) -> None:
        """Start dashboard server in a background thread."""
        self.resource_monitor.start()
        self._server_thread = threading.Thread(target=self._run, daemon=True)
        self._server_thread.start()
        ts = _dt.now().strftime("%H:%M:%S")
        print(f"[{ts}] Dashboard running at http://localhost:{self.port}")

    def _run(self) -> None:
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self.socketio.run(
            self.app, host=self.host, port=self.port,
            debug=False, allow_unsafe_werkzeug=True, log_output=False,
        )

    def add_game(self, moves: list, result: float,
                 metadata: Optional[dict] = None) -> None:
        """Push a completed game for display and replay."""
        entry = self.store.add_game(moves, result, metadata)
        self.socketio.emit("game_complete", {
            "game_idx": entry["game_idx"],
            "total_games": self.store.total_games,
            "result": result,
            "num_moves": len(moves),
            "moves": moves,
            "metadata": metadata or {},
        })
        self.socketio.emit("stats_update", self.store.get_stats())

    def add_metric(self, **kwargs: Any) -> None:
        """Push training metrics (iteration, loss, elo, wins, etc)."""
        self.store.add_metric(**kwargs)
        self.socketio.emit("stats_update", self.store.get_stats())

    def update_progress(self, step: int, total: int,
                        loss: Optional[float] = None,
                        phase: str = "training") -> None:
        """Update progress bar on the dashboard."""
        pct = round(step / max(total, 1) * 100, 1)
        self.socketio.emit("train_progress", {
            "step": step,
            "total": total,
            "loss": round(loss, 4) if loss is not None else None,
            "pct": pct,
            "phase": phase,
        })

    # -- Routes --

    def _setup_routes(self) -> None:
        app = self.app
        store = self.store
        rmon = self.resource_monitor

        @app.route("/")
        def index():
            return Response(DASHBOARD_HTML, content_type="text/html")

        # --- GET endpoints ---

        @app.route("/api/stats")
        def api_stats():
            s = store.get_stats()
            r = rmon.latest
            s["cpu_pct"] = r.get("cpu_pct", 0)
            s["ram_pct"] = r.get("ram_pct", 0)
            return jsonify(s)

        @app.route("/api/elo")
        def api_elo():
            return jsonify(store.get_elo_history())

        @app.route("/api/losses")
        def api_losses():
            return jsonify(store.get_loss_history())

        @app.route("/api/games")
        def api_games():
            n = request.args.get("n", 10, type=int)
            return jsonify(store.recent_games(n))

        @app.route("/api/resources")
        def api_resources():
            return jsonify({
                "current": rmon.latest,
                "history": rmon.get_history(),
            })

        @app.route("/api/winrates")
        def api_winrates():
            return jsonify(store.get_winrate_history())

        @app.route("/api/speed")
        def api_speed():
            return jsonify(store.get_speed_history())

        @app.route("/api/gamelength")
        def api_gamelength():
            return jsonify(store.get_gamelength_history())

        # --- POST endpoints ---

        @app.route("/api/game", methods=["POST"])
        def api_post_game():
            d = request.get_json(force=True)
            moves = d.get("moves", [])
            result = d.get("result", 0.0)
            meta = {k: v for k, v in d.items() if k not in ("moves", "result")}
            self.add_game(moves, result, meta)
            return jsonify({"ok": True, "game_idx": store.total_games})

        @app.route("/api/metric", methods=["POST"])
        def api_post_metric():
            d = request.get_json(force=True)
            self.add_metric(**d)
            return jsonify({"ok": True})

    # -- Socket events --

    def _setup_socket_events(self) -> None:
        sio = self.socketio
        store = self.store

        @sio.on("connect")
        def on_connect():
            sio.emit("stats_update", store.get_stats())

        @sio.on("game_result")
        def on_game_result(data):
            moves = data.get("moves", [])
            result = data.get("result", 0.0)
            meta = {k: v for k, v in data.items() if k not in ("moves", "result")}
            self.add_game(moves, result, meta)

        @sio.on("metric")
        def on_metric(data):
            self.add_metric(**data)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEX BOT</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#000;font-family:'SF Mono','Courier New',monospace;font-size:13px;line-height:1.5}
header{border-bottom:1px solid #000;padding:14px 24px;display:flex;align-items:center;gap:20px}
.title{font-size:15px;font-weight:700;letter-spacing:4px;text-transform:uppercase}
.status{margin-left:auto;letter-spacing:3px;font-size:11px;font-weight:700}
main{display:flex;border-bottom:1px solid #000;min-height:calc(100vh - 130px)}
.left{width:50%;border-right:1px solid #000;padding:20px;display:flex;flex-direction:column;align-items:center}
.right{width:50%;display:flex;flex-direction:column;overflow-y:auto}
.chart-box{padding:16px 20px;border-bottom:1px solid #000;display:flex;flex-direction:column}
.chart-box:last-child{border-bottom:none}
.chart-header{font-weight:700;letter-spacing:2px;font-size:10px;text-transform:uppercase;
  cursor:pointer;user-select:none;display:flex;align-items:center;gap:6px}
.chart-header .toggle{font-size:8px;transition:transform .15s}
.chart-header .toggle.collapsed{transform:rotate(-90deg)}
.chart-body{overflow:hidden;transition:max-height .25s ease,opacity .2s ease}
.chart-body.collapsed{max-height:0!important;opacity:0;padding:0}
.canvas-wrap{position:relative;min-height:180px;height:180px;margin-top:8px}
.canvas-wrap canvas{position:absolute;top:0;left:0;width:100%;height:100%}
#hex-canvas-wrap{flex:1;position:relative;width:100%;min-height:0}
#hex-canvas{position:absolute;top:0;left:0;width:100%;height:100%;border:1px solid #eee}
.game-info{font-size:11px;letter-spacing:1px;min-height:18px;margin-top:8px;flex-shrink:0}
footer{padding:12px 24px;display:flex;gap:20px;flex-wrap:wrap;font-size:11px;letter-spacing:.5px;border-top:1px solid #000}
footer b{font-weight:700}
footer span{white-space:nowrap}
.sep{color:#ccc}
.res-bar{display:flex;gap:4px;align-items:center}
.res-meter{width:40px;height:8px;border:1px solid #000;display:inline-block;position:relative;vertical-align:middle}
.res-meter-fill{height:100%;background:#000;transition:width .3s}
#live-stats{width:100%;padding:0 0 12px 0;font:11px 'SF Mono','Courier New',monospace;
  border-bottom:1px solid #eee;margin-bottom:12px}
#live-stats .row{display:flex;justify-content:space-between;margin-bottom:4px}
#progress-wrap{width:100%;margin-bottom:8px;display:none}
.progress-bar{display:flex;align-items:center;gap:8px}
.progress-track{flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:#000;width:0%;transition:width .2s}
.progress-label{font:700 10px 'Courier New';min-width:120px;text-align:right}
.settings-btn{cursor:pointer;font-size:18px;margin-left:12px;user-select:none}
.settings-panel{position:absolute;top:42px;right:20px;background:#fff;border:1px solid #000;
  padding:16px;z-index:100;font:11px 'SF Mono','Courier New',monospace;min-width:260px;box-shadow:2px 2px 0 #0001}
.settings-row{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.settings-row label{width:110px;font-weight:700;flex-shrink:0}
.settings-row input[type=range]{flex:1;accent-color:#000}
.settings-row span.val{width:50px;text-align:right;font-size:10px}
</style>
</head>
<body>
<header>
  <span class="title">Hex Bot Training</span>
  <span class="status" id="status">IDLE</span>
  <span class="settings-btn" onclick="toggleSettings()">&#9881;</span>
</header>
<div id="settings-panel" class="settings-panel" style="display:none">
  <div style="font-weight:700;margin-bottom:12px;letter-spacing:2px;font-size:10px;text-transform:uppercase">Settings</div>
  <div class="settings-row">
    <label>Replay speed</label>
    <input type="range" id="set-speed" min="50" max="500" value="120" step="10"
      oninput="saveSetting('replaySpeed',+this.value);el('set-speed-val').textContent=this.value+'ms'">
    <span class="val" id="set-speed-val">120ms</span>
  </div>
  <div class="settings-row">
    <label>Dot size</label>
    <input type="range" id="set-dotsize" min="1" max="5" value="2" step="0.5"
      oninput="saveSetting('dotSize',+this.value);el('set-dotsize-val').textContent=this.value;drawHex()">
    <span class="val" id="set-dotsize-val">2</span>
  </div>
  <div class="settings-row">
    <label>Grid radius</label>
    <input type="range" id="set-radius" min="1" max="4" value="2" step="1"
      oninput="saveSetting('emptyHexRadius',+this.value);el('set-radius-val').textContent=this.value;drawHex()">
    <span class="val" id="set-radius-val">2</span>
  </div>
  <div class="settings-row">
    <label>Auto-refresh</label>
    <input type="checkbox" id="set-autorefresh" checked
      onchange="saveSetting('autoRefresh',this.checked)">
  </div>
</div>
<main>
  <div class="left">
    <div id="live-stats">
      <div class="row">
        <span><b>Iteration</b> <span id="ls-iter">&mdash;</span></span>
        <span><b>Workers</b> <span id="ls-workers">&mdash;</span></span>
        <span><b>Games</b> <span id="ls-games">&mdash;</span></span>
      </div>
      <div class="row">
        <span><b>Avg Length</b> <span id="ls-len">&mdash;</span></span>
        <span><b>Win%</b> P0:<span id="ls-w0">&mdash;</span> P1:<span id="ls-w1">&mdash;</span></span>
        <span><b>Loss</b> <span id="ls-loss">&mdash;</span></span>
      </div>
    </div>
    <div id="progress-wrap">
      <div class="progress-bar">
        <div class="progress-track">
          <div class="progress-fill" id="progress-fill"></div>
        </div>
        <span class="progress-label" id="progress-label"></span>
      </div>
    </div>
    <div style="font-weight:700;letter-spacing:2px;font-size:10px;text-transform:uppercase;margin-bottom:4px">
      Training Game #<span id="game-num">&mdash;</span>
    </div>
    <div id="hex-canvas-wrap"><canvas id="hex-canvas"></canvas></div>
    <div class="game-info" id="game-info">Waiting for games...</div>
  </div>
  <div class="right">
    <div class="chart-box" id="box-elo">
      <div class="chart-header" onclick="toggleChart('elo')">
        <span class="toggle" id="tog-elo">&#9660;</span> Elo Progression
      </div>
      <div class="chart-body" id="body-elo">
        <div class="canvas-wrap"><canvas id="elo-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-loss">
      <div class="chart-header" onclick="toggleChart('loss')">
        <span class="toggle" id="tog-loss">&#9660;</span> Loss Curves
      </div>
      <div class="chart-body" id="body-loss">
        <div class="canvas-wrap"><canvas id="loss-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-winrate">
      <div class="chart-header" onclick="toggleChart('winrate')">
        <span class="toggle" id="tog-winrate">&#9660;</span> Win Rates
      </div>
      <div class="chart-body" id="body-winrate">
        <div class="canvas-wrap"><canvas id="winrate-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-gamelength">
      <div class="chart-header" onclick="toggleChart('gamelength')">
        <span class="toggle" id="tog-gamelength">&#9660;</span> Game Length
      </div>
      <div class="chart-body" id="body-gamelength">
        <div class="canvas-wrap"><canvas id="gamelength-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-speed">
      <div class="chart-header" onclick="toggleChart('speed')">
        <span class="toggle" id="tog-speed">&#9660;</span> Training Speed
      </div>
      <div class="chart-body" id="body-speed">
        <div class="canvas-wrap"><canvas id="speed-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-resources">
      <div class="chart-header" onclick="toggleChart('resources')">
        <span class="toggle" id="tog-resources">&#9660;</span> Resources (CPU / RAM)
      </div>
      <div class="chart-body" id="body-resources">
        <div class="canvas-wrap" style="min-height:140px;height:140px"><canvas id="res-chart"></canvas></div>
      </div>
    </div>
  </div>
</main>
<footer>
  <span>Iter <b id="s-iter">0</b></span>
  <span class="sep">|</span>
  <span>Games <b id="s-games">0</b></span>
  <span class="sep">|</span>
  <span>Elo <b id="s-elo">1000</b></span>
  <span class="sep">|</span>
  <span>Win P0:<b id="s-w0">0</b>% P1:<b id="s-w1">0</b>%</span>
  <span class="sep">|</span>
  <span class="res-bar">CPU <div class="res-meter"><div class="res-meter-fill" id="cpu-fill" style="width:0%"></div></div> <b id="s-cpu">0</b>%</span>
  <span class="res-bar">RAM <div class="res-meter"><div class="res-meter-fill" id="ram-fill" style="width:0%"></div></div> <b id="s-ram">0</b>%</span>
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script>
// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
const DPR = window.devicePixelRatio || 1;
const socket = io();
socket.on('connect', () => console.log('Socket connected:', socket.id));
socket.on('disconnect', () => console.log('Socket disconnected'));

let stones0 = [], stones1 = [];

function el(id) { return document.getElementById(id); }
function setInfo(t) { el('game-info').textContent = t; }

// ---------------------------------------------------------------------------
// HiDPI canvas helper
// ---------------------------------------------------------------------------
function sizeCanvas(cv) {
  const r = cv.parentElement.getBoundingClientRect();
  const w = Math.round(r.width), h = Math.round(r.height);
  if (cv.width !== w * DPR || cv.height !== h * DPR) {
    cv.width = w * DPR; cv.height = h * DPR;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  return { w, h };
}

// ---------------------------------------------------------------------------
// Collapsible charts
// ---------------------------------------------------------------------------
const chartState = {};
function toggleChart(name) {
  const body = el('body-' + name);
  const tog = el('tog-' + name);
  const collapsed = !body.classList.contains('collapsed');
  if (collapsed) {
    body.classList.add('collapsed');
    tog.classList.add('collapsed');
  } else {
    body.classList.remove('collapsed');
    tog.classList.remove('collapsed');
    // Redraw after expand
    setTimeout(fetchCharts, 50);
  }
  chartState[name] = collapsed;
}

// ---------------------------------------------------------------------------
// Hex canvas rendering
// ---------------------------------------------------------------------------
const HEX_SIZE = 18;
const S3 = Math.sqrt(3);

// Settings (persisted to localStorage)
const defaultSettings = { replaySpeed: 120, dotSize: 2, emptyHexRadius: 2, autoRefresh: true };
let settings = Object.assign({}, defaultSettings);
try { const s = JSON.parse(localStorage.getItem('hexdash_settings')); if (s) Object.assign(settings, s); } catch(e) {}
function saveSetting(key, val) { settings[key] = val; localStorage.setItem('hexdash_settings', JSON.stringify(settings)); }
function toggleSettings() {
  const p = el('settings-panel');
  p.style.display = p.style.display === 'none' ? '' : 'none';
}

function axToPixel(q, r) {
  return [HEX_SIZE * (S3 * q + S3 / 2 * r), HEX_SIZE * (1.5 * r)];
}

function drawHex() {
  const cv = el('hex-canvas');
  const { w: W, h: H } = sizeCanvas(cv);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const all = [...stones0, ...stones1];
  if (!all.length) {
    ctx.fillStyle = '#aaa';
    ctx.font = '12px Courier New';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for training game...', W / 2, H / 2);
    return;
  }

  // Compute bounding box
  let mnX = 1e9, mxX = -1e9, mnY = 1e9, mxY = -1e9;
  for (const [q, r] of all) {
    const [px, py] = axToPixel(q, r);
    if (px < mnX) mnX = px; if (px > mxX) mxX = px;
    if (py < mnY) mnY = py; if (py > mxY) mxY = py;
  }
  const mg = HEX_SIZE * 3;
  const spanX = mxX - mnX + mg * 2, spanY = mxY - mnY + mg * 2;
  const sc = Math.min(W / spanX, H / spanY, 2.5);
  const ox = W / 2 - (mnX + mxX) / 2 * sc, oy = H / 2 - (mnY + mxY) / 2 * sc;

  function toS(q, r) {
    const [px, py] = axToPixel(q, r);
    return [px * sc + ox, py * sc + oy];
  }
  function hexPath(cx, cy, sz) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 3 * i - Math.PI / 6;
      const hx = cx + sz * Math.cos(a), hy = cy + sz * Math.sin(a);
      i === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
    }
    ctx.closePath();
  }
  const hr = HEX_SIZE * sc * 0.88;

  // Empty hex dots around placed stones
  const stoneSet = new Set(all.map(s => s[0] + ',' + s[1]));
  const emptySet = new Set();
  const dotR = settings.emptyHexRadius;
  for (const [q, r] of all) {
    for (let dq = -dotR; dq <= dotR; dq++) {
      for (let dr = -dotR; dr <= dotR; dr++) {
        if (Math.abs(dq) + Math.abs(dr) > dotR + 1) continue;
        const key = (q + dq) + ',' + (r + dr);
        if (!stoneSet.has(key) && !emptySet.has(key)) {
          emptySet.add(key);
          const [sx, sy] = toS(q + dq, r + dr);
          ctx.fillStyle = '#ccc';
          ctx.beginPath();
          ctx.arc(sx, sy, Math.max(1.5, settings.dotSize * sc), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  // P0: solid black hexagons
  for (const [q, r] of stones0) {
    const [sx, sy] = toS(q, r);
    hexPath(sx, sy, hr);
    ctx.fillStyle = '#000'; ctx.fill();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
  }

  // P1: white hexagons with hatching
  for (const [q, r] of stones1) {
    const [sx, sy] = toS(q, r);
    hexPath(sx, sy, hr);
    ctx.fillStyle = '#fff'; ctx.fill();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.save();
    hexPath(sx, sy, hr); ctx.clip();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 0.6;
    const step = Math.max(3, 4 * sc);
    for (let d = -hr * 2; d <= hr * 2; d += step) {
      ctx.beginPath();
      ctx.moveTo(sx + d - hr, sy - hr);
      ctx.lineTo(sx + d + hr, sy + hr);
      ctx.stroke();
    }
    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// Game replay
// ---------------------------------------------------------------------------
let replayTimer = null, replayBusy = false, pendingGame = null;
let gameStats = { w0: 0, w1: 0, totalLen: 0, count: 0 };

function replayGame(d) {
  replayBusy = true;
  stones0 = []; stones1 = [];
  drawHex();
  el('game-num').textContent = d.game_idx;
  let mi = 0, curPlayer = 0, stonesInTurn = 0;
  const moves = d.moves;
  setInfo('Game #' + d.game_idx + ' playing... (' + moves.length + ' moves)');
  replayTimer = setInterval(() => {
    if (mi >= moves.length) {
      clearInterval(replayTimer); replayTimer = null;
      const winner = d.result > 0 ? 'P0 (black)' : 'P1 (white)';
      setInfo('Game #' + d.game_idx + ': ' + winner + ' wins in ' + moves.length + ' moves');
      setTimeout(() => {
        replayBusy = false;
        if (pendingGame) { const g = pendingGame; pendingGame = null; replayGame(g); }
      }, 1500);
      return;
    }
    const m = moves[mi];
    if (curPlayer === 0) stones0.push(m); else stones1.push(m);
    mi++; stonesInTurn++;
    // Turn structure: first move is 1 stone, then 2 each
    const needed = (mi <= 1) ? 1 : 2;
    if (stonesInTurn >= needed) { curPlayer = 1 - curPlayer; stonesInTurn = 0; }
    drawHex();
    setInfo('Game #' + d.game_idx + ' \u2014 Move ' + mi + '/' + moves.length);
  }, settings.replaySpeed);
}

// ---------------------------------------------------------------------------
// Socket events
// ---------------------------------------------------------------------------
socket.on('game_complete', d => {
  try {
    if (!d.moves || !d.moves.length) return;
    // Cancel any in-progress replay
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    el('status').textContent = 'GAME ' + d.game_idx;
    // Progress bar for self-play
    if (d.total_games) {
      const pw = el('progress-wrap');
      const pf = el('progress-fill');
      const plab = el('progress-label');
      if (pw) pw.style.display = '';
      if (pf) pf.style.width = Math.round(d.game_idx / d.total_games * 100) + '%';
      if (plab) plab.textContent = 'Self-play ' + d.game_idx + '/' + d.total_games;
    }
    // Update live stats
    gameStats.count++;
    gameStats.totalLen += d.num_moves;
    if (d.result > 0) gameStats.w0++; else if (d.result < 0) gameStats.w1++;
    const lsG = el('ls-games');
    if (lsG) lsG.textContent = gameStats.count;
    const lsL = el('ls-len');
    if (lsL) lsL.textContent = Math.round(gameStats.totalLen / gameStats.count);
    const lsW0 = el('ls-w0');
    if (lsW0) lsW0.textContent = Math.round(gameStats.w0 / gameStats.count * 100) + '%';
    const lsW1 = el('ls-w1');
    if (lsW1) lsW1.textContent = Math.round(gameStats.w1 / gameStats.count * 100) + '%';
    // Queue or play replay
    if (replayBusy) {
      pendingGame = d;
    } else {
      replayGame(d);
    }
  } catch (e) { console.error('game_complete error:', e); }
});

socket.on('stats_update', d => {
  try {
    if (d.iteration != null) {
      el('s-iter').textContent = d.iteration;
      el('ls-iter').textContent = d.iteration;
    }
    if (d.total_games != null) el('s-games').textContent = d.total_games;
    if (d.current_elo != null) el('s-elo').textContent = Math.round(d.current_elo);
    if (d.workers != null) el('ls-workers').textContent = d.workers;
    if (d.win_p0 != null) el('s-w0').textContent = Math.round(d.win_p0);
    if (d.win_p1 != null) el('s-w1').textContent = Math.round(d.win_p1);
    if (d.latest_loss != null) {
      const lv = typeof d.latest_loss === 'object' ? d.latest_loss.total : d.latest_loss;
      if (lv != null) el('ls-loss').textContent = parseFloat(lv).toFixed(4);
    }
    if (d.avg_game_length) el('ls-len').textContent = d.avg_game_length;
    fetchCharts();
  } catch (e) { console.error('stats_update error:', e); }
});

socket.on('train_progress', d => {
  try {
    const pw = el('progress-wrap');
    const pf = el('progress-fill');
    const plab = el('progress-label');
    if (pw) pw.style.display = '';
    if (pf) pf.style.width = d.pct + '%';
    const phase = d.phase || 'Training';
    const lossStr = d.loss != null ? ' (loss ' + d.loss + ')' : '';
    if (plab) plab.textContent = phase + ' ' + d.step + '/' + d.total + lossStr;
    if (d.loss != null) el('ls-loss').textContent = d.loss;
    el('status').textContent = phase.toUpperCase() + ' ' + d.step + '/' + d.total;
  } catch (e) { console.error('train_progress error:', e); }
});

// ---------------------------------------------------------------------------
// Line chart (HiDPI, auto-scaling)
// ---------------------------------------------------------------------------
function drawLineChart(cv, datasets, opts) {
  if (!cv) return;
  // Skip collapsed charts
  const body = cv.closest('.chart-body');
  if (body && body.classList.contains('collapsed')) return;

  const { w: W, h: H } = sizeCanvas(cv);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const pad = { t: 14, r: 16, b: 24, l: 52 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
  if (pW < 10 || pH < 10) return;

  const pts = datasets.flatMap(d => d.data);
  if (pts.length < 2) {
    ctx.fillStyle = '#ccc'; ctx.font = '11px Courier New'; ctx.textAlign = 'center';
    ctx.fillText('No data yet', W / 2, H / 2);
    return;
  }

  let xMn = Math.min(...pts.map(p => p.x)), xMx = Math.max(...pts.map(p => p.x));
  let yMn = opts.yMin != null ? opts.yMin : Math.min(...pts.map(p => p.y));
  let yMx = opts.yMax != null ? opts.yMax : Math.max(...pts.map(p => p.y));
  if (xMn === xMx) xMx = xMn + 1;
  if (opts.yMin == null) { const yP = (yMx - yMn) * 0.1 || 1; yMn -= yP; yMx += yP; }

  function toP(x, y) {
    return [pad.l + (x - xMn) / (xMx - xMn) * pW, pad.t + pH - (y - yMn) / (yMx - yMn) * pH];
  }

  // Axes
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + pH); ctx.lineTo(pad.l + pW, pad.t + pH);
  ctx.stroke();

  // Y ticks
  ctx.fillStyle = '#000'; ctx.font = '10px Courier New'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const yV = yMn + (yMx - yMn) * i / 4;
    const [, sy] = toP(xMn, yV);
    ctx.fillText(yV.toFixed(1), pad.l - 4, sy + 3);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, sy); ctx.lineTo(pad.l + pW, sy); ctx.stroke();
  }

  // X ticks
  ctx.textAlign = 'center';
  const xStep = Math.max(1, Math.ceil((xMx - xMn) / 6));
  for (let x = Math.ceil(xMn); x <= xMx; x += xStep) {
    const [sx] = toP(x, yMn);
    ctx.fillStyle = '#000'; ctx.fillText(x, sx, pad.t + pH + 14);
  }

  // Lines
  const defaultDash = [[], [6, 3], [2, 2], [8, 2, 2, 2]];
  datasets.forEach((ds, di) => {
    if (ds.data.length < 2) return;
    ctx.setLineDash(ds.dash || defaultDash[di % defaultDash.length] || []);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.3;
    ctx.beginPath();
    ds.data.forEach((p, i) => {
      const [sx, sy] = toP(p.x, p.y);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Legend
  ctx.font = '10px Courier New'; ctx.textAlign = 'left';
  datasets.forEach((ds, di) => {
    const lx = pad.l + 8 + di * 90, ly = pad.t + 10;
    ctx.setLineDash(ds.dash || defaultDash[di % defaultDash.length] || []);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#000'; ctx.fillText(ds.label, lx + 22, ly + 3);
  });
}

// ---------------------------------------------------------------------------
// Fetch chart data from API
// ---------------------------------------------------------------------------
function fetchCharts() {
  fetch('/api/elo').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('elo-chart'),
      [{ label: 'ELO', data: data.map(d => ({ x: d.iteration, y: d.elo })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/losses').then(r => r.json()).then(data => {
    if (!data.length) return;
    const ds = [{ label: 'total', data: data.filter(d => d.total != null).map(d => ({ x: d.iteration, y: d.total })), dash: [] }];
    if (data.some(d => d.value != null))
      ds.push({ label: 'value', data: data.filter(d => d.value != null).map(d => ({ x: d.iteration, y: d.value })), dash: [6, 3] });
    if (data.some(d => d.policy != null))
      ds.push({ label: 'policy', data: data.filter(d => d.policy != null).map(d => ({ x: d.iteration, y: d.policy })), dash: [2, 2] });
    drawLineChart(el('loss-chart'), ds, {});
  }).catch(() => {});

  fetch('/api/winrates').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('winrate-chart'), [
      { label: 'P0%', data: data.map(d => ({ x: d.iteration, y: d.p0 })), dash: [] },
      { label: 'P1%', data: data.map(d => ({ x: d.iteration, y: d.p1 })), dash: [6, 3] },
      { label: 'Draw%', data: data.map(d => ({ x: d.iteration, y: d.draw })), dash: [2, 2] },
    ], { yMin: 0, yMax: 100 });
  }).catch(() => {});

  fetch('/api/gamelength').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('gamelength-chart'),
      [{ label: 'Avg Moves', data: data.map(d => ({ x: d.iteration, y: d.avg_game_length })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/speed').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('speed-chart'),
      [{ label: 'Games/s', data: data.map(d => ({ x: d.iteration, y: d.games_per_sec })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/resources').then(r => r.json()).then(data => {
    const h = data.history || [];
    if (h.length < 2) return;
    drawLineChart(el('res-chart'), [
      { label: 'CPU%', data: h.map((d, i) => ({ x: i, y: d.cpu_pct })), dash: [] },
      { label: 'RAM%', data: h.map((d, i) => ({ x: i, y: d.ram_pct })), dash: [6, 3] },
    ], { yMin: 0, yMax: 100 });
  }).catch(() => {});
}

// ---------------------------------------------------------------------------
// Resource polling
// ---------------------------------------------------------------------------
setInterval(() => {
  fetch('/api/stats').then(r => r.json()).then(d => {
    if (d.cpu_pct != null) {
      el('s-cpu').textContent = Math.round(d.cpu_pct);
      el('cpu-fill').style.width = Math.round(d.cpu_pct) + '%';
    }
    if (d.ram_pct != null) {
      el('s-ram').textContent = Math.round(d.ram_pct);
      el('ram-fill').style.width = Math.round(d.ram_pct) + '%';
    }
  }).catch(() => {});
}, 5000);

// ---------------------------------------------------------------------------
// Resize handler
// ---------------------------------------------------------------------------
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { drawHex(); fetchCharts(); }, 150);
});

// ---------------------------------------------------------------------------
// Init — restore settings from localStorage
// ---------------------------------------------------------------------------
(function initSettings() {
  el('set-speed').value = settings.replaySpeed;
  el('set-speed-val').textContent = settings.replaySpeed + 'ms';
  el('set-dotsize').value = settings.dotSize;
  el('set-dotsize-val').textContent = settings.dotSize;
  el('set-radius').value = settings.emptyHexRadius;
  el('set-radius-val').textContent = settings.emptyHexRadius;
  el('set-autorefresh').checked = settings.autoRefresh;
})();
setTimeout(() => { drawHex(); fetchCharts(); }, 100);
fetch('/api/stats').then(r => r.json()).then(s => {
  el('s-games').textContent = s.total_games;
  el('s-elo').textContent = Math.round(s.current_elo);
}).catch(() => {});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Module-level singleton for standalone mode
# ---------------------------------------------------------------------------

_default_dashboard: Optional[Dashboard] = None


def get_dashboard(port: int = 5001) -> Dashboard:
    """Get or create the module-level Dashboard singleton."""
    global _default_dashboard
    if _default_dashboard is None:
        _default_dashboard = Dashboard(port=port)
    return _default_dashboard


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    dash = get_dashboard(port)
    dash.resource_monitor.start()

    ts = _dt.now().strftime("%H:%M:%S")
    print(f"[{ts}] HEX BOT -- Clean Training Dashboard")
    print(f"[{ts}] CPU cores: {multiprocessing.cpu_count()}")
    print(f"[{ts}] Open http://localhost:{port}")

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    dash.socketio.run(
        dash.app, host="0.0.0.0", port=port,
        debug=False, allow_unsafe_werkzeug=True, log_output=False,
    )
