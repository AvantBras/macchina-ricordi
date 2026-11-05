"""A control panel in the browser, for someone who will never open a terminal.

The page is a control surface, not an audio path: sound leaves through the
machine's own speakers via the engine, and the browser only sends knob positions
and draws what came back. Streaming audio into a tab would add a second buffer,
a second clock and a second thing that can drop out.

Built as a desktop app the program may have no window of its own — on macOS it
runs with no Dock icon — so the page carries everything a window would: adding a
model, and quitting. Opening the app a second time finds the first one and shows
its page again rather than starting another player on top of it.

Everything is inlined. No CDN, no fonts fetched at load: a program that only
works with an internet connection is not self-contained.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from macchina_ricordi.instrument import controls

from .box import BOX_SUFFIX, read_spec
from .engine import Engine
from .loader import find_boxes, load_player, recommended_knobs
from .places import models_dir, recordings_dir
from .player import Knobs
from .wav import write_wav

DEFAULT_PORT = 8731
MAX_UPLOAD_BYTES = 2_000_000_000
TITLE = "macchina ricordi"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>macchina ricordi</title>
<style>
:root{color-scheme:light dark;
  --ground:#eef0ee;--panel:#f7f8f7;--edge:#d5dad6;--ink:#171b19;--soft:#5c6660;
  --faint:#87908a;--signal:#a8641b;--phos:#2f7a6a;}
@media (prefers-color-scheme:dark){:root{
  --ground:#0e1211;--panel:#161b19;--edge:#293230;--ink:#e8ebe9;--soft:#9aa5a0;
  --faint:#6e7a75;--signal:#d99a3f;--phos:#5cb8a3;}}
*{box-sizing:border-box}
body{margin:0;padding:38px 20px 70px;background:var(--ground);color:var(--ink);
  font:400 16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:660px;margin-inline:auto}
header{display:flex;justify-content:space-between;align-items:baseline;gap:16px}
h1{font-size:1.6rem;letter-spacing:-.02em;margin:0 0 .1em}
.sub{color:var(--soft);margin:0 0 1.8em;font-size:.95rem}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:10px;padding:20px 22px;margin-bottom:18px}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
button{font:600 .95rem/1 inherit;border:0;border-radius:7px;padding:14px 26px;cursor:pointer;
  background:var(--phos);color:var(--ground)}
button.stop{background:var(--signal)}
button:disabled{opacity:.45;cursor:default}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--edge);padding:11px 16px}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--phos);outline-offset:3px}
select{font:inherit;padding:10px 12px;border-radius:7px;border:1px solid var(--edge);
  background:var(--ground);color:var(--ink);max-width:100%;flex:1 1 180px}
label{display:block;margin:20px 0 0}
.lab{display:flex;justify-content:space-between;align-items:baseline;font-size:.9rem;margin-bottom:3px}
.lab b{font-weight:600}
.lab span{color:var(--faint);font-variant-numeric:tabular-nums;font-size:.85rem}
.hint{color:var(--faint);font-size:.82rem;margin:4px 0 0}
input[type=range]{width:100%;accent-color:var(--phos)}
.stat{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:14px}
.stat div{font-size:.85rem;color:var(--faint)}
.stat b{display:block;font-size:1.25rem;color:var(--ink);font-variant-numeric:tabular-nums;
  font-weight:600;line-height:1.3}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--faint);margin-right:8px;vertical-align:middle}
.dot.on{background:var(--phos)}
.warn{color:var(--signal)}
.meter{height:6px;background:var(--edge);border-radius:4px;overflow:hidden;margin-top:14px}
.meter i{display:block;height:100%;background:var(--phos);width:0;transition:width .3s}
.empty{border-style:dashed}
.empty p{margin:0 0 12px;color:var(--soft)}
code{font-size:.85em}
h2{font-size:1.02rem;margin:0 0 .2em;letter-spacing:-.01em}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0 22px}
.bands{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0 22px}
.pins{margin-top:20px;padding-top:16px;border-top:1px solid var(--edge)}
.inline{display:flex;align-items:center;gap:8px;margin:0;font-size:.9rem;color:var(--soft)}
.inline select{flex:0 0 auto}
input:disabled{opacity:.4}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><main>
<header><h1>macchina ricordi</h1>
  <button id="quit" class="ghost" title="Stop the music and close the program">Quit</button></header>
<p class="sub" id="sub">&nbsp;</p>

<div class="card empty" id="empty" hidden>
  <p>No model yet. A model is a <code>.nrbox</code> file — add one to start.</p>
  <button id="add-first">Add a model…</button>
</div>

<div class="card">
  <div class="row">
    <button id="toggle">Play</button>
    <select id="model" aria-label="Model"></select>
    <button id="add" class="ghost">Add a model…</button>
    <button id="rec" class="ghost">Record</button>
  </div>
  <input type="file" id="file" accept=".nrbox" hidden>
  <div class="meter" aria-hidden="true"><i id="buf"></i></div>
  <p class="hint" id="status"><span class="dot"></span>stopped</p>
</div>

<div class="card">
  <label><span class="lab"><b>Wildness</b><span id="tv">1.00</span></span>
    <input type="range" id="temperature" min="0.3" max="2.0" step="0.05" value="1.0">
    <span class="hint">Low is careful and can drift into silence. High is unruly.</span></label>
  <label><span class="lab"><b>Variety</b><span id="kv">250</span></span>
    <input type="range" id="top_k" min="10" max="1000" step="10" value="250">
    <span class="hint">How many options it considers each step.</span></label>
</div>

<div class="card" id="steer-card">
  <h2>The past it listens to</h2>
  <p class="hint">Each round the model re-reads what just played and carries on from it.
    These change what it re-reads — never what you hear.</p>
  <div class="two">
    <label><span class="lab"><b>Memory</b><span id="tau_seconds_v">5.0 s</span></span>
      <input type="range" id="tau_seconds" min="0.5" max="5" step="0.5" value="5">
      <span class="hint">How much of what just played it re-reads.</span></label>
    <label><span class="lab"><b>Stride</b><span id="round_seconds_v">4.0 s</span></span>
      <input type="range" id="round_seconds" min="1" max="10" step="0.5" value="4">
      <span class="hint">How much new music each round adds. Longer answers the controls more slowly.</span></label>
  </div>
  <p class="hint warn" id="horizon" hidden></p>
  <div class="bands">
    <label class="steer"><span class="lab"><b>Low</b><span id="tilt_low_db_v">0 dB</span></span>
      <input type="range" id="tilt_low_db" min="-12" max="12" step="1.5" value="0"></label>
    <label class="steer"><span class="lab"><b>Mid</b><span id="tilt_mid_db_v">0 dB</span></span>
      <input type="range" id="tilt_mid_db" min="-12" max="12" step="1.5" value="0"></label>
    <label class="steer"><span class="lab"><b>High</b><span id="tilt_high_db_v">0 dB</span></span>
      <input type="range" id="tilt_high_db" min="-12" max="12" step="1.5" value="0"></label>
  </div>
  <label class="steer"><span class="lab"><b>Tape speed</b><span id="prompt_rate_v">1.00×</span></span>
    <input type="range" id="prompt_rate" min="0.9" max="1.1" step="0.01" value="1">
    <span class="hint">Plays the past back to it slower and lower, or faster and higher.</span></label>
  <p class="hint">Tilt and tape speed compound while held: the music drifts, and stays where it drifted
    when you bring them back.</p>
  <div class="row" style="margin-top:12px"><button id="neutral" class="ghost steer-btn">Back to neutral</button></div>
  <div class="row pins">
    <button id="pin" class="steer-btn">Pin this moment</button>
    <span class="hint" id="pins">nothing pinned yet</span>
  </div>
  <div class="row" style="margin-top:10px">
    <button id="light" class="ghost steer-btn">Bring it back, lightly</button>
    <button id="heavy" class="ghost steer-btn">Bring it back, strongly</button>
    <label class="inline">Recall a pin
      <select id="recap_every" class="steer-btn">
        <option value="0">never</option><option value="2">every 2 rounds</option>
        <option value="3">every 3 rounds</option><option value="4">every 4 rounds</option>
        <option value="6">every 6 rounds</option><option value="8">every 8 rounds</option>
      </select></label>
  </div>
  <p class="hint warn" id="no-encoder" hidden>This model file has no encoder, so tilt, tape speed and
    pins are off. Re-pack it with the current version to use them.</p>
</div>

<div class="card stat">
  <div>Speed margin<b id="xrt">—</b></div>
  <div>Worst round<b id="worst">—</b></div>
  <div>Dropouts<b id="under">—</b></div>
  <div>Playing for<b id="played">—</b></div>
</div>
<p class="hint" id="where"></p>
<script>
const $ = id => document.getElementById(id);
let playing = false, modelsKey = '', notice = '', noticeUntil = 0, timer = null, knobsSeen = false;
const KNOBS = ['temperature', 'top_k', 'tau_seconds', 'round_seconds',
               'tilt_low_db', 'tilt_mid_db', 'tilt_high_db', 'prompt_rate'];
const db = v => (+v > 0 ? '+' : '') + (+v).toFixed(1).replace('.0', '') + ' dB';
const fmt = {temperature: v => (+v).toFixed(2), top_k: v => v,
             tau_seconds: v => (+v).toFixed(1) + ' s', round_seconds: v => (+v).toFixed(1) + ' s',
             tilt_low_db: db, tilt_mid_db: db, tilt_high_db: db, prompt_rate: v => (+v).toFixed(2) + '×'};
const out = {temperature: 'tv', top_k: 'kv'};
const outFor = k => $(out[k] || (k + '_v'));
function show(k, v) { $(k).value = v; outFor(k).textContent = fmt[k](v); }
function say(text, seconds) { notice = text; noticeUntil = Date.now() + (seconds || 6) * 1000; render(); }
async function post(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
                               body: JSON.stringify(body || {})});
  return r.json();
}
KNOBS.forEach(k => {
  const el = $(k);
  el.addEventListener('input', () => { outFor(k).textContent = fmt[k](el.value); });
  el.addEventListener('change', () => post('/api/knobs', {[k]: +el.value}));
});
$('model').addEventListener('change', e => { say('loading model…', 30); post('/api/model', {path: e.target.value}); });
$('toggle').addEventListener('click', async () => {
  $('toggle').disabled = true;
  await post(playing ? '/api/stop' : '/api/play');
  $('toggle').disabled = false;
  tick();
});
$('rec').addEventListener('click', async () => {
  const r = await post('/api/record');
  if (r.saved) say('saved ' + r.saved, 15);
  tick();
});
$('add').addEventListener('click', () => $('file').click());
$('add-first').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', () => {
  const f = $('file').files[0];
  if (!f) return;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/models');
  xhr.setRequestHeader('X-Filename', encodeURIComponent(f.name));
  xhr.upload.onprogress = e => {
    if (e.lengthComputable) say('adding ' + f.name + ' — ' + Math.round(100 * e.loaded / e.total) + '%', 30);
  };
  xhr.onload = () => {
    let r = {}; try { r = JSON.parse(xhr.responseText || '{}'); } catch (e) {}
    say(xhr.status === 200 ? 'added ' + r.added : (r.error || 'That file could not be added.'), 12);
    $('file').value = ''; tick();
  };
  xhr.onerror = () => { say('The upload failed.', 12); $('file').value = ''; };
  xhr.send(f);
});
$('recap_every').addEventListener('change', e => post('/api/knobs', {recap_every: +e.target.value}));
$('neutral').addEventListener('click', async () => {
  await post('/api/neutral');
  ['tilt_low_db', 'tilt_mid_db', 'tilt_high_db'].forEach(k => show(k, 0));
  show('prompt_rate', 1);
  say('back to neutral — the music stays where it drifted', 8);
});
$('pin').addEventListener('click', async () => {
  const r = await post('/api/pin');
  say(r.error ? r.error : 'pinned what is playing now', 8); tick();
});
['light', 'heavy'].forEach(strength => $(strength).addEventListener('click', async () => {
  await post('/api/bring-back', {strength});
  say('bringing it back in the next round', 8);
}));
$('quit').addEventListener('click', async () => {
  clearInterval(timer);
  try { await post('/api/quit'); } catch (e) {}
  document.querySelector('main').innerHTML = '<h1>macchina ricordi</h1><p class="sub">'
    + 'Stopped. You can close this tab; open the app again to start it.</p>';
});
let last = null;
function render() {
  const s = last; if (!s) return;
  const models = s.models || [];
  $('empty').hidden = models.length > 0;
  $('toggle').disabled = models.length === 0;
  const key = models.map(m => m.path).join('|');
  if (key !== modelsKey) {
    modelsKey = key;
    $('model').innerHTML = '';
    models.forEach(m => {
      const o = document.createElement('option');
      o.value = m.path;
      o.textContent = m.name;
      $('model').appendChild(o);
    });
  }
  if (s.current) $('model').value = s.current;
  if (s.knobs && !knobsSeen) {
    knobsSeen = true;
    const eff = s.effective || {};
    KNOBS.forEach(k => {
      let v = s.knobs[k];
      if (v === null && k === 'tau_seconds') v = eff.tau;
      if (v === null && k === 'round_seconds') v = eff.round;
      if (v !== null && v !== undefined) show(k, v);
    });
    $('recap_every').value = String(s.knobs.recap_every || 0);
  }
  const steer = !!s.steering;
  document.querySelectorAll('.steer input, .steer-btn').forEach(el => { el.disabled = !steer || !models.length; });
  $('no-encoder').hidden = steer || !models.length;
  $('pins').textContent = s.pins ? s.pins + ' moment' + (s.pins > 1 ? 's' : '') + ' pinned' : 'nothing pinned yet';
  const eff = s.effective || {};
  const asked = +$('round_seconds').value;
  $('horizon').hidden = !(eff.round && asked - eff.round > 0.05);
  if (!$('horizon').hidden) $('horizon').textContent = 'This model was trained on ' + eff.horizon
    + ' s of music, so memory and stride have to fit inside that: the stride is ' + eff.round.toFixed(1) + ' s.';
  const h = s.health || {};
  $('toggle').textContent = playing ? 'Stop' : 'Play';
  $('toggle').className = playing ? 'stop' : '';
  $('rec').textContent = s.recording ? 'Stop recording' : 'Record';
  $('xrt').textContent = h.mean_xrt ? h.mean_xrt.toFixed(1) + '×' : '—';
  $('worst').textContent = h.worst_xrt ? h.worst_xrt.toFixed(1) + '×' : '—';
  $('under').textContent = h.underruns ?? '—';
  $('under').className = h.underruns ? 'warn' : '';
  const p = h.played_seconds || 0;
  $('played').textContent = p >= 60 ? Math.floor(p / 60) + ' min' : Math.round(p) + ' s';
  $('buf').style.width = Math.min(100, ((h.buffered_seconds || 0) / 8) * 100) + '%';
  const lr = s.last_round || {};
  let msg = playing ? 'playing' : (models.length ? 'stopped' : 'waiting for a model');
  if (s.error) msg = '<span class="warn"></span>';
  else if (playing && h.worst_xrt && h.worst_xrt < 1.2)
    msg = 'playing — <span class="warn">little margin left'
          + (lr.steered ? ', Back to neutral frees some' : '') + '</span>';
  if (playing && lr.steered) msg += lr.motif ? ' · bringing a pinned moment back' : ' · steering the past';
  $('status').innerHTML = '<span class="dot' + (playing ? ' on' : '') + '"></span>' + msg;
  if (s.error) $('status').querySelector('.warn').textContent = s.error;
  if (Date.now() < noticeUntil) {
    const n = document.createElement('span');
    n.textContent = ' · ' + notice;
    $('status').appendChild(n);
  }
  $('sub').textContent = s.model || 'no model loaded';
  $('where').textContent = 'Models are kept in ' + s.models_dir + ' · recordings in ' + s.recordings_dir;
}
function tick() {
  fetch('/api/state').then(r => r.json()).then(s => { last = s; playing = s.playing; render(); }).catch(() => {});
}
timer = setInterval(tick, 700); tick();
</script></main></body></html>"""


def _with_vocabulary(page: str) -> str:
    """Every slider's range and step from the vocabulary the stream shares (macchina_ricordi/instrument/controls.py)."""
    for c in controls.CONTROLS:
        low, high, step = c.panel
        page = re.sub(
            rf'(<input type="range" id="{c.key}") min="[^"]*" max="[^"]*" step="[^"]*"',
            rf'\1 min="{low:g}" max="{high:g}" step="{step:g}"',
            page,
        )
    return page


PAGE = _with_vocabulary(PAGE)


def install_box(stream, length: int, filename: str, dest: Path) -> Path:
    """Copy an uploaded file into the models folder, refusing anything that is not a box.

    The file is written beside its final name and only renamed once it has been
    read back as a box, so a half-finished or foreign upload can never appear in
    the model list.
    """
    if length <= 0:
        raise ValueError("That file is empty.")
    if length > MAX_UPLOAD_BYTES:
        raise ValueError("That file is too large to be a model.")
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(filename or "model").name).strip(" .") or "model"
    if not name.endswith(BOX_SUFFIX):
        name = name.rsplit(".", 1)[0] + BOX_SUFFIX
    dest.mkdir(parents=True, exist_ok=True)
    partial = dest / (name + ".partial")
    remaining = length
    with open(partial, "wb") as out:
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                break
            out.write(chunk)
            remaining -= len(chunk)
    if remaining:
        partial.unlink(missing_ok=True)
        raise ValueError("The upload was interrupted.")
    try:
        read_spec(partial)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise ValueError("That file is not a macchina-ricordi model (.nrbox).") from exc
    target = dest / name
    partial.replace(target)
    return target


class Session:
    """One model, one engine, one audio stream — swappable while the page stays open."""

    def __init__(
        self, roots: list[Path] | None = None, *, device=None, threads: int | None = None, fast: bool = False
    ) -> None:
        self.roots = [Path(r) for r in roots] if roots else None
        self.device = device
        self.threads = threads
        self.fast = fast
        self.lock = threading.Lock()
        self.engine: Engine | None = None
        self.spec = None
        self.current: Path | None = None
        self.playing = False
        self.error: str | None = None
        self.boxes: list[Path] = []
        self._stream = None
        self._knobs = Knobs()
        self._touched = False
        self._recording: list[np.ndarray] | None = None
        self._last_round: dict = {}
        self.rescan()
        if self.boxes:
            try:
                self.load(self.boxes[0])
            except Exception as exc:  # a broken box must not stop the panel from opening
                self.error = f"could not open {self.boxes[0].name}: {exc}"

    def rescan(self) -> list[Path]:
        found = find_boxes(*self.roots, models_dir()) if self.roots else find_boxes()
        with self.lock:
            self.boxes = found
        return found

    def load(self, path: Path) -> None:
        with self.lock:
            was_playing = self.playing
        if was_playing:
            self.stop()
        player, spec = load_player(Path(path), threads=self.threads, fast=self.fast)
        knobs = recommended_knobs(spec, self._knobs if self._touched else None)
        engine = Engine(player, knobs=knobs)
        engine.on_round(self._note_round)
        with self.lock:
            self._knobs = knobs
            self.engine = engine
            self._last_round = {}
            self.spec = spec
            self.current = Path(path)
            self.error = None
        if was_playing:
            self.play()

    def _note_round(self, info: dict) -> None:
        keys = ("steered", "steer_skipped", "rate_applied", "motif", "tau_s", "round_s", "revived")
        self._last_round = {k: info[k] for k in keys if k in info}

    def add_model(self, stream, length: int, filename: str) -> Path:
        path = install_box(stream, length, filename, models_dir())
        self.rescan()
        try:
            self.load(path)
        except Exception as exc:
            with self.lock:
                self.error = f"added {path.name}, but it would not open: {exc}"
        return path

    def set_knobs(self, changes: dict) -> None:
        with self.lock:
            engine = self.engine
        allowed = {k: v for k, v in changes.items() if k in Knobs.__dataclass_fields__}
        if not allowed:
            return
        self._touched = True
        if engine:
            self._knobs = engine.set_knobs(**allowed)
        else:
            self._knobs = Knobs(**(self._knobs.__dict__ | allowed)).clamped()

    def play(self) -> None:
        with self.lock:
            if self.playing or self.engine is None:
                return
            engine = self.engine
        try:
            import sounddevice as sd

            engine.prefill()
            engine.start()

            def callback(outdata, frames, _time, status):  # noqa: ANN001
                audio = engine.pull(frames)
                if self._recording is not None:
                    self._recording.append(audio.copy())
                outdata[:, 0] = audio

            stream = sd.OutputStream(
                samplerate=engine.player.spec.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=2048,
                device=self.device,
                callback=callback,
            )
            stream.start()
        except Exception as exc:
            with self.lock:
                self.error = f"audio device unavailable: {exc}"
            engine.stop()
            return
        with self.lock:
            self._stream = stream
            self.playing = True
            self.error = None

    def stop(self) -> None:
        with self.lock:
            stream, engine = self._stream, self.engine
            self._stream = None
            self.playing = False
        if stream:
            stream.stop()
            stream.close()
        if engine:
            engine.stop()

    def pin(self) -> int:
        with self.lock:
            engine, playing = self.engine, self.playing
        if engine is None:
            raise ValueError("no model loaded")
        return engine.pin(playing=playing)

    def bring_back(self, strength: str) -> None:
        with self.lock:
            engine = self.engine
        if engine is None or not engine.player.motifs:
            raise ValueError("pin a moment first")
        engine.player.bring_back("heavy" if strength == "heavy" else "light")

    def neutral(self) -> None:
        self.set_knobs(controls.neutral())

    def toggle_record(self) -> str | None:
        if self._recording is None:
            self._recording = []
            return None
        chunks, self._recording = self._recording, None
        if not chunks or self.spec is None:
            return None
        name = f"{self.spec.name}-{datetime.now():%Y%m%d-%H%M%S}.wav"
        path = write_wav(recordings_dir() / name, np.concatenate(chunks), self.spec.sample_rate)
        return str(path)

    def state(self) -> dict:
        with self.lock:
            engine, spec = self.engine, self.spec
            return {
                "playing": self.playing,
                "recording": self._recording is not None,
                "error": self.error,
                "model": (f"{spec.name} · {spec.dim}d/{spec.num_layers}L · {spec.sample_rate / 1000:g} kHz")
                if spec
                else "",
                "current": str(self.current) if self.current else "",
                "models": [{"path": str(p), "name": p.stem} for p in self.boxes],
                "models_dir": str(models_dir()),
                "recordings_dir": str(recordings_dir()),
                "health": engine.health.snapshot() if engine else {},
                "knobs": self._knobs.__dict__,
                "revivals": engine.player.revivals if engine else 0,
                "steering": engine.player.can_steer if engine else False,
                "pins": len(engine.player.motifs) if engine else 0,
                "last_round": self._last_round,
                "effective": self._effective(engine, spec),
            }

    def _effective(self, engine, spec) -> dict:
        """Memory and stride as they will actually be used, after the model's horizon."""
        if engine is None or spec is None:
            return {}
        context, fresh = engine.player.frames_for(self._knobs)
        return {
            "tau": context / spec.frame_rate,
            "round": fresh / spec.frame_rate,
            "horizon": round(spec.segment_duration, 1),
        }


def _handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # keep the console for the stream
            pass

        def _send(self, payload: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, obj, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self) -> None:
            if self.path.startswith("/api/state"):
                self._json(session.state())
            elif self.path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            # Any web page can make the browser send a request to 127.0.0.1. Only
            # this panel's own page may stop the music, add files, or quit.
            origin = self.headers.get("Origin")
            if origin and origin not in self.server.allowed_origins:
                self.close_connection = True
                self._json({"error": "requests from other sites are not accepted"}, 403)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if self.path == "/api/models":
                filename = urllib.parse.unquote(self.headers.get("X-Filename") or "model.nrbox")
                try:
                    path = session.add_model(self.rfile, length, filename)
                except ValueError as exc:
                    self.close_connection = True
                    self._json({"error": str(exc)}, 400)
                    return
                self._json({"added": path.stem, **session.state()})
                return
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if self.path == "/api/play":
                threading.Thread(target=session.play, daemon=True).start()
            elif self.path == "/api/stop":
                session.stop()
            elif self.path == "/api/knobs":
                session.set_knobs(body)
            elif self.path == "/api/model":
                threading.Thread(target=_load_quietly, args=(session, Path(body["path"])), daemon=True).start()
            elif self.path == "/api/pin":
                try:
                    self._json({"pinned": session.pin()})
                except ValueError as exc:
                    self._json({"error": str(exc)}, 409)
                return
            elif self.path == "/api/bring-back":
                try:
                    session.bring_back(str(body.get("strength", "light")))
                except ValueError as exc:
                    self._json({"error": str(exc)}, 409)
                    return
            elif self.path == "/api/neutral":
                session.neutral()
            elif self.path == "/api/record":
                self._json({"saved": session.toggle_record()})
                return
            elif self.path == "/api/quit":
                session.stop()
                self._json({"quitting": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            else:
                self.send_error(404)
                return
            self._json(session.state())

    return Handler


def _load_quietly(session: Session, path: Path) -> None:
    try:
        session.load(path)
    except Exception as exc:
        with session.lock:
            session.error = f"could not open {path.name}: {exc}"


def make_server(session: Session, port: int) -> ThreadingHTTPServer:
    """Bind the panel, falling back to any free port if another program holds ``port``."""
    handler = _handler(session)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    bound = server.server_address[1]
    server.allowed_origins = {f"http://127.0.0.1:{bound}", f"http://localhost:{bound}"}
    return server


def already_running(port: int) -> bool:
    """Is this panel — not just anything — already answering on ``port``?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1.5) as response:
            return "health" in json.loads(response.read())
    except Exception:
        return False


def serve(args) -> int:
    port = getattr(args, "port", None) or DEFAULT_PORT
    if already_running(port):
        url = f"http://127.0.0.1:{port}/"
        print(f"already running at {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    session = Session(args.boxes or None, device=args.device, threads=args.threads, fast=getattr(args, "fast", False))
    server = make_server(session, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    count = len(session.boxes)
    print(f"panel at {url}  ({count} model{'s' if count != 1 else ''})", flush=True)
    if not count:
        print(f"no models yet: add one from the page, or put .nrbox files in {models_dir()}", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        session.stop()
        server.server_close()
    return 0
