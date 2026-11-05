// Scène: play a running stream from the browser. The page is a control surface, never an audio path: the stream's
// own process plays the sound, so nothing here can make it drop out. It draws what `play monitor` draws and edits
// control.json the way the TUI's keys do (same keys, same steps, same lock), through /api/stage.
(function () {
  "use strict";
  const NR = window.NR;
  const { esc, fmt } = NR;
  const PREFS = "nr.stage";
  const prefs = Object.assign({ follow: "generated", blind: false },
    NR.storage(() => JSON.parse(localStorage.getItem(PREFS) || "{}"), {}));
  const savePrefs = () => NR.storage(() => localStorage.setItem(PREFS, JSON.stringify(prefs)));
  const PLAYING = ["live", "stopping"];
  const STOP_CONFIRM = "Stop the stream? It finishes the round in progress, plays out what is buffered and writes its manifest.";
  const PHASES = { starting: "starting", waiting: "starting round", throttled: "throttled — buffer full", resting: "resting before the round", generating: "generating", scoring: "scoring", stopped: "stopped" };
  const STATES = { live: "live", stopping: "stopping", ended: "ended", unfinished: "ended without a manifest" };
  const GLYPHS = { up: "↑", down: "↓", left: "←", right: "→" };
  const LABELS = {
    temperature: "temperature", top_k: "top_k", candidates: "candidates", tau_seconds: "tau",
    completion_duration_seconds: "completion", novelty_weight: "novelty", stuck_weight: "stuck", stuck_kick_rounds: "stuck kick",
    prompt_tilt_low_db: "tilt low", prompt_tilt_mid_db: "tilt mid", prompt_tilt_high_db: "tilt high", prompt_rate: "rate",
    round_delay_seconds: "rest",
  };
  const UNITS = { tau_seconds: "s", completion_duration_seconds: "s", prompt_tilt_low_db: "dB", prompt_tilt_mid_db: "dB", prompt_tilt_high_db: "dB", round_delay_seconds: "s" };
  const OFF_AT_ZERO = ["top_k", "stuck_kick_rounds", "round_delay_seconds"];
  // Where a bipolar knob is flat: the slider marks it, and a knob away from it reads as engaged.
  const NEUTRAL = { prompt_tilt_low_db: 0, prompt_tilt_mid_db: 0, prompt_tilt_high_db: 0, prompt_rate: 1 };
  const GROUPS = [
    ["Sampling", "", ["temperature", "top_k", "candidates"]],
    ["Window", "seconds re-read · seconds added per round", ["tau_seconds", "completion_duration_seconds"]],
    ["Ranking & drones", "novelty and stuck rank best-of-N takes, inert at one candidate · a kick runs one hotter round after that many drones", ["novelty_weight", "stuck_weight", "stuck_kick_rounds"]],
    ["Prompt steering", "reshapes the past the model re-reads · compounds while held", ["prompt_tilt_low_db", "prompt_tilt_mid_db", "prompt_tilt_high_db", "prompt_rate"]],
    ["Pacing", "a rest before each round, to set its knobs · cut short before the buffer runs dry", ["round_delay_seconds"]],
  ];
  const CHARTS = [["arc", "arc coherence"], ["novelty", "novelty"], ["silence", "silence"], ["buffer", "buffer ahead, s"]];
  // What a round row records, as a glyph over its tile on the ribbon.
  const MARKS = {
    pinned: ["●", "pinned"], reroll: ["↻", "rerolled"], pinned_injection: ["◆", "pinned motif injected"], motif_recap: ["◇", "motif recap"],
    stuck_kick: ["↯", "stuck kick"], tail_rescue: ["▲", "tail rescued"], model_swap: ["⇄", "model swap"], generation_slow_warning: ["!", "slower than real time"],
  };
  const GENERATING = ["generating", "scoring"];
  const LAUNCH = "nr.stage.launch";
  // The knobs a launch starts at (the server knows their flags); every one stays live on the session's page.
  const START = [
    ["completion_duration_seconds", "completion, s", "fresh audio each round"],
    ["tau_seconds", "tau, s", "seconds of the last round re-read"],
    ["temperature", "temperature", ""],
    ["top_k", "top_k", "0 = off"],
    ["candidates", "candidates", "best-of-N"],
    ["round_delay_seconds", "rest, s", "before each round · 0 = off"],
  ];

  let pollId = 0;
  let pollTimer = null;
  let raf = 0;
  let page = null; // the open session: { id, frame, fetchedAt, model, chartsKey, dragging, flash, trail }

  function stopPolling() { pollId++; clearTimeout(pollTimer); cancelAnimationFrame(raf); }
  function every(ms, fn) {
    const id = ++pollId;
    const tick = async () => {
      if (id !== pollId) return;
      try { await fn(); } catch (_) { /* keep polling through blips */ }
      if (id === pollId) pollTimer = setTimeout(tick, document.visibilityState === "hidden" ? ms * 4 : ms);
    };
    pollTimer = setTimeout(tick, ms);
  }
  window.addEventListener("hashchange", () => { stopPolling(); page = null; });

  // ---------- small formatting ----------
  const sigOf = (path) => (path ? String(path).replace(/\/+$/, "").split("/").pop() : null);
  const xpOf = (path) => NR.state.bySig.get(sigOf(path));
  const nameOf = (path) => { const xp = xpOf(path); return (xp ? NR.title(xp) : sigOf(path) || "base model").replace(/_/g, " "); };
  const glyph = (key) => GLYPHS[key] || key;
  function secs(s) {
    if (s == null || !isFinite(s) || s < 0) return "—";
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 5400) return `${Math.round(s / 60)} min`;
    return `${Math.floor(s / 3600)} h ${String(Math.round((s % 3600) / 60)).padStart(2, "0")}`;
  }
  function when(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
  }
  // The keymap's Python format specs: .2f, .1f, +.1f, .3f, d.
  function spec(value, format) {
    if (value == null || !isFinite(value)) return "—";
    if (format === "d") return String(Math.round(value));
    const text = Number(value).toFixed(Number((format.match(/\.(\d+)f/) || [])[1] || 2));
    return format.startsWith("+") && value >= 0 ? `+${text}` : text;
  }
  function knobValue(knob, value) {
    if (OFF_AT_ZERO.includes(knob.field) && value === 0) return "off";
    return spec(value, knob.fmt) + (UNITS[knob.field] ? ` ${UNITS[knob.field]}` : "");
  }
  function stateChip(state) {
    const look = state === "live" ? "running" : state === "stopping" ? "stopping" : "stopped";
    return `<span class="state state-${look}">${state === "live" ? '<span class="live-dot"></span>' : ""}${esc(STATES[state] || state)}</span>`;
  }
  function coverOf(path) {
    const xp = xpOf(path);
    return xp ? NR.coverHtml(xp, { subtitle: false }) : `<div class="cover" style="background:var(--ink)"></div>`;
  }

  // ---------- the list ----------
  async function viewList() {
    NR.setNav("stage");
    stopPolling();
    page = null;
    await NR.loadXps();
    const overview = await NR.getJSON("/api/stage");
    NR.view.innerHTML = `<div class="stage">
      <div class="listen-intro">
        <h1 class="title-xl">Scène</h1>
        <p>Start a stream here, or in a terminal, and play it with every control and reading of <code>play stream --tui</code>.
          The sound stays in the stream's own process: this page only edits its <code>control.json</code>, the way the TUI's keys do.</p>
      </div>
      <div class="section"><h3>En scène <span class="dim" id="st-live-note"></span></h3><div id="st-live" class="run-cards"></div></div>
      <div class="section"><h3>Entrée en scène <span class="dim">plays through this machine's speakers, and keeps playing if you close the page</span></h3>
        ${launchForm(overview)}</div>
      <div class="section"><h3>Archives <span class="dim">every stream session, newest first · open one to read its last frame</span></h3><div id="st-past"></div></div>
    </div>`;
    bindLaunch();
    let shown = 30;
    let sessions = overview.sessions;
    const draw = () => {
      const playing = sessions.filter((s) => PLAYING.includes(s.state));
      const past = sessions.filter((s) => !PLAYING.includes(s.state));
      document.getElementById("st-live-note").textContent = playing.length ? `${playing.length} playing` : "";
      document.getElementById("st-live").innerHTML = playing.length
        ? playing.map((s) => `<a class="run-card" href="#/stage/${encodeURIComponent(s.id)}">
            <div class="mini">${coverOf(s.model)}</div>
            <div class="run-card-body">
              <div>${stateChip(s.state)} ${s.phase ? `<span class="dim">${esc(PHASES[s.phase] || s.phase)}</span>` : ""}</div>
              <div class="run-card-title">${esc(nameOf(s.model))}</div>
              <div class="dim">${esc(s.id)} · ${s.rounds} round${s.rounds === 1 ? "" : "s"} · since ${esc(when(s.started_at))}</div>
            </div></a>`).join("")
        : `<p class="dim">Nothing is playing.</p>`;
      const length = (s) => (s.started_at && s.finished_at ? secs((Date.parse(s.finished_at) - Date.parse(s.started_at)) / 1000) : "—");
      document.getElementById("st-past").innerHTML = past.length ? `<div class="table-wrap"><table class="stage-past"><thead><tr>
          <th>session</th><th>model</th><th class="r">rounds</th><th>started</th><th class="r">length</th><th>ended</th></tr></thead><tbody>
          ${past.slice(0, shown).map((s) => `<tr data-href="#/stage/${encodeURIComponent(s.id)}">
            <td>${esc(s.id)}</td><td>${esc(nameOf(s.model))} <span class="dim">${esc(s.model || "")}</span></td>
            <td class="r num">${s.rounds}</td><td class="dim">${esc(when(s.started_at))}</td><td class="r dim num">${length(s)}</td>
            <td class="dim">${s.state === "unfinished" ? "no manifest" : esc(s.stopped_by || "")}</td></tr>`).join("")}
          </tbody></table></div>${past.length > shown ? `<button class="more" id="st-more">${past.length - shown} more</button>` : ""}`
        : `<p class="dim">No stream has been played yet.</p>`;
      const more = document.getElementById("st-more");
      if (more) more.addEventListener("click", () => { shown += 60; draw(); });
    };
    draw();
    every(4000, async () => { sessions = (await NR.getJSON("/api/stage")).sessions; draw(); });
  }

  // ---------- starting one ----------
  const savedLaunch = () => NR.storage(() => JSON.parse(localStorage.getItem(LAUNCH) || "{}"), {});

  function launchForm(o) {
    const byField = new Map(o.knobs.map((k) => [k.field, k]));
    const saved = savedLaunch();
    const start = (field) => saved[field] ?? o.defaults[field];
    return `<form id="st-launch" class="ab-form" autocomplete="off">
      <fieldset><legend>Models <em>the first one starts · the number keys swap between them</em></legend>
        <div class="picked" id="st-picks"></div>
        <div class="pick-add">
          <input id="st-model" placeholder="type a name, label, dataset or sig (midwife@best for a checkpoint)">
          <button type="button" class="ghost" id="st-add">add</button>
        </div>
      </fieldset>
      <fieldset class="grid-2"><legend>Start at <em>every knob stays live on the session's page</em></legend>
        ${START.map(([field, label, hint]) => {
          const k = byField.get(field);
          return `<label><span>${label}${hint ? ` <em>${hint}</em>` : ""}</span>
            <input type="number" name="${field}" min="${k.low}" max="${k.high}" step="${k.step}" value="${esc(start(field))}" required></label>`;
        }).join("")}
        <label><span>buffer cap, s <em>blank: no throttle</em></span>
          <input type="number" name="max_buffer_ahead_seconds" min="5" max="300" step="5" value="${esc(saved.max_buffer_ahead_seconds ?? "")}"></label>
        <label><span>buffer floor, × cap <em>blank: cap − 10 s</em></span>
          <input type="number" name="buffer_floor_ratio" min="0" max="1" step="0.1" value="${esc(saved.buffer_floor_ratio ?? "")}"></label>
      </fieldset>
      <div id="st-plan"></div>
      <div class="form-actions"><button class="solid big" type="submit" id="st-play" disabled>Play</button><span class="form-msg" id="st-launch-msg"></span></div>
      <pre class="log" id="st-launch-log" hidden></pre>
    </form>`;
  }

  function pickName(text) {
    const [base, selector] = text.split("@");
    const xp = NR.state.bySig.get(base);
    return xp ? `${NR.title(xp).replace(/_/g, " ")}${selector ? ` @${selector}` : ""}` : text;
  }

  function planHtml(plan) {
    const models = plan.models.map((m, i) => `<div class="plan-model${m.sig ? "" : " bad"}">
        <span class="mini">${m.sig ? coverOf(m.folder) : ""}</span>
        <span><kbd>${i + 1}</kbd> ${esc(m.sig ? nameOf(m.folder) : m.text)}<br>
          <span class="dim small">${m.sig ? `${esc(m.sig)} · ${esc(m.checkpoint)}${m.segment_duration ? ` · ${m.segment_duration} s segments` : ""}` : "not found"}</span>
          ${m.note ? `<br><span class="dim small">${esc(m.note)}</span>` : ""}</span>
      </div>`).join("");
    const line = (warning, text) => (warning ? `<p><span class="err">⚠</span> ${esc(text)}</p>` : `<p class="dim">· ${esc(text)}</p>`);
    return `<div class="plan-models">${models}</div>
      ${plan.errors.map((e) => `<p class="err">${esc(e)}</p>`).join("")}
      ${plan.warnings.map((w) => line(true, w)).join("")}
      ${plan.guidance.map((a) => line(a.level === "warning", a.message)).join("")}
      ${NR.gpuHtml ? NR.gpuHtml(plan.gpu) : ""}
      <details><summary>the command it runs</summary><pre class="cmd">${esc(plan.command)}</pre></details>`;
  }

  function bindLaunch() {
    const form = document.getElementById("st-launch");
    const play = document.getElementById("st-play");
    const msg = document.getElementById("st-launch-msg");
    const logEl = document.getElementById("st-launch-log");
    const saved = savedLaunch();
    const picks = Array.isArray(saved.models) ? saved.models.slice(0, 9) : [];
    let plan = null;
    let planned = 0;
    let timer = null;
    const request = () => {
      const r = { models: picks };
      form.querySelectorAll('input[type="number"]').forEach((el) => (r[el.name] = el.value === "" ? null : Number(el.value)));
      return r;
    };
    const save = () => NR.storage(() => localStorage.setItem(LAUNCH, JSON.stringify(request())));
    const syncPlay = () => {
      play.disabled = !plan || plan.errors.length > 0;
      play.textContent = plan && plan.warnings.length ? "Play anyway" : "Play";
    };
    const drawPicks = () => {
      document.getElementById("st-picks").innerHTML = picks.length
        ? picks.map((text, i) => `<span class="pill"><kbd>${i + 1}</kbd>${esc(pickName(text))}<button type="button" class="unpick" data-unpick="${i}" aria-label="remove ${esc(text)}">✕</button></span>`).join("")
        : `<span class="dim">no model yet: add one below</span>`;
    };
    const preview = () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        const seq = ++planned;
        const box = document.getElementById("st-plan");
        if (!picks.length) { plan = null; box.innerHTML = ""; syncPlay(); return; }
        try {
          const result = await NR.postJSON("/api/stage/preview", { request: request() });
          if (seq !== planned) return;
          plan = result;
          box.innerHTML = planHtml(result);
        } catch (err) {
          if (seq !== planned) return;
          plan = { errors: [err.message], warnings: [] };
          box.innerHTML = `<p class="err">${esc(err.message)}</p>`;
        }
        syncPlay();
      }, 300);
    };
    const changed = () => { drawPicks(); save(); preview(); };
    const add = () => {
      const input = document.getElementById("st-model");
      const text = input.value.trim();
      if (!text) return;
      if (picks.length >= 9) { msg.textContent = "nine models at most: one per number key"; return; }
      picks.push(text);
      input.value = "";
      changed();
    };
    NR.modelPicker(document.getElementById("st-model"), { models: NR.state.xps.filter((x) => x.checkpoint_count), onPick: add });
    form.addEventListener("click", (ev) => {
      if (ev.target.closest("#st-add")) add();
      const unpick = ev.target.closest("[data-unpick]");
      if (unpick) { picks.splice(Number(unpick.dataset.unpick), 1); changed(); }
    });
    form.addEventListener("keydown", (ev) => {
      if (ev.key !== "Enter" || ev.target.tagName !== "INPUT") return;
      ev.preventDefault(); // Enter adds a model; it never starts a stream by accident
      if (ev.target.id === "st-model") add();
    });
    form.addEventListener("input", (ev) => { if (ev.target.type === "number") { save(); preview(); } });
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      play.disabled = true;
      msg.classList.remove("err");
      msg.textContent = "starting… the model loads on the session's page";
      logEl.hidden = true;
      try {
        const launched = await NR.postJSON("/api/stage/launch", { request: request() });
        location.hash = `#/stage/${encodeURIComponent(launched.id)}`;
      } catch (err) {
        const [first, ...rest] = err.message.split("\n");
        msg.textContent = first;
        msg.classList.add("err");
        logEl.textContent = rest.join("\n");
        logEl.hidden = !rest.length;
        syncPlay();
      }
    });
    drawPicks();
    preview();
  }

  // ---------- a session ----------
  const fetchFrame = (id) => NR.getJSON(`/api/stage/${encodeURIComponent(id)}?follow=${encodeURIComponent(prefs.follow)}`);

  async function viewSession(id) {
    NR.setNav("stage");
    stopPolling();
    await NR.loadXps();
    let frame;
    try {
      frame = await fetchFrame(id);
    } catch (err) {
      NR.view.innerHTML = `<div class="empty">${esc(err.message)}<br><a href="#/stage">Scène</a></div>`;
      return;
    }
    page = { id, frame, fetchedAt: performance.now(), model: undefined, chartsKey: null, dragging: null, flash: null, trail: [] };
    NR.view.innerHTML = skeleton(frame);
    bind();
    draw(frame);
    if (PLAYING.includes(frame.state)) {
      every(1000, async () => { if (page && page.id === id) draw(await fetchFrame(id)); });
      const loop = () => { if (!page || page.id !== id) return; tick(); raf = requestAnimationFrame(loop); };
      raf = requestAnimationFrame(loop);
    }
  }

  function knobRow(knob) {
    const label = LABELS[knob.field] || knob.field;
    return `<div class="knob" data-field="${esc(knob.field)}">
      <span class="lbl">${esc(label)}<kbd>${esc(glyph(knob.cut))}</kbd><kbd>${esc(glyph(knob.boost))}</kbd></span>
      <button class="nudge" data-key="${esc(knob.cut)}" aria-label="${esc(label)} down">−</button>
      <input type="range" min="${knob.low}" max="${knob.high}" step="${knob.step}" aria-label="${esc(label)}"${knob.field in NEUTRAL ? ` class="bipolar" style="--neutral:${(NEUTRAL[knob.field] - knob.low) / (knob.high - knob.low)}"` : ""}>
      <button class="nudge" data-key="${esc(knob.boost)}" aria-label="${esc(label)} up">+</button>
      <output class="val num"></output>
    </div>`;
  }

  function skeleton(f) {
    const byField = new Map(f.knobs.map((k) => [k.field, k]));
    const groups = GROUPS.map(([title, note, fields]) => `<div class="knob-group">
        <h4>${title}${title === "Prompt steering" ? `<button class="ghost" data-key="0">flat <kbd>0</kbd></button>` : ""}</h4>
        ${note ? `<p class="dim small">${note}</p>` : ""}
        ${fields.filter((field) => byField.has(field)).map((field) => knobRow(byField.get(field))).join("")}
      </div>`).join("");
    return `<div class="stage stage-session">
      <div class="crumb"><a href="#/stage">Scène</a> / ${esc(f.id)}</div>
      <div class="stage-head">
        <a class="mini" id="st-cover"></a>
        <div><h1 class="title-xl" id="st-title"></h1><div class="byline" id="st-byline"></div></div>
        <div class="stage-modes">
          <span class="seg"><span>readings</span><button data-follow="generated">generated</button><button data-follow="playing">playing</button><kbd>f</kbd></span>
          <button class="ghost" id="st-blind" aria-pressed="false">blind <kbd>b</kbd></button>
          <button class="ghost" id="st-help-open">keys <kbd>?</kbd></button>
        </div>
      </div>
      <div class="stage-now" id="st-now" aria-live="polite"></div>
      <div class="ribbon" id="st-ribbon" aria-hidden="true"></div>
      <div class="ribbon-key dim small" id="st-ribbon-key"></div>
      <div class="stage-grid">
        <section class="stage-controls" aria-label="controls">
          ${groups}
          <div class="knob-group"><h4>Buffer</h4>
            <p class="dim small">how far generation may run ahead of the speakers · smaller lands every edit sooner</p>
            <div class="buffer-controls">
              <span class="lbl">cap<kbd>-</kbd><kbd>+</kbd></span>
              <button class="nudge" data-key="-" aria-label="buffer cap down">−</button>
              <output class="val num" id="st-cap"></output>
              <button class="nudge" data-key="+" aria-label="buffer cap up">+</button>
              <span class="lbl">floor<kbd>{</kbd><kbd>}</kbd></span>
              <button class="nudge" data-key="{" aria-label="buffer floor down">−</button>
              <output class="val num" id="st-floor"></output>
              <button class="nudge" data-key="}" aria-label="buffer floor up">+</button>
              <button class="ghost" data-key="t" id="st-throttle" aria-pressed="true">throttle <kbd>t</kbd></button>
            </div>
          </div>
          <div class="knob-group"><h4>Gestures</h4>
            <div class="gestures">
              <button class="ghost" data-key="r">reroll <kbd>r</kbd></button>
              <button class="ghost" data-key="p">pin what is playing <kbd>p</kbd></button>
              <button class="ghost" data-key="i">inject pinned, light <kbd>i</kbd></button>
              <button class="ghost" data-key="I">inject pinned, heavy <kbd>I</kbd></button>
              <button class="ghost danger" data-key="Q">stop <kbd>Q</kbd></button>
            </div>
          </div>
          <div class="knob-group" id="st-models-group" hidden><h4>Models <span class="dim small">from --playlist</span></h4><div class="gestures" id="st-models"></div></div>
        </section>
        <section class="stage-readings" aria-label="readings">
          <div class="knob-group"><h4><span id="st-focus"></span></h4>
            <div id="st-flags" class="dim small"></div>
            <div id="st-metrics"></div>
            <div id="st-cands"></div>
          </div>
          <div class="knob-group"><h4>History <span class="dim small" id="st-history-note"></span></h4><div class="spark-grid" id="st-charts"></div></div>
        </section>
      </div>
      <div class="section"><h3>Guidance <span class="dim">what the stream-statistics corpus says about these settings on this model · never acted on</span></h3><div id="st-guidance"></div></div>
      <div class="section"><h3>Events <span class="dim" id="st-events-note"></span></h3><div class="stage-events" id="st-events"></div></div>
      <div class="section" id="st-log-box" hidden><h3>Log <span class="dim" id="st-log-note"></span></h3>
        <pre class="log" id="st-log"></pre>
        <details><summary>the command it runs</summary><pre class="cmd" id="st-command"></pre></details></div>
      <dialog class="guide stage-help" id="st-help">
        <div class="guide-head"><h3>Keys</h3><span></span><button type="button" class="ghost" id="st-help-close">Close</button></div>
        <p class="dim">The TUI's keys, one for one (Tab excepted: the browser needs it). They act while no text field has the focus.</p>
        <table class="kv">${helpRows(f).map(([keys, what]) => `<tr><td>${keys}</td><td>${what}</td></tr>`).join("")}</table>
      </dialog>
    </div>`;
  }

  function helpRows(f) {
    return [
      ...f.knobs.map((k) => [`<kbd>${esc(glyph(k.cut))}</kbd><kbd>${esc(glyph(k.boost))}</kbd>`, `${esc(LABELS[k.field] || k.field)} ∓ / ± ${k.step}`]),
      ["<kbd>0</kbd>", "prompt tilt and rate back to flat"],
      ["<kbd>-</kbd><kbd>+</kbd>", "buffer cap ∓ / ± 5 s, the floor keeping its share of it"],
      ["<kbd>{</kbd><kbd>}</kbd>", "buffer floor ∓ / ± 0.1 of the cap: 1 tops the buffer up round by round, 0 drains it dry"],
      ["<kbd>t</kbd>", "throttle on / off: hold it open to bank a cushion before a model swap"],
      ["<kbd>1</kbd>–<kbd>9</kbd>", "swap to that playlist model (sessions started with --playlist)"],
      ["<kbd>r</kbd>", "reroll the round being generated"],
      ["<kbd>p</kbd>", "pin the round coming out of the speakers into the motif bank"],
      ["<kbd>i</kbd><kbd>I</kbd>", "inject the newest pinned motif into the next round, light or heavy"],
      ["<kbd>Q</kbd>", "stop the stream"],
      ["<kbd>f</kbd>", "readings follow the round just generated, or the one playing"],
      ["<kbd>b</kbd>", "blind: hide every reading (still recorded)"],
      ["<kbd>?</kbd>", "this list"],
    ];
  }

  // ---------- drawing ----------
  function draw(f) {
    if (!page) return;
    page.frame = f;
    page.fetchedAt = performance.now();
    const writable = f.state === "live";
    const root = document.querySelector(".stage-session");
    root.classList.toggle("read-only", !writable);
    root.querySelectorAll(".stage-controls button, .stage-controls input").forEach((el) => (el.disabled = !writable));
    drawHead(f);
    drawNow(f);
    drawRibbon(f);
    drawControls(f);
    drawReadings(f);
    drawCharts(f);
    document.getElementById("st-guidance").innerHTML = f.guidance.length
      ? f.guidance.map((a) => (a.level === "warning" ? `<p><span class="err">⚠</span> ${esc(a.message)}</p>` : `<p class="dim">· ${esc(a.message)}</p>`)).join("")
      : `<p class="dim">nothing to say about these settings</p>`;
    drawEvents(f);
    drawLog(f);
    tick();
    if (!PLAYING.includes(f.state)) stopPolling();
  }

  // ---------- between two polls ----------
  // The page polls once a second; these move what moves in real time in between (the buffer drains at one second
  // per second, the playing round advances, the generation runs), so the frame never sits still.
  function aheadNow(f) {
    const h = f.head;
    if (!PLAYING.includes(f.state) || !h || h.seconds_ahead_now == null || h.drained) return null;
    const since = f.exact
      ? (Date.now() - Date.parse(f.status.updated_at)) / 1000 || 0
      : ((f.latest && f.latest.age_seconds) || 0) + (performance.now() - page.fetchedAt) / 1000;
    return Math.max(0, h.seconds_ahead_now - Math.max(0, since));
  }
  // Seconds queued behind the tile at index i: what plays after it.
  function queuedAfter(h, i) {
    let after = 0;
    for (let j = i + 1; j < h.playback.length; j++) after += h.playback[j] || 0;
    return after;
  }
  function phaseSeconds(status) {
    return status.phase_seconds + Math.max(0, (Date.now() - Date.parse(status.updated_at)) / 1000 || 0);
  }

  function tick() {
    if (!page) return;
    const f = page.frame;
    const ahead = aheadNow(f);
    const aheadEl = document.getElementById("st-ahead");
    if (aheadEl && ahead != null) aheadEl.textContent = `${ahead.toFixed(1)} s`;
    const bar = document.getElementById("st-ahead-bar");
    const cap = f.control.max_buffer_ahead_seconds;
    if (bar && ahead != null && cap) bar.style.width = `${Math.min(100, (ahead / cap) * 100)}%`;
    const phaseEl = document.getElementById("st-phase-s");
    if (phaseEl && f.status) phaseEl.textContent = `${phaseSeconds(f.status).toFixed(1)} s`;

    const playing = document.querySelector("#st-ribbon .tile.playing");
    if (playing && ahead != null) {
      const h = f.history;
      const i = Number(playing.dataset.i);
      const span = h.playback[i];
      if (span) playing.style.setProperty("--p", Math.max(0, Math.min(1, 1 - (ahead - queuedAfter(h, i)) / span)).toFixed(4));
    }
    const gen = document.querySelector("#st-ribbon .tile.gen");
    if (gen && f.status) {
      const expected = f.latest && f.latest.elapsed_seconds;
      const done = f.status.phase === "scoring" ? 1 : expected ? Math.min(0.97, phaseSeconds(f.status) / expected) : 0;
      gen.style.setProperty("--g", done.toFixed(4));
    }
  }

  // ---------- the round ribbon ----------
  function editTexts(f, edits) {
    const knobs = new Map(f.knobs.map((k) => [k.field, k]));
    const plain = (v) => (typeof v === "number" ? String(+v.toFixed(3)) : String(v));
    return (edits || []).map(([field, before, after]) => {
      if (knobs.has(field)) return `${LABELS[field] || field} ${knobValue(knobs.get(field), before)} → ${knobValue(knobs.get(field), after)}`;
      if (field === "model_xp") return `model → ${nameOf(after)}`;
      if (field === "throttle_enabled") return `throttle ${after ? "on" : "off"}`;
      return `${field} ${plain(before)} → ${plain(after)}`;
    });
  }

  function drawRibbon(f) {
    const el = document.getElementById("st-ribbon");
    const h = f.history;
    const live = f.state === "live" && f.status ? f.status : null;
    const head = f.head || {};
    const tiles = h ? h.round_idx.map((r, i) => ({ r, i, model: h.model[i], marks: h.marks[i], edits: editTexts(f, h.edits[i]) })) : [];
    // Rounds not logged yet: the one being generated, and the one an edit made now lands in.
    if (live) {
      const last = tiles.length ? tiles[tiles.length - 1].r : -1;
      const upto = Math.max(live.round_idx ?? -1, head.edit_lands_at ?? -1);
      for (let r = last + 1; r <= upto; r++) tiles.push({ r, ghost: true });
    }
    const playingAt = live && !head.drained && head.audible_round != null ? tiles.map((t) => t.r).lastIndexOf(head.audible_round) : -1;
    const armed = new Set(page.trail.filter((e) => !e.heardAt).map((e) => e.lands));
    const colors = new Map();
    const color = (model) => {
      const sig = sigOf(model) || "stage";
      if (!colors.has(sig)) colors.set(sig, NR.chartColor(sig));
      return colors.get(sig);
    };
    const seen = new Set();
    el.innerHTML = tiles.map((t, k) => {
      const cls = [t.ghost ? "ghost-tile" : "", k === playingAt ? "playing" : ""];
      const notes = [`round ${t.r}`];
      if (t.ghost) {
        if (live && GENERATING.includes(live.phase) && t.r === live.round_idx) { cls.push("gen"); notes.push("being generated"); }
      } else notes.push(nameOf(t.model));
      if (live && t.r === head.edit_lands_at) { cls.push("lands"); notes.push("an edit made now lands here"); }
      if (armed.has(t.r) && !(f.latest && f.latest.round_idx >= t.r)) cls.push("armed");
      if (k === playingAt) notes.push("playing");
      for (const kind of t.marks || []) { seen.add(kind); notes.push(MARKS[kind] ? MARKS[kind][1] : kind); }
      notes.push(...(t.edits || []));
      const glyphs = (t.marks || []).map((kind) => (MARKS[kind] ? MARKS[kind][0] : "")).join("");
      return `<div class="tile ${cls.join(" ")}"${t.ghost ? "" : ` data-i="${t.i}" style="--c:${color(t.model)}"`} title="${esc(notes.join(" · "))}">
        ${glyphs ? `<span class="tile-marks">${glyphs}</span>` : ""}${t.edits && t.edits.length ? '<i class="tile-edit"></i>' : ""}${t.r % 5 === 0 ? `<span class="tile-n">${t.r}</span>` : ""}</div>`;
    }).join("");
    const key = [...seen].filter((kind) => MARKS[kind]).map((kind) => `${MARKS[kind][0]} ${MARKS[kind][1]}`);
    if (tiles.some((t) => t.edits && t.edits.length)) key.push('<i class="key-edit"></i> knobs changed');
    document.getElementById("st-ribbon-key").innerHTML = tiles.length
      ? `rounds, oldest first · colour = model${live ? " · ▮ playing · dashed: generating, and where an edit lands" : ""}${key.length ? ` · ${key.join(" · ")}` : ""}`
      : "";
  }

  // A stream started from this page has no terminal: its output, worker diagnostics included, is this log.
  function drawLog(f) {
    const box = document.getElementById("st-log-box");
    box.hidden = !f.launch;
    if (!f.launch) return;
    document.getElementById("st-log-note").textContent = `started from this page ${when(f.launch.launched_at)} · ${f.launch.log}`;
    document.getElementById("st-command").textContent = f.launch.command;
    const el = document.getElementById("st-log");
    const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
    el.textContent = f.launch.log_tail.join("\n") || "(nothing yet)";
    if (stick) el.scrollTop = el.scrollHeight;
  }

  function drawHead(f) {
    const model = (f.latest && f.latest.model_xp) || f.model;
    if (page.model !== model) {
      page.model = model;
      const cover = document.getElementById("st-cover");
      cover.innerHTML = coverOf(model);
      cover.href = xpOf(model) ? `#/xp/${encodeURIComponent(sigOf(model))}` : "#/stage";
      document.getElementById("st-title").textContent = nameOf(model);
    }
    const swapTo = sigOf(f.control.model_xp);
    const swapping = f.latest && swapTo && swapTo !== sigOf(f.latest.model_xp);
    const parts = [sigOf(model) || "base model"];
    if (f.latest) parts.push(`round ${f.latest.round_idx}${f.latest.take_idx ? ` take ${f.latest.take_idx}` : ""}`);
    document.getElementById("st-byline").innerHTML = esc(parts.join(" · ")) + (swapping ? ` <span class="err">→ swapping to ${esc(nameOf(f.control.model_xp))} next round</span>` : "");
  }

  function drawNow(f) {
    const lines = [];
    const status = f.status;
    let lead = stateChip(f.state);
    if (status && f.state === "live") {
      const since = status.phase_seconds + Math.max(0, (Date.now() - Date.parse(status.updated_at)) / 1000);
      const where = status.round_idx != null ? ` round ${status.round_idx}${status.take_idx ? ` take ${status.take_idx}` : ""}` : "";
      lead += `<span class="phase">${esc(PHASES[status.phase] || status.phase)}${where} · <span id="st-phase-s">${since.toFixed(1)} s</span></span>`;
    } else if (f.state === "ended" || f.state === "unfinished") {
      const e = f.ended || {};
      const rounds = e.rounds_written ?? (f.latest ? f.latest.round_idx + 1 : 0);
      lead += `<span class="phase">${e.finished_at ? `${esc(when(e.finished_at))} · ` : ""}${rounds} round${rounds === 1 ? "" : "s"}${e.stopped_by ? ` · stopped by ${esc(e.stopped_by)}` : ""}</span>`;
    }
    lines.push(`<div class="now-lead">${lead}</div>`);
    // Only a stream with no player at all reports no buffer: a real one says 0 before its first round plays.
    if (f.exact && status.buffer_seconds_ahead == null) {
      lines.push(`<div><span class="err">⚠ playback is off</span> <span class="dim">· this stream plays nothing: started with --no-play, or it found no sound device${f.launch ? " (see its log)" : ""}</span></div>`);
    }

    if (PLAYING.includes(f.state)) {
      const h = f.head;
      if (!h) lines.push(`<div>waiting for the first round…</div>`);
      else if (h.drained) lines.push(`<div>playback drained — nothing new for ${secs(f.latest.age_seconds)}; the generator is stopped, throttled or held</div>`);
      else if (h.audible_round != null) {
        const approx = f.exact ? "" : "~";
        lines.push(`<div><span class="k">hearing</span>round ${approx}${h.audible_round}${h.rounds_behind ? ` <span class="dim">· ${h.rounds_behind} behind</span>` : ""}
          <span class="k">an edit now lands at</span>round ${h.edit_lands_at}</div>`);
      }
      if (h) lines.push(`<div>${bufferHtml(f)}</div>`);
      if (!f.exact) lines.push(`<div class="dim small">estimated: this stream started before the live status existed — restart it for the exact playing round and pin target</div>`);
    }
    if (f.latest) {
      const l = f.latest;
      const ratio = l.realtime_ratio;
      lines.push(`<div><span class="k">last round</span>${esc(l.round_wav || "—")} <span class="dim">· gen ${fmt(l.elapsed_seconds, 1)} s / fresh ${fmt(l.playback_seconds, 1)} s</span>${ratio != null ? ` <span class="dim">· ${ratio.toFixed(2)}× rt</span>${ratio > 1 ? ' <span class="err">⚠ starving</span>' : ""}` : ""}${l.stale && PLAYING.includes(f.state) ? ` <span class="err">⚠ readings are ${secs(l.age_seconds)} old</span>` : ""}</div>`);
    }
    const pending = f.control.command && f.control.command !== "continue" ? `<span class="pending">pending: ${esc(f.control.command)}</span>` : "";
    lines.push(`<div class="trail">${trailHtml(f)}</div>`);
    lines.push(`<div class="now-flash"><span class="${page.flash && page.flash.err ? "err" : ""}">${page.flash ? esc(page.flash.text) : ""}</span>${pending}</div>`);
    document.getElementById("st-now").innerHTML = lines.join("");
  }

  // A share of the cap, as the TUI's { and } keys read it; unset means the cap less the 10 s default gap.
  function floorRatio(c) {
    const cap = c.max_buffer_ahead_seconds;
    if (cap == null || cap <= 0) return null;
    return Math.min(1, Math.max(0, (c.resume_buffer_ahead_seconds ?? cap - 10) / cap));
  }

  function bufferHtml(f) {
    const c = f.control;
    const ahead = f.head.seconds_ahead_now;
    const cap = c.max_buffer_ahead_seconds;
    const open = cap != null && c.throttle_enabled === false;
    let meter = "";
    if (ahead != null && cap && !open) {
      const resume = c.resume_buffer_ahead_seconds;
      meter = `<span class="buffer-meter" aria-hidden="true"><span id="st-ahead-bar" style="width:${Math.min(100, (ahead / cap) * 100)}%"></span>${resume != null ? `<i style="left:${Math.min(100, (resume / cap) * 100)}%" title="floor at ${resume.toFixed(0)} s"></i>` : ""}</span>`;
    }
    const note = open ? `<span class="err">⚠ throttle off — running ahead (cap ${cap.toFixed(0)} s parked)</span>`
      : cap != null ? `<span class="dim">cap ${cap.toFixed(0)} s${c.resume_buffer_ahead_seconds != null ? ` · floor ${c.resume_buffer_ahead_seconds.toFixed(0)} s` : ""}</span>`
      : `<span class="dim">no throttle</span>`;
    return `<span class="k">buffer ahead</span><b class="num" id="st-ahead">${ahead != null ? `${ahead.toFixed(1)} s` : "—"}</b>${meter || " "}${note}`;
  }

  function drawControls(f) {
    const c = f.control;
    for (const knob of f.knobs) {
      const row = document.querySelector(`.knob[data-field="${CSS.escape(knob.field)}"]`);
      if (!row) continue;
      const value = c[knob.field];
      if (page.dragging !== knob.field) row.querySelector("input").value = value;
      if (page.dragging !== knob.field) row.querySelector("output").textContent = knobValue(knob, value);
      row.classList.toggle("engaged", knob.field in NEUTRAL && value !== NEUTRAL[knob.field]);
    }
    const cap = c.max_buffer_ahead_seconds;
    document.getElementById("st-cap").textContent = cap != null ? `${cap.toFixed(0)} s` : "off";
    const floor = floorRatio(c);
    document.getElementById("st-floor").textContent = floor != null ? `${floor.toFixed(1)} ×` : "";
    const throttle = document.getElementById("st-throttle");
    const on = cap != null && c.throttle_enabled !== false;
    throttle.setAttribute("aria-pressed", String(on));
    throttle.classList.toggle("on", on);
    throttle.firstChild.textContent = on ? "throttle on " : "throttle off ";
    const playlist = f.playlist || [];
    document.getElementById("st-models-group").hidden = !playlist.length;
    const current = sigOf(c.model_xp);
    document.getElementById("st-models").innerHTML = playlist.slice(0, 9).map((entry, i) => `<button class="ghost ${sigOf(entry) === current ? "on" : ""}" data-key="${i + 1}" ${f.state === "live" ? "" : "disabled"}>
        ${esc(nameOf(entry))} <kbd>${i + 1}</kbd></button>`).join("");
  }

  function meterHtml(value, range, alarm) {
    if (value == null) return `<span class="meter"></span>`;
    const [lo, hi] = range || [0, 1];
    const pct = Math.max(0, Math.min(100, ((value - lo) / (hi - lo)) * 100));
    return `<span class="meter${alarm ? " alarm" : ""}"><span style="width:${pct}%"></span></span>`;
  }

  function drawReadings(f) {
    const focus = f.focus;
    document.getElementById("st-focus").textContent = prefs.blind ? "Readings" : focus ? `Readings · ${focus.label}` : "Readings";
    const metricsEl = document.getElementById("st-metrics");
    const candsEl = document.getElementById("st-cands");
    document.getElementById("st-flags").textContent = !prefs.blind && focus && focus.flags.length ? `flags · ${focus.flags.join(", ")}` : "";
    if (prefs.blind) {
      metricsEl.innerHTML = `<p class="dim">metrics hidden — playing blind; the numbers are still recorded</p>`;
      candsEl.innerHTML = "";
      return;
    }
    if (!focus) {
      metricsEl.innerHTML = `<p class="dim">waiting for the first round…</p>`;
      candsEl.innerHTML = "";
      return;
    }
    metricsEl.innerHTML = f.metric_rows.map(([label, key]) => {
      const value = focus.metrics[key];
      if (key === "timbral_coherence" && value != null && value >= f.timbral_alarm) {
        return `<div class="metric"><span class="lbl">${esc(label)}</span><span class="num">steady</span><span class="dim small">≥ ${f.timbral_alarm}: carries almost nothing up here</span></div>`;
      }
      const alarm = key === "timbral_coherence" && value != null;
      return `<div class="metric${alarm ? " alarm" : ""}"><span class="lbl">${esc(label)}</span><span class="num">${fmt(value, 3)}</span>
        <span>${meterHtml(value, f.meter_ranges[key], alarm)}${alarm ? ' <span class="err">⚠ lurching</span>' : ""}</span></div>`;
    }).join("");
    const cands = focus.candidates;
    candsEl.innerHTML = cands.length ? `<table class="cands"><thead><tr><th>candidate</th><th class="r">arc</th><th class="r">tail silence</th><th class="r">novelty</th></tr></thead><tbody>
      ${cands.map((c) => `<tr class="${c.selected ? "won" : ""}"><td>${c.selected ? "◀ " : ""}#${esc(c.candidate)}</td><td class="r num">${fmt(c.arc_coherence_score, 3)}</td>
        <td class="r num">${fmt(c.tail_silence_fraction, 3)}</td><td class="r num">${fmt(c.novelty, 4)}</td></tr>`).join("")}</tbody></table>` : "";
  }

  function drawCharts(f, force) {
    const h = f.history;
    const key = `${h ? h.round_idx.join(",") : ""}|${prefs.blind}|${sigOf(page.model)}`;
    if (!force && key === page.chartsKey) return;
    page.chartsKey = key;
    const shown = CHARTS.filter(([k]) => !prefs.blind || k === "buffer");
    document.getElementById("st-history-note").textContent = h ? `last ${h.round_idx.length} rounds, each self-scaled · dashed: a knob moved` : "";
    const box = document.getElementById("st-charts");
    box.innerHTML = shown.map(([k, title]) => `<div class="spark"><h5>${title}</h5><div id="st-chart-${k}"></div></div>`).join("");
    const color = NR.chartColor(sigOf(page.model) || "stage");
    // A dashed line where a knob moved, so a reading can be read against what was turned.
    const notes = h ? h.edits.map((edits) => editTexts(f, edits).join(", ")) : [];
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
    const marks = notes.map((label, i) => (label ? { epoch: i + 1, full: true, color: accent, label } : null)).filter(Boolean);
    for (const [k] of shown) {
      NR.drawChart(document.getElementById(`st-chart-${k}`), [{ name: k, values: h ? h[k] : [], color, width: 1.6 }],
        { height: 116, labels: h ? h.round_idx : [], unit: "round", ticks: 2, legend: false, empty: "no readings yet", marks, notes });
    }
  }

  function drawEvents(f) {
    const el = document.getElementById("st-events");
    const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
    document.getElementById("st-events-note").textContent = f.events_derived ? "derived from round rows — no events.jsonl" : "newest last";
    el.innerHTML = f.events.length ? f.events.map((e) => {
      const t = e.timestamp ? new Date(e.timestamp).toLocaleTimeString(undefined, { hour12: false }) : "--:--:--";
      return `<div class="ev ev-${esc(e.level || "info")}"><span class="dim num">${esc(t)}</span><span class="dim num">${e.round_idx != null ? `r${esc(e.round_idx)}` : ""}</span>
        <span><b>${esc(e.kind)}</b> ${esc(e.message)}</span></div>`;
    }).join("") : `<p class="dim">no events yet</p>`;
    if (stick) el.scrollTop = el.scrollHeight;
  }

  // ---------- editing ----------
  let queued = [];
  let sending = false;

  function flash(text, err) {
    if (!page) return;
    page.flash = text ? { text, err } : null;
    drawNow(page.frame);
  }

  // Edits the loop only reads at the start of a round; the buffer ones act at once, a pin or a reroll on its own round.
  const AT_ONCE = ["pin_target", "max_buffer_ahead_seconds", "resume_buffer_ahead_seconds", "throttle_enabled"];
  function edited(result) {
    if (!page) return;
    const before = page.frame.control;
    const changed = Object.keys(result.control).filter((k) => k in before && !AT_ONCE.includes(k) && JSON.stringify(before[k]) !== JSON.stringify(result.control[k])
      && (k !== "command" || String(result.control.command).startsWith("inject")));
    page.frame.control = result.control;
    drawControls(page.frame);
    const lands = page.frame.head && page.frame.head.edit_lands_at;
    if (changed.length && lands != null && page.frame.state === "live") {
      const tag = changed.join(",");
      const last = page.trail[page.trail.length - 1];
      if (last && last.tag === tag && last.lands === lands && !last.heardAt) last.text = result.message;
      else page.trail = [...page.trail, { tag, fields: changed, lands, text: result.message, at: Date.now() }].slice(-4);
      for (const field of changed) {
        const row = document.querySelector(`.knob[data-field="${CSS.escape(field)}"]`);
        if (row) { row.classList.remove("just"); void row.offsetWidth; row.classList.add("just"); }
      }
      drawRibbon(page.frame);
      tick();
      flash(null);
      return;
    }
    flash(result.message, false);
  }

  // Each edit, from the key to the speakers: waiting for its round, generated, audible.
  function trailHtml(f) {
    if (f.state !== "live") { page.trail = []; return ""; }
    const now = Date.now();
    // An edit undone before its round never shows up in one: let it go after a minute.
    page.trail = page.trail.filter((e) => (e.heardAt ? now - e.heardAt < 8000 : now - e.at < 60000));
    if (!page.trail.length) return "";
    const h = f.history;
    const head = f.head || {};
    const latest = f.latest ? f.latest.round_idx : -1;
    // The frame an edit's round was read from is up to a poll old: a round that started meanwhile ran without it.
    // The logged rounds say which one took it.
    for (const e of page.trail) {
      if (!h || e.heardAt || e.fields.includes("command")) continue;
      let i;
      while (e.lands <= latest && (i = h.round_idx.indexOf(e.lands)) >= 0 && !h.edits[i].some(([field]) => e.fields.includes(field))) e.lands++;
    }
    const hearing = head.audible_round != null && !head.drained;
    const ahead = aheadNow(f);
    return page.trail.map((e) => {
      let step, cls;
      if (latest < e.lands) [step, cls] = [`waiting · lands in round ${e.lands}`, "sent"];
      else if (hearing && head.audible_round >= e.lands) {
        e.heardAt = e.heardAt || now;
        [step, cls] = ["audible now", "heard"];
      } else {
        const i = h ? h.round_idx.indexOf(e.lands) : -1;
        const wait = hearing && i >= 0 && ahead != null ? ahead - (h.playback[i] || 0) - queuedAfter(h, i) : null;
        if (!hearing) e.heardAt = e.heardAt || now; // nothing plays: generated is as far as it goes
        [step, cls] = [`generated in round ${e.lands}${wait != null ? ` · audible in ~${Math.max(0, Math.round(wait))} s` : ""}`, "made"];
      }
      return `<span class="trail-step ${cls}"><b>${esc(e.text || "edit")}</b> ${esc(step)}</span>`;
    }).join("");
  }

  // A held key or nudge arrives as a burst: one request at a time, the keys pressed meanwhile sent together.
  function press(keys) {
    queued.push(...keys);
    if (!sending) send();
  }
  async function send() {
    if (!page || !queued.length) { queued = []; return; }
    const id = page.id;
    const keys = queued.splice(0, 64);
    sending = true;
    try {
      edited(await NR.postJSON(`/api/stage/${encodeURIComponent(id)}/keys`, { keys }));
    } catch (err) {
      flash(err.message, true);
    } finally {
      sending = false;
    }
    if (queued.length) send();
  }

  async function setKnob(field, value) {
    try {
      edited(await NR.postJSON(`/api/stage/${encodeURIComponent(page.id)}/set`, { field, value }));
    } catch (err) {
      flash(err.message, true);
      drawControls(page.frame);
    }
  }

  function bind() {
    const root = document.querySelector(".stage-session");
    let swept = false;
    // Holding a nudge sweeps, as holding a key does in the TUI.
    root.addEventListener("pointerdown", (ev) => {
      const b = ev.target.closest("button.nudge");
      if (!b || b.disabled) return;
      swept = false;
      let timer = setTimeout(function repeat() { swept = true; press([b.dataset.key]); timer = setTimeout(repeat, 70); }, 400);
      const stop = () => {
        clearTimeout(timer);
        window.removeEventListener("pointerup", stop);
        window.removeEventListener("pointercancel", stop);
        setTimeout(() => { swept = false; }); // after the click a release on the button fires, which a sweep owns
      };
      window.addEventListener("pointerup", stop);
      window.addEventListener("pointercancel", stop);
    });
    root.addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-key]");
      if (b && !b.disabled) {
        if (swept) return;
        if (b.dataset.key === "Q" && !confirm(STOP_CONFIRM)) return;
        press([b.dataset.key]);
        return;
      }
      const follow = ev.target.closest("[data-follow]");
      if (follow) setFollow(follow.dataset.follow);
    });
    root.addEventListener("input", (ev) => {
      const input = ev.target.closest(".knob input");
      if (!input) return;
      const row = input.closest(".knob");
      page.dragging = row.dataset.field;
      const knob = page.frame.knobs.find((k) => k.field === row.dataset.field);
      row.querySelector("output").textContent = knobValue(knob, Number(input.value));
    });
    root.addEventListener("change", (ev) => {
      const input = ev.target.closest(".knob input");
      if (!input) return;
      page.dragging = null;
      setKnob(input.closest(".knob").dataset.field, Number(input.value));
      if (!input.matches(":focus-visible")) input.blur(); // hand the arrow keys back to the page after a drag
    });
    document.getElementById("st-blind").addEventListener("click", toggleBlind);
    const help = document.getElementById("st-help");
    document.getElementById("st-help-open").addEventListener("click", () => help.showModal());
    document.getElementById("st-help-close").addEventListener("click", () => help.close());
    syncModes();
  }

  function syncModes() {
    document.querySelectorAll("[data-follow]").forEach((b) => b.classList.toggle("on", b.dataset.follow === prefs.follow));
    const blind = document.getElementById("st-blind");
    if (blind) { blind.classList.toggle("on", prefs.blind); blind.setAttribute("aria-pressed", String(prefs.blind)); }
  }

  async function setFollow(mode) {
    prefs.follow = mode;
    savePrefs();
    syncModes();
    flash(`readings now follow the ${mode} round`, false);
    if (page) draw(await fetchFrame(page.id));
  }

  function toggleBlind() {
    prefs.blind = !prefs.blind;
    savePrefs();
    syncModes();
    flash(prefs.blind ? "playing blind — readings hidden" : "readings visible again", false);
    drawReadings(page.frame);
    drawCharts(page.frame, true);
  }

  // The TUI's keys. Display keys stay here; the rest go to the server, which applies the TUI's own keymap.
  const ARROWS = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right" };
  function bound(key, f) {
    if (f.knobs.some((k) => k.cut === key || k.boost === key) || "+-t0rpiIQ{}".includes(key)) return true;
    return /^[1-9]$/.test(key) && (f.playlist || []).length > 0;
  }
  document.addEventListener("keydown", (ev) => {
    if (!page || !document.querySelector(".stage-session") || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName) || document.getElementById("st-help").open) return;
    const key = ARROWS[ev.key] || (ev.key.length === 1 ? ev.key : null);
    if (!key || key === " ") return;
    if (key === "?") document.getElementById("st-help").showModal();
    else if (key === "f") setFollow(prefs.follow === "generated" ? "playing" : "generated");
    else if (key === "b") toggleBlind();
    else if (page.frame.state === "live" && bound(key, page.frame)) press([key]);
    else return;
    ev.preventDefault();
    ev.stopImmediatePropagation();
  }, true);

  let resizing = 0;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizing);
    resizing = requestAnimationFrame(() => { if (page && document.getElementById("st-charts")) drawCharts(page.frame, true); });
  });

  NR.views.stage = (args) => (args[0] ? viewSession(args[0]) : viewList());
})();
