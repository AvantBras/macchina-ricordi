// Blind A/B from the browser. The server holds the pairs; this page only ever sees
// "A", "B" and opaque audio URLs until the session is finished and revealed.
// Verdicts land in the same store as `judge ab`, so `judge prefs` reads them as-is.
(function () {
  "use strict";
  const NR = window.NR;
  const { esc } = NR;
  const PREFS_KEY = "nr.listen";

  function loadPrefs() {
    return Object.assign(
      { question: "better", testing: "metric", knob: "temperature", pairs: 20, seconds: 0, group_by: "model_xp", cross_group: true, targeting: "none",
        target_metric: "", disagree_a: "", disagree_b: "", target_quantile: 0.5, context_window: 3,
        listener: "me", sources: [], autoplay: true, metrics_csv: "" },
      NR.storage(() => JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"), {}),
    );
  }
  function savePrefs(p) {
    NR.storage(() => localStorage.setItem(PREFS_KEY, JSON.stringify(p)));
  }

  // ---------- setup ----------
  async function viewSetup() {
    NR.setNav("listen");
    const prefs = loadPrefs();
    NR.view.innerHTML = `<div class="loading">préparation…</div>`;
    const setup = await NR.getJSON("/api/ab/setup");
    const chosen = new Set(prefs.sources);
    const sourceRows = (items) => items.map((s) => `<tr>
        <td><input type="checkbox" name="src" value="${esc(s.path)}" data-kind="${esc(s.kind)}" ${chosen.has(s.path) ? "checked" : ""}></td>
        <td>${esc(s.id)}</td><td class="dim">${esc(s.kind)}</td><td class="dim">${esc(s.models.join(", "))}</td><td class="r num">${s.takes}</td></tr>`).join("");

    NR.view.innerHTML = `<div class="listen-setup">
      <div class="listen-intro">
        <h1 class="title-xl">Écoute à l'aveugle</h1>
        <p>Two clips, labelled only A and B. Which model, which temperature, which side: all hidden until the session ends.
        Verdicts go to the same store as <code>judge ab</code>, and pairs you have already judged for a question are never offered again.</p>
      </div>
      <form id="ab-form" class="ab-form">
        <fieldset><legend>Testing</legend>
          ${[["metric", "a metric", "both clips from one model at the same knobs: only chance differs, so the verdict is about what the metric measures"],
             ["parameter", "a generation parameter", "one model, every other knob equal, this one different"],
             ["model", "models", "the same knobs, different models: pick sessions of the models to compare"],
             ["groups", "groups", "clips from different models, sessions or metric buckets"]].map(([v, l, d]) => `<label class="radio"><input type="radio" name="testing" value="${v}" ${prefs.testing.split(":")[0] === v ? "checked" : ""}>
            <span><b>${l}</b> <span class="dim">${d}</span></span></label>`).join("")}
          <label data-k><span>parameter</span><input name="knob" list="knobs" value="${esc(prefs.knob)}">
            <datalist id="knobs">${["temperature", "top_k", "top_p", "min_p", "candidates", "novelty_weight", "cfg_coef"].map((k) => `<option value="${k}">`).join("")}</datalist></label>
        </fieldset>
        <fieldset><legend>Question</legend>
          ${setup.questions.map((q) => `<label class="radio"><input type="radio" name="question" value="${esc(q.id)}" ${q.id === prefs.question ? "checked" : ""}>
            <span><b>${esc(q.prompt)}</b> <span class="dim">${esc(q.id)}${q.type === "pairwise_context" ? " · stream sessions only, plays a context reel first" : ""}${q.metrics.length ? ` · validates ${esc(q.metrics.join(", "))}` : ""}</span></span></label>`).join("")}
        </fieldset>
        <fieldset class="grid-2"><legend>Session</legend>
          <label><span>pairs</span><input type="number" name="pairs" min="1" max="200" value="${prefs.pairs}"></label>
          <label><span>excerpt seconds <em>0 = whole clip</em></span><input type="number" name="seconds" min="0" step="1" value="${prefs.seconds}"></label>
          <label data-g><span>groups are</span><select name="group_by">
            ${[["model_xp", "models"], ["session", "sessions / evenings"]].map(([v, l]) => `<option value="${v}" ${prefs.group_by === v ? "selected" : ""}>${l}</option>`).join("")}
            ${prefs.group_by.startsWith("bucket:") ? `<option selected value="${esc(prefs.group_by)}">${esc(prefs.group_by)}</option>` : ""}
          </select></label>
          <label data-g><span>or bucket by metric <em>top vs bottom tercile</em></span><input name="bucket" placeholder="e.g. novelty_whitened" value="${esc(prefs.group_by.startsWith("bucket:") ? prefs.group_by.slice(7) : "")}"></label>
          <label class="check" data-g><input type="checkbox" name="cross_group" ${prefs.cross_group ? "checked" : ""}> cross-group pairs only</label>
          <label class="check"><input type="checkbox" name="autoplay" ${prefs.autoplay ? "checked" : ""}> play A then B automatically</label>
          <label><span>seed</span><input type="number" name="seed" value="${Math.floor(Math.random() * 100000)}"></label>
          <label><span>listener</span><input name="listener" value="${esc(prefs.listener)}"></label>
        </fieldset>
        <fieldset class="grid-2"><legend>Targeting <em>spend judgements where a metric has something to learn</em></legend>
          <label><span>mode</span><select name="targeting">
            ${[["none", "none"], ["metric", "pairs a metric strongly separates"], ["disagree", "pairs two metrics rank oppositely"]].map(([v, l]) => `<option value="${v}" ${prefs.targeting === v ? "selected" : ""}>${l}</option>`).join("")}
          </select></label>
          <label><span>quantile cut</span><input type="number" name="target_quantile" min="0" max="0.99" step="0.05" value="${prefs.target_quantile}"></label>
          <label data-t="metric"><span>metric</span><input name="target_metric" value="${esc(prefs.target_metric)}" placeholder="arc_coherence_score"></label>
          <label data-t="disagree"><span>metric 1</span><input name="disagree_a" value="${esc(prefs.disagree_a)}"></label>
          <label data-t="disagree"><span>metric 2</span><input name="disagree_b" value="${esc(prefs.disagree_b)}"></label>
        </fieldset>
        <fieldset><legend>Sources <span class="dim" id="src-count"></span></legend>
          <div class="src-tools"><input type="search" id="src-filter" placeholder="filter sources…"> <button type="button" class="linkish" id="src-all">select shown</button> <button type="button" class="linkish" id="src-none">clear</button></div>
          <div class="src-table"><table>
            <thead><tr><th></th><th>session</th><th>kind</th><th>models</th><th class="r" title="a stream offers only the rounds it scored live">takes</th></tr></thead>
            <tbody>${sourceRows(setup.sessions) || `<tr><td colspan="5" class="dim">no sessions yet: <code>play generate</code>, <code>play sweep</code> or <code>play stream</code> make them</td></tr>`}</tbody>
          </table></div>
        </fieldset>
        <div class="form-actions"><button class="solid big" type="submit">Begin</button><span class="form-msg" id="ab-msg"></span></div>
      </form>
      <div id="ab-summary"></div>
    </div>`;

    const form = document.getElementById("ab-form");
    const syncTargeting = () => {
      const mode = form.elements.targeting.value;
      form.querySelectorAll("[data-t]").forEach((el) => (el.hidden = el.dataset.t !== mode));
    };
    const syncCount = () => {
      const n = form.querySelectorAll("input[name=src]:checked").length;
      document.getElementById("src-count").textContent = n ? `${n} selected` : "";
    };
    const testing = () => form.querySelector("input[name=testing]:checked")?.value || "metric";
    const syncTesting = () => {
      form.querySelectorAll("[data-g]").forEach((el) => (el.hidden = testing() !== "groups"));
      form.querySelectorAll("[data-k]").forEach((el) => (el.hidden = testing() !== "parameter"));
    };
    form.elements.targeting.addEventListener("change", syncTargeting);
    form.querySelectorAll("input[name=testing]").forEach((radio) => radio.addEventListener("change", syncTesting));
    // `newer` is judged high-vs-low novelty within one session (docs/scoring-structure.md).
    form.querySelectorAll("input[name=question]").forEach((radio) => radio.addEventListener("change", () => {
      if (!radio.checked || radio.value !== "newer") return;
      if (testing() === "groups") {
        if (!form.elements.bucket.value) form.elements.bucket.value = "novelty_whitened";
      } else if (form.elements.targeting.value === "none") {
        form.elements.targeting.value = "metric";
        form.elements.target_metric.value = form.elements.target_metric.value || "novelty_whitened";
        syncTargeting();
      }
    }));
    form.addEventListener("change", syncCount);
    syncTargeting();
    syncTesting();
    syncCount();
    const filter = document.getElementById("src-filter");
    filter.addEventListener("input", () => {
      const q = filter.value.toLowerCase();
      form.querySelectorAll("input[name=src]").forEach((cb) => {
        cb.closest("tr").hidden = q && !cb.closest("tr").textContent.toLowerCase().includes(q);
      });
    });
    document.getElementById("src-all").addEventListener("click", () => {
      form.querySelectorAll("input[name=src]").forEach((cb) => { if (!cb.closest("tr").hidden) cb.checked = true; });
      syncCount();
    });
    document.getElementById("src-none").addEventListener("click", () => {
      form.querySelectorAll("input[name=src]").forEach((cb) => (cb.checked = false));
      syncCount();
    });

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const f = form.elements;
      const bucket = f.bucket.value.trim();
      const p = {
        question: form.querySelector("input[name=question]:checked")?.value || "better", knob: f.knob.value.trim(),
        testing: testing() === "parameter" ? `parameter:${f.knob.value.trim()}` : testing(),
        pairs: +f.pairs.value, seconds: +f.seconds.value, group_by: bucket ? `bucket:${bucket}` : f.group_by.value,
        cross_group: f.cross_group.checked, autoplay: f.autoplay.checked, targeting: f.targeting.value,
        target_metric: f.target_metric.value.trim(), disagree_a: f.disagree_a.value.trim(), disagree_b: f.disagree_b.value.trim(),
        target_quantile: +f.target_quantile.value, context_window: prefs.context_window, listener: f.listener.value.trim() || "me",
        sources: [...form.querySelectorAll("input[name=src]:checked")].map((cb) => cb.value), metrics_csv: prefs.metrics_csv,
      };
      savePrefs(p);
      const msg = document.getElementById("ab-msg");
      msg.classList.remove("err");
      msg.textContent = "drawing pairs…";
      try {
        const state = await NR.postJSON("/api/ab/sessions", {
          sources: p.sources, question: p.question, testing: p.testing, pairs: p.pairs, seed: +f.seed.value, cross_group: p.cross_group,
          group_by: p.group_by, target_metric: p.targeting === "metric" ? p.target_metric : "",
          disagree: p.targeting === "disagree" ? [p.disagree_a, p.disagree_b] : [], target_quantile: p.target_quantile,
          context_window: p.context_window, seconds: p.seconds || null, listener: p.listener,
        });
        location.hash = `#/listen/${encodeURIComponent(state.session_id)}`;
      } catch (err) {
        msg.textContent = err.message;
        msg.classList.add("err");
      }
    });

    drawSummary(prefs);
  }

  function witnessHtml(summary) {
    const counts = summary.verdict_counts || {};
    const labels = Object.entries(counts.label || {}).map(([q, n]) => `${esc(q)} ${n}`).join(" · ");
    const rows = (summary.witness || []).map((w) => {
      const r = w.ratings || {};
      const axes = ["shape", "coherence", "intent", "engagement"].map((k) => r[k] != null ? `${k} ${(+r[k]).toFixed(2)}` : "").filter(Boolean).join(" · ");
      const name = esc(w.path.split("/").slice(-3).join("/"));
      // only clips inside the repo can be served
      const clip = w.path.startsWith("/") ? name : `<a href="/files/${w.path.split("/").map(encodeURIComponent).join("/")}" target="_blank" rel="noopener">${name}</a>`;
      return `<tr><td>${clip}</td>
        <td class="r num">${w.score.toFixed(3)}</td><td class="dim small">${esc(axes)}${w.error ? ` · ${esc(w.error)}` : ""}</td>
        <td class="dim small">${esc(w.by)}</td><td class="r dim small">${esc((w.created_at || "").slice(0, 16).replace("T", " "))}</td></tr>`;
    }).join("");
    if (!labels && !rows) return "";
    return `<div class="section"><h3>Other verdicts <span class="dim">alarm labels and the captioner witness · <code>judge log</code> lists them all</span></h3>
      ${labels ? `<p class="dim">alarm labels, by question: ${labels} · <code>judge labels</code> fits their thresholds</p>` : ""}
      ${rows ? `<table class="kv"><thead><tr><th>witness rated</th><th class="r">score</th><th>axes (0–1)</th><th>by</th><th class="r">when</th></tr></thead><tbody>${rows}</tbody></table>` : ""}
    </div>`;
  }

  async function drawSummary(prefs) {
    const box = document.getElementById("ab-summary");
    if (!box) return;
    let summary;
    try {
      summary = await NR.getJSON(`/api/ab/summary${prefs.metrics_csv ? `?metrics_csv=${encodeURIComponent(prefs.metrics_csv)}` : ""}`);
    } catch (err) {
      prefs.metrics_csv = "";
      savePrefs(prefs);
      summary = await NR.getJSON("/api/ab/summary");
    }
    const byQuestion = summary.questions.map((q) => `<tr><td>${esc(q.question)}</td><td class="r num">${q.judgements}</td><td class="r num">${q.decided}</td></tr>`).join("");
    const agreement = summary.agreement.length
      ? `<table class="agree"><thead><tr><th>question</th><th>metric</th><th class="r">pairs</th><th class="r">agreement</th><th class="r">z</th><th>verdict</th></tr></thead><tbody>
        ${summary.agreement.map((a) => `<tr><td class="dim">${esc(a.question)}</td><td>${esc(a.metric)}</td><td class="r num">${a.pairs}</td>
          <td class="r num"><span class="bar" style="--v:${Math.round(a.agreement * 100)}%"></span>${a.agreement.toFixed(3)}</td><td class="r num">${a.z_score.toFixed(2)}</td><td>${esc(a.verdict)}</td></tr>`).join("")}
        </tbody></table>`
      : `<p class="dim">${summary.judgements ? (summary.metrics_csv ? "not enough scored pairs yet (8 per metric)" : "pick a scored CSV to see which metrics agree with your ear") : "nothing judged yet"}</p>`;
    box.innerHTML = `<div class="section"><h3>The store <span class="dim">${esc(summary.store)} · ${summary.judgements} judgements</span></h3>
        ${byQuestion ? `<table class="kv"><thead><tr><th>question</th><th class="r">judged</th><th class="r">decided</th></tr></thead><tbody>${byQuestion}</tbody></table>` : `<p class="dim">No blind verdicts recorded yet. None of the scoring axes has been checked by ear; twenty pairs is an evening's start.</p>`}
      </div>
      <div class="section"><h3>Metric agreement
          <select id="metrics-csv"><option value="">scored CSV…</option>${summary.metrics_csvs.map((c) => `<option ${c === summary.metrics_csv ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>
          <span class="dim">how often "higher" picked your winner — 0.5 is noise</span></h3>
        ${agreement}
      </div>
      ${witnessHtml(summary)}
      ${summary.sessions.length ? `<div class="section"><h3>Sessions</h3><table class="kv">${summary.sessions.map((s) => `<tr><td>${esc(s.session_id)}</td><td>${esc(s.question)} · ${s.judgements}/${s.offered} judged · ${esc(s.source)}</td></tr>`).join("")}</table></div>` : ""}`;
    document.getElementById("metrics-csv").addEventListener("change", (ev) => {
      prefs.metrics_csv = ev.target.value;
      savePrefs(prefs);
      drawSummary(prefs);
    });
  }

  // ---------- session ----------
  const clip = new Audio();
  clip.preload = "auto";
  let session = null;
  let heard = new Set();
  let queue = [];
  let playing = null;

  function audioUrl(slot) {
    return `/api/ab/sessions/${encodeURIComponent(session.session_id)}/audio/${session.position}/${slot}`;
  }

  function playSlot(slot, keepQueue) {
    if (!session || session.finished) return;
    if (!keepQueue) queue = [];
    NR.pauseMain();
    playing = slot;
    heard.add(slot);
    clip.src = audioUrl(slot);
    clip.play().catch(() => {});
    paint();
  }

  function playSequence() {
    queue = session.contextual ? ["context", "a", "b"] : ["a", "b"];
    playSlot(queue.shift(), true);
  }

  clip.addEventListener("timeupdate", () => {
    if (!session) return;
    const limit = playing !== "context" && session.seconds ? session.seconds : null;
    if (limit && clip.currentTime >= limit) {
      clip.pause();
      clip.dispatchEvent(new Event("ended"));
      return;
    }
    const d = limit ? Math.min(limit, clip.duration || limit) : clip.duration;
    const el = document.querySelector(`.tile[data-slot="${playing}"] .tile-fill`);
    if (el && d) el.style.width = `${Math.min(100, (clip.currentTime / d) * 100)}%`;
  });
  clip.addEventListener("ended", () => {
    if (queue.length) playSlot(queue.shift(), true);
    else { playing = null; paint(); }
  });
  clip.addEventListener("pause", () => paint());
  clip.addEventListener("play", () => paint());

  function ready() {
    const needed = session.contextual ? ["context", "a", "b"] : ["a", "b"];
    return needed.every((s) => heard.has(s));
  }

  function paint() {
    if (!session || session.finished) return;
    document.querySelectorAll(".tile").forEach((t) => {
      t.classList.toggle("on", t.dataset.slot === playing && !clip.paused);
      t.classList.toggle("heard", heard.has(t.dataset.slot));
    });
    const ok = ready();
    document.querySelectorAll("[data-verdict]").forEach((b) => (b.disabled = !ok && b.dataset.verdict !== "skip"));
    const hint = document.getElementById("ab-hint");
    if (hint) hint.textContent = ok ? "" : session.contextual ? "hear the context, A and B first" : "hear both first";
    const undo = document.getElementById("ab-undo");
    if (undo) undo.disabled = !session.can_undo;
  }

  function tileHtml(slot, label, sub) {
    return `<button class="tile" data-slot="${slot}">
      <svg viewBox="0 0 300 470" aria-hidden="true"><rect width="300" height="470" class="tile-ground"/>
        <line x1="145.8" y1="120" x2="145.8" y2="380" class="tile-rule"/><line x1="154.2" y1="120" x2="154.2" y2="380" class="tile-rule"/></svg>
      <span class="tile-letter">${label}</span>
      <span class="tile-sub">${sub}</span>
      <span class="tile-bar"><span class="tile-fill"></span></span>
    </button>`;
  }

  function drawPair() {
    heard = new Set();
    queue = [];
    playing = null;
    clip.pause();
    const s = session;
    NR.view.innerHTML = `<div class="ab-stage">
      <div class="ab-head">
        <div class="crumb"><a href="#/listen">Écoute</a> / ${esc(s.session_id)} · ${esc(s.question)}${s.seconds ? ` · ${s.seconds}s excerpts` : ""}</div>
        <div class="ab-progress num">${s.position + 1} <span class="dim">/ ${s.total}</span></div>
      </div>
      <h1 class="title-xl ab-prompt">${esc(s.prompt)}</h1>
      <div class="tiles ${s.contextual ? "three" : ""}">
        ${s.contextual ? tileHtml("context", "·", "context <kbd>c</kbd>") : ""}
        ${tileHtml("a", "A", "<kbd>1</kbd>")}
        ${tileHtml("b", "B", "<kbd>2</kbd>")}
      </div>
      <div class="verdicts">
        <button class="verdict" data-verdict="a">A <kbd>a</kbd></button>
        <button class="verdict" data-verdict="tie">tie <kbd>t</kbd></button>
        <button class="verdict" data-verdict="b">B <kbd>b</kbd></button>
      </div>
      <div class="ab-tools">
        <button class="linkish" id="ab-replay">replay ${s.contextual ? "context, A, B" : "A then B"} <kbd>r</kbd></button>
        <button class="linkish" data-verdict="skip">skip pair <kbd>s</kbd></button>
        <button class="linkish" id="ab-undo">undo <kbd>u</kbd></button>
        <button class="linkish" id="ab-end">end session &amp; reveal <kbd>q</kbd></button>
        <span class="form-msg" id="ab-hint"></span>
      </div>
      <div class="ab-foot dim">${s.judgements} recorded this session · space pauses</div>
    </div>`;
    NR.view.querySelectorAll(".tile").forEach((t) => t.addEventListener("click", () => {
      if (playing === t.dataset.slot && !clip.paused) clip.pause();
      else if (playing === t.dataset.slot && clip.paused && clip.currentTime > 0) clip.play();
      else playSlot(t.dataset.slot);
    }));
    NR.view.querySelectorAll("[data-verdict]").forEach((b) => b.addEventListener("click", () => answer(b.dataset.verdict)));
    document.getElementById("ab-replay").addEventListener("click", playSequence);
    document.getElementById("ab-undo").addEventListener("click", undo);
    document.getElementById("ab-end").addEventListener("click", finish);
    paint();
    if (loadPrefs().autoplay) playSequence();
  }

  let busy = false;
  async function act(fn) {
    if (busy) return;
    busy = true;
    try { await fn(); } catch (err) {
      const hint = document.getElementById("ab-hint");
      if (hint) { hint.textContent = err.message; hint.classList.add("err"); }
    } finally { busy = false; }
  }

  function answer(verdict) {
    if (!session || session.finished) return;
    if (verdict !== "skip" && !ready()) return;
    act(async () => {
      session = await NR.postJSON(`/api/ab/sessions/${encodeURIComponent(session.session_id)}/answer`, { index: session.position, verdict });
      if (session.position >= session.total) await doFinish();
      else drawPair();
    });
  }

  function undo() {
    if (!session || !session.can_undo) return;
    act(async () => {
      session = await NR.postJSON(`/api/ab/sessions/${encodeURIComponent(session.session_id)}/undo`);
      drawPair();
    });
  }

  function finish() {
    if (!session || session.finished) return;
    if (session.position < session.total && !confirm(`End now? ${session.total - session.position} pairs stay unjudged (they are recorded as unreached).`)) return;
    act(doFinish);
  }

  async function doFinish() {
    clip.pause();
    const reveal = await NR.postJSON(`/api/ab/sessions/${encodeURIComponent(session.session_id)}/finish`);
    session = reveal;
    drawReveal(reveal);
  }

  // Groups are short XP sigs when grouping by model; show the label people know them by.
  // A model session names a checkpoint as `<sig>@<epoch>`.
  function groupName(group) {
    const xp = NR.state.bySig.get(group.split("@")[0]);
    return xp && xp.label ? `${xp.label} · ${group}` : group;
  }

  function sideHtml(side, won) {
    return `<button class="track ${won ? "won" : ""}" data-src="${esc(side.path)}" data-label="${esc(groupName(side.group) + " — " + side.label)}">
      <span class="ico"></span><span class="nm"><b>${esc(groupName(side.group))}</b> <span class="dim">${esc(side.label)}</span>
      ${Object.entries(side.metrics || {}).map(([k, v]) => `<span class="num">${esc(k)} ${v.toFixed(3)}</span>`).join(" ")}
      ${side.knob ? `<span class="num">${esc(side.knob)}</span>` : ""}</span></button>`;
  }

  function drawReveal(r) {
    const verdictText = { a: "A", b: "B", tie: "tie", skipped: "skipped", unreached: "—" };
    const max = Math.max(1, ...r.groups.map((g) => g.appearances));
    const wins = (a) => +(a.agreement * a.pairs).toFixed(1);
    const agreement = `<div class="section"><h3>Metric agreement <span class="dim">how often "higher" picked your winner — 0.5 is noise${r.testing === "metric" ? "; the store column counts only same-condition verdicts" : ""}</span></h3>
        ${r.agreement.length ? `<table class="agree"><thead><tr><th>metric</th><th class="r">this session</th><th class="r">store</th><th class="r">agreement</th><th>verdict</th></tr></thead><tbody>
        ${r.agreement.map(({ session: a, store: s }) => `<tr><td>${esc(a.metric)}</td><td class="r num">${wins(a)} / ${a.pairs}</td>
          <td class="r num">${s ? `${wins(s)} / ${s.pairs}` : "—"}</td>
          <td class="r num">${s ? `<span class="bar" style="--v:${Math.round(s.agreement * 100)}%"></span>${s.agreement.toFixed(3)}` : ""}</td><td>${s ? esc(s.verdict) : ""}</td></tr>`).join("")}
        </tbody></table>` : `<p class="dim">no scored, decided pairs</p>`}
      </div>`;
    const matchups = `<div class="section"><h3>By ${esc(r.knob)} <span class="dim">${r.knob === "model" ? "first model wins – ties – second model wins" : "lower value wins – ties – higher value wins"}</span></h3>
        ${(r.matchups || []).length ? `<table class="kv">${r.matchups.map((m) => `<tr><td>${r.knob === "model" ? `${esc(groupName(m.low))} vs ${esc(groupName(m.high))}` : `${esc(m.low)} vs ${esc(m.high)}`}</td>
          <td class="num">${m.low_wins} – ${m.ties} – ${m.high_wins}</td></tr>`).join("")}</table>` : `<p class="dim">no verdicts</p>`}
      </div>`;
    const byGroup = `<div class="section"><h3>By group <span class="dim">wins / appearances</span></h3>
        ${r.groups.length ? `<table class="kv groups">${r.groups.map((g) => `<tr><td>${NR.state.bySig.has(g.group) ? `<a href="#/xp/${encodeURIComponent(g.group)}">${esc(groupName(g.group))}</a>` : esc(g.group)}</td>
          <td><span class="gbar"><span style="width:${(g.appearances / max) * 100}%"><span style="width:${g.appearances ? (g.wins / g.appearances) * 100 : 0}%"></span></span></span>
          ${g.wins} / ${g.appearances}${g.ties ? ` <span class="dim">(${g.ties} ties)</span>` : ""}</td></tr>`).join("")}</table>` : `<p class="dim">no verdicts</p>`}
      </div>`;
    NR.view.innerHTML = `<div class="ab-stage reveal">
      <div class="crumb"><a href="#/listen">Écoute</a> / ${esc(r.session_id)}</div>
      <h1 class="title-xl">Révélation</h1>
      <p class="byline">${r.judgements} verdict${r.judgements === 1 ? "" : "s"} on “${esc(r.prompt)}”</p>
      <p class="dim store-line">appended to ${esc(r.store)}${r.sidecar ? ` · sidecar ${esc(r.sidecar)}` : ""}</p>
      ${r.knob ? matchups : r.testing === "metric" ? agreement : byGroup + agreement}
      <div class="section"><h3>Pairs <span class="dim">now unblinded — replay any side</span></h3>
        <table class="reveal-table"><thead><tr><th>#</th><th>verdict</th><th>A</th><th>B</th></tr></thead><tbody>
        ${r.pairs.map((p) => `<tr><td class="num dim">${p.index + 1}</td><td>${verdictText[p.outcome] || esc(p.outcome)}</td>
          <td>${sideHtml(p.a, p.outcome === "a")}</td><td>${sideHtml(p.b, p.outcome === "b")}</td></tr>`).join("")}
        </tbody></table>
      </div>
      <div class="form-actions"><a class="solid" href="#/listen">Another session</a></div>
    </div>`;
  }

  async function viewSession(id) {
    NR.setNav("listen");
    await NR.loadXps();
    try {
      session = await NR.getJSON(`/api/ab/sessions/${encodeURIComponent(id)}`);
    } catch (_) {
      session = null;
      NR.view.innerHTML = `<div class="empty">This session is gone (sessions live in the server; a restart finishes them and writes their sidecar).<br><a href="#/listen">Start another</a></div>`;
      return;
    }
    if (session.finished) drawReveal(session);
    else if (session.position >= session.total) act(doFinish);
    else drawPair();
  }

  document.addEventListener("keydown", (ev) => {
    if (!session || session.finished || !document.querySelector(".ab-stage") || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    const key = ev.key.toLowerCase();
    const map = { 1: () => playSlot("a"), 2: () => playSlot("b"), c: () => session.contextual && playSlot("context"),
      r: playSequence, a: () => answer("a"), b: () => answer("b"), t: () => answer("tie"), s: () => answer("skip"),
      u: undo, q: finish, " ": () => (clip.paused ? clip.play() : clip.pause()) };
    if (map[key]) {
      ev.preventDefault();
      ev.stopImmediatePropagation();
      map[key]();
    }
  }, true);

  window.addEventListener("hashchange", () => {
    if (!location.hash.startsWith("#/listen/")) { clip.pause(); session = null; }
  });

  NR.views.listen = (args) => (args[0] ? viewSession(args[0]) : viewSetup());
})();
