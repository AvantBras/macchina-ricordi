// Corpus: the prepared workspaces (what they hold, by source and length), new corpora built from
// the library (`data build`, run as a detached job), and weights.
(function () {
  "use strict";
  const NR = window.NR;
  const { esc } = NR;
  const FORM_KEY = "nr.corpus.form";
  let poll = null;

  function stopPolling() {
    if (poll) { clearTimeout(poll); poll = null; }
  }
  window.addEventListener("hashchange", stopPolling);
  function later(ms, fn) {
    stopPolling();
    poll = setTimeout(async () => {
      try { await fn(); } catch (_) { /* next tick retries */ }
    }, ms);
  }

  const hours = (s) => (s == null ? "—" : s >= 3600 ? `${(s / 3600).toFixed(s >= 36000 ? 0 : 1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);
  const secs = (s) => (s == null ? "—" : s >= 120 ? `${(s / 60).toFixed(s >= 600 ? 0 : 1)} min` : `${s.toFixed(s < 10 ? 1 : 0)} s`);
  const wsUrl = (name) => `#/corpus/ws/${encodeURIComponent(name)}`;
  const pretty = (name) => (Cover.prettyWorkspace(name) || name);
  const chip = (name) => { const p = Cover.palette(name); return `<span class="chip" style="background:${p.ground};--chip-ink:${p.ink}"></span>`; };

  function stateHtml(job) {
    return `<span class="state state-${esc(job.state === "completed" ? "completed" : job.state)}">${job.active ? '<span class="live-dot"></span>' : ""}${esc(job.state)}</span>`;
  }

  function jobsHtml(jobs) {
    if (!jobs.length) return `<p class="dim">No data jobs yet.</p>`;
    return `<div class="table-wrap"><table class="kv runs-table">${jobs.slice(0, 12).map((j) => `<tr data-href="#/corpus/job/${encodeURIComponent(j.id)}">
      <td>${stateHtml(j)}</td><td>${esc(j.title)}</td><td class="dim num">${esc(j.progress || "")}</td><td class="dim">${NR.ago(j.created_at)}</td></tr>`).join("")}</table></div>`;
  }

  // ---------- the list ----------
  async function viewList() {
    NR.setNav("corpus");
    NR.view.innerHTML = `<div class="loading">corpus…</div>`;
    const data = await NR.getJSON("/api/corpus");
    const ui = { q: "", sort: "prepared", dir: -1 };
    const cols = [
      ["workspace", (w) => w.name],
      ["train", (w) => (w.splits.train || {}).clips || 0, "r"],
      ["valid", (w) => (w.splits.valid || {}).clips || 0, "r"],
      ["audio", (w) => (w.splits.train || {}).seconds || 0, "r"],
      ["clip length", (w) => ((w.splits.train || {}).stats || {}).median || 0, "r"],
      ["source", (w) => w.source || ""],
      ["XPs", (w) => w.xps.length, "r"],
      ["prepared", (w) => w.prepared_at || ""],
    ];
    NR.view.innerHTML = `<div class="corpus">
      <div class="corpus-head">
        <p class="dim">Each dataset is a set of clips split into train and valid, with AudioCraft manifests in <code>egs/</code>. New ones are built into <code>datasets/&lt;name&gt;</code> and never change afterwards; the names starting with a dot are workspaces prepared before, now frozen. Open one to hear it, see what it is made of, and weight it.</p>
        <a class="solid big" href="#/corpus/new">+ New corpus</a>
      </div>
      <div class="section"><h3>Travaux <span class="dim">data jobs, newest first</span></h3><div id="jobs">${jobsHtml(data.jobs)}</div></div>
      <div class="section"><h3>Espaces de travail <span class="dim">${data.workspaces.length} workspaces</span></h3>
        <div class="toolbar"><input type="search" id="ws-q" placeholder="filter by name or source" autocomplete="off"><span class="count" id="ws-count"></span></div>
        <div class="table-wrap"><table class="corpus-table"><thead><tr><th></th>${cols.map(([n, , c], i) => `<th data-col="${i}" class="${c || ""}">${n}</th>`).join("")}<th></th></tr></thead><tbody id="ws-rows"></tbody></table></div>
      </div>
    </div>`;
    const draw = () => {
      const col = cols.find(([n]) => n === ui.sort);
      const list = data.workspaces
        .filter((w) => !ui.q || `${w.name} ${w.source || ""}`.toLowerCase().includes(ui.q))
        .sort((a, b) => { const va = col[1](a), vb = col[1](b); return (va < vb ? -1 : va > vb ? 1 : 0) * ui.dir; });
      document.querySelectorAll(".corpus-table th[data-col]").forEach((th) => {
        const name = cols[+th.dataset.col][0];
        th.classList.toggle("sorted", name === ui.sort);
        th.textContent = name + (name === ui.sort ? (ui.dir > 0 ? " ↑" : " ↓") : "");
      });
      document.getElementById("ws-count").textContent = `${list.length} / ${data.workspaces.length}`;
      document.getElementById("ws-rows").innerHTML = list.map((w) => {
        const train = w.splits.train || {}, valid = w.splits.valid || {}, st = train.stats || {};
        return `<tr data-href="${wsUrl(w.name)}">
          <td>${chip(w.name)}</td>
          <td>${w.training ? '<span class="live-dot" title="training on it now"></span>' : ""}${esc(pretty(w.name))} <span class="dim">${esc(w.name)}</span></td>
          <td class="r num">${train.clips || 0}</td><td class="r num dim">${valid.clips || 0}</td>
          <td class="r num">${hours(train.seconds)}</td>
          <td class="r num">${st.median != null ? secs(st.median) : "—"}${st.min != null && st.max - st.min > 1 ? ` <span class="dim">${secs(st.min)}–${secs(st.max)}</span>` : ""}</td>
          <td class="dim src">${w.source ? esc(w.source) : "—"}${w.source && !w.source_exists ? ' <span class="tag gone" title="the folder it was prepared from is gone; data/ is the only copy">gone</span>' : ""}</td>
          <td class="r num">${w.xps.length || ""}</td>
          <td class="r dim">${NR.ago(w.prepared_at)}</td>
          <td>${w.rebalanced.length ? '<span class="tag">weights</span>' : ""}${w.speed_variants ? ' <span class="tag">speed</span>' : ""}</td>
        </tr>`;
      }).join("");
    };
    document.getElementById("ws-q").addEventListener("input", (ev) => { ui.q = ev.target.value.trim().toLowerCase(); draw(); });
    document.querySelectorAll(".corpus-table th[data-col]").forEach((th) => th.addEventListener("click", () => {
      const name = cols[+th.dataset.col][0];
      if (ui.sort === name) ui.dir *= -1; else { ui.sort = name; ui.dir = ["workspace", "source"].includes(name) ? 1 : -1; }
      draw();
    }));
    draw();
    if (data.jobs.some((j) => j.active)) {
      const tick = async () => {
        const fresh = await NR.getJSON("/api/corpus");
        document.getElementById("jobs").innerHTML = jobsHtml(fresh.jobs);
        if (fresh.jobs.some((j) => j.active)) later(3000, tick); else viewList();
      };
      later(3000, tick);
    }
  }

  // ---------- a workspace ----------
  function histogramHtml(durations) {
    if (!durations.length) return "";
    const max = Math.max(...durations), min = Math.min(...durations);
    if (max - min < 0.5) return `<p class="dim">Every clip is ${secs(max)} long.</p>`;
    const bins = 28, width = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    for (const d of durations) counts[Math.min(bins - 1, Math.floor((d - min) / width))]++;
    const top = Math.max(...counts);
    return `<div class="hist">${counts.map((c, i) => `<span title="${secs(min + i * width)}–${secs(min + (i + 1) * width)}: ${c} clips" style="height:${c ? Math.max(3, (c / top) * 100) : 0}%"></span>`).join("")}</div>
      <div class="hist-axis dim num"><span>${secs(min)}</span><span>${secs(max)}</span></div>`;
  }

  function clusterTableHtml(table, opts) {
    const hasGens = table.some((r) => r.gen_share != null);
    const maxShare = Math.max(...table.map((r) => Math.max(r.share, r.gen_share || 0)), 0.01);
    return `<div class="table-wrap"><table class="cluster-table"><thead><tr><th>cluster</th><th class="r">clips</th><th>share of the data</th>${hasGens ? "<th>share of the generations</th>" : ""}<th class="r">weight</th><th>listen</th></tr></thead><tbody>
      ${table.map((r) => `<tr>
        <td class="num">${r.cluster}</td><td class="r num">${r.segments}</td>
        <td><span class="sharebar"><span style="width:${(r.share / maxShare) * 100}%"></span></span> <span class="num">${(r.share * 100).toFixed(1)}%</span></td>
        ${hasGens ? `<td><span class="sharebar gens"><span style="width:${((r.gen_share || 0) / maxShare) * 100}%"></span></span> <span class="num">${((r.gen_share || 0) * 100).toFixed(1)}%</span></td>` : ""}
        <td class="r num"><b>${r.weight.toFixed(2)}</b>×</td>
        <td class="examples">${(r.examples || []).map((p, i) => `<button class="track mini-track" data-src="${esc(p)}" data-label="${esc(`${opts.label} — cluster ${r.cluster}, example ${i + 1}`)}" title="${esc(p.split("/").pop())}"><span class="ico"></span></button>`).join("")}</td>
      </tr>`).join("")}</tbody></table></div>`;
  }

  async function viewWorkspace(name) {
    NR.setNav("corpus");
    NR.view.innerHTML = `<div class="loading">reading ${esc(name)}…</div>`;
    const [d] = await Promise.all([NR.getJSON(`/api/corpus/workspaces/${encodeURIComponent(name)}`), NR.loadXps()]);
    let dataset = d.datasets[0];
    if (!dataset) {
      NR.view.innerHTML = `<div class="empty">${esc(name)} has no prepared manifest.<br><a href="#/corpus">Corpus</a></div>`;
      return;
    }
    const train = dataset.splits.train, valid = dataset.splits.valid;
    const st = train.stats || {};
    const prep = d.prepared || {};
    const groups = Object.entries(dataset.groups).sort((a, b) => b[1].train - a[1].train);
    const trainTotal = train.clips.length || 1;
    const xps = d.xps.map((sig) => NR.state.bySig.get(sig)).filter(Boolean);
    const ui = { group: "", q: "", sort: "name", split: "train", shown: 120 };

    NR.view.innerHTML = `<div class="ws-page">
      <div class="crumb"><a href="#/corpus">Corpus</a> / ${esc(name)}</div>
      <h1 class="title-xl">${esc(pretty(name))}</h1>
      <div class="byline">${esc(dataset.name)}${d.source ? ` · from ${esc(d.source)}${d.source_exists ? "" : ' <span class="tag gone">gone</span>'}` : ""}${d.corpus ? ` · corpus <code>${esc(d.corpus)}/corpus.json</code>` : ""}</div>
      ${d.training ? `<div class="note warn"><p><span class="k">training</span>A run is training on this workspace now; weights cannot be changed until it ends.</p></div>` : ""}
      ${d.source && !d.source_exists ? `<div class="note"><p><span class="k">only copy</span>The folder it was prepared from is gone: <code>${esc(name)}/data/</code> is now the only copy of these clips.</p></div>` : ""}
      <dl class="facts ws-facts">
        <dt>train</dt><dd>${train.clips.length} clips · ${hours(train.clips.reduce((s, c) => s + c.duration, 0))}</dd>
        <dt>valid</dt><dd>${valid.clips.length} clips · ${hours(valid.clips.reduce((s, c) => s + c.duration, 0))}</dd>
        <dt>clip length</dt><dd>${secs(st.min)} · median ${secs(st.median)} · p90 ${secs(st.p90)} · ${secs(st.max)}</dd>
        <dt>prepared</dt><dd>${NR.ago(d.prepared_at)}${prep.valid_fraction != null ? ` · valid share ${prep.valid_fraction}, seed ${prep.seed}` : ""}</dd>
        ${prep.train_dropped_silent || prep.valid_dropped_silent ? `<dt>dropped</dt><dd>${(prep.train_dropped_silent || 0) + (prep.valid_dropped_silent || 0)} mostly silent clips (over ${prep.max_silence_fraction} silent)</dd>` : ""}
        ${prep.train_speed_variants ? `<dt>speed</dt><dd>${prep.train_speed_variants} speed variants of train clips (${esc(prep.speed_mode)}, ratios ${esc((prep.speed_ratios || []).join(", "))})</dd>` : ""}
      </dl>
      <div class="ws-grid">
        <div class="section"><h3>Longueurs <span class="dim">train clips</span></h3>${histogramHtml(train.clips.map((c) => c.duration))}
          ${dataset.audit.length ? `<div class="table-wrap"><table class="kv audit">${dataset.audit.map((a) => `<tr class="${a.padded_share > 0.05 ? "warn" : ""}">
            <td class="num">segment ${a.segment_duration} s</td>
            <td class="num">${a.padded_share ? `${(a.padded_share * 100).toFixed(a.padded_share < 0.1 ? 1 : 0)}% padded <span class="dim">(${a.padded_clips} clips)</span>` : "no padding"}</td>
            <td class="dim profiles">${a.profiles.map(esc).join(", ")}</td></tr>`).join("")}</table></div>
            <p class="dim small">Clips shorter than a recipe's segment are padded with silence when it trains here; the model learns that silence.</p>` : ""}
        </div>
        <div class="section"><h3>Sources <span class="dim">${groups.length} · click to filter the clips</span></h3>
          <div class="table-wrap"><table class="groups-table">${groups.map(([g, t]) => `<tr data-group="${esc(g)}">
            <td class="gname">${esc(g)}</td>
            <td><span class="sharebar"><span style="width:${(t.train / trainTotal) * 100}%"></span></span></td>
            <td class="r num">${((t.train / trainTotal) * 100).toFixed(1)}%</td>
            <td class="r num dim">${t.train}${t.valid ? ` + ${t.valid}` : ""}</td>
            <td class="r num dim">${hours(t.train_seconds)}</td></tr>`).join("")}</table></div>
        </div>
      </div>
      ${xps.length ? `<div class="section"><h3>Entraînés ici <span class="dim">${xps.length} XPs</span></h3><div class="ws-xps">${xps.map((x) => `<a class="mini" href="#/xp/${encodeURIComponent(x.sig)}" title="${esc(NR.title(x))}">${NR.coverHtml(x, { subtitle: false })}</a>`).join("")}</div></div>` : ""}
      <div class="section" id="rebalance"></div>
      <div class="section"><h3>Clips <span class="dim" id="clip-count"></span></h3>
        <div class="toolbar">
          <input type="search" id="clip-q" placeholder="filter clips" autocomplete="off">
          <div class="seg" id="clip-split"><button data-split="train" class="on">train</button><button data-split="valid">valid</button></div>
          <div class="seg" id="clip-sort"><span>order</span><button data-sort="name" class="on">name</button><button data-sort="short">shortest</button><button data-sort="long">longest</button>${train.clips.some((c) => c.weight != null) ? '<button data-sort="weight">weight</button>' : ""}</div>
          <button class="linkish" id="clip-group" hidden></button>
        </div>
        <div id="clips"></div>
      </div>
    </div>`;

    const drawClips = () => {
      const pool = (ui.split === "train" ? train : valid).clips;
      let list = pool.filter((c) => (!ui.group || c.group === ui.group) && (!ui.q || c.name.toLowerCase().includes(ui.q)));
      const order = { name: (a, b) => a.name.localeCompare(b.name), short: (a, b) => a.duration - b.duration, long: (a, b) => b.duration - a.duration, weight: (a, b) => (b.weight || 1) - (a.weight || 1) }[ui.sort];
      list = list.slice().sort(order);
      document.getElementById("clip-count").textContent = `${list.length} of ${pool.length} ${ui.split} clips`;
      const groupBtn = document.getElementById("clip-group");
      groupBtn.hidden = !ui.group;
      groupBtn.textContent = ui.group ? `source: ${ui.group} ✕` : "";
      document.getElementById("clips").innerHTML = `<div class="tracks clip-tracks">${list.slice(0, ui.shown).map((c) => `<button class="track" data-src="${esc(c.path)}" data-label="${esc(`${name} — ${c.name}`)}">
          <span class="ico"></span><span class="nm" title="${esc(c.name)}">${esc(c.name.replace(/\.[^.]+$/, "").replace(/^.*?__/, ""))}</span>
          <span class="dur num dim">${secs(c.duration)}${c.weight != null ? ` · ${c.weight.toFixed(2)}×` : ""}</span></button>`).join("")}</div>
        ${list.length > ui.shown ? `<button class="more" id="clips-more">${list.length - ui.shown} more</button>` : ""}`;
      const more = document.getElementById("clips-more");
      if (more) more.addEventListener("click", () => { ui.shown += 240; drawClips(); });
    };
    document.getElementById("clip-q").addEventListener("input", (ev) => { ui.q = ev.target.value.trim().toLowerCase(); ui.shown = 120; drawClips(); });
    const segs = (id, key) => document.querySelectorAll(`#${id} [data-${key}]`).forEach((b) => b.addEventListener("click", () => {
      ui[key] = b.dataset[key];
      ui.shown = 120;
      document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b));
      drawClips();
    }));
    segs("clip-split", "split");
    segs("clip-sort", "sort");
    document.getElementById("clip-group").addEventListener("click", () => { ui.group = ""; drawClips(); });
    document.querySelectorAll(".groups-table tr[data-group]").forEach((tr) => tr.addEventListener("click", () => {
      ui.group = ui.group === tr.dataset.group ? "" : tr.dataset.group;
      ui.shown = 120;
      document.querySelectorAll(".groups-table tr").forEach((x) => x.classList.toggle("on", x.dataset.group === ui.group));
      drawClips();
      document.getElementById("clips").scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    drawClips();
    drawRebalance(d, dataset, name);
  }

  function overlaySettings(w) {
    if (w.mode === "folder-mix") return `folder mix: ${w.cluster_table.map((r) => `${esc(r.folder)} ${+(r.share * 100).toFixed(1)}%`).join(", ")}`;
    return `${esc(w.mode)}, k=${w.k}, alpha=${w.alpha}, cap=${w.cap}`;
  }

  function drawRebalance(d, dataset, name) {
    const el = document.getElementById("rebalance");
    const current = dataset.rebalance;
    const saved_weights = dataset.weights || [];
    const job = d.rebalance_job && d.rebalance_job.dataset === dataset.name ? d.rebalance_job : null;
    const gens = d.generation_dirs || [];
    const saved = NR.storage(() => JSON.parse(localStorage.getItem(FORM_KEY + ".rebalance") || "{}"), {});
    const mixable = Object.entries(dataset.groups).filter(([, t]) => t.train > 0).sort((a, b) => a[0].localeCompare(b[0]));
    el.innerHTML = `<h3>Rééquilibrage <span class="dim">cluster weights: how often training draws each kind of clip</span></h3>
      <p class="dim small intro">Clips are clustered by timbre (MFCC and spectral statistics). Without generations the weights flatten the clusters toward equal shares; with a model's generations they favour what that model under-produces. Weights stay within [1/cap, cap] and average 1. Validation is never weighted.</p>
      ${current ? `<div class="plan current">
        <div class="plan-summary"><b>Weights written into this workspace</b> (from before saved weights existed) since ${NR.ago(current.written_at)}: ${esc(current.mode)}, k=${current.k}, alpha=${current.alpha}, cap=${current.cap}, digest <code>${esc(current.weights_digest)}</code>. Every run here uses them, with the label suffix <code>_rw${esc(String(current.weights_digest).slice(0, 8))}</code>.</div>
        ${current.cluster_table && current.cluster_table.length ? clusterTableHtml(current.cluster_table, { label: name }) : ""}
        <div class="form-actions"><button class="ghost danger" id="rb-reset" ${d.training ? "disabled" : ""}>Clear them</button><span class="form-msg" id="rb-reset-msg"></span></div>
      </div>` : ""}
      ${saved_weights.length ? saved_weights.map((w) => `<div class="plan">
        <div class="plan-summary"><b>${esc(w.name)}</b> ${w.stale ? '<span class="tag">stale: the manifest changed since</span>' : ""} · ${overlaySettings(w)} · <code>_rw${esc(String(w.weights_digest).slice(0, 8))}</code> · saved ${NR.ago(w.written_at)}
          <span class="dim">· train with it: pick it in the Atelier, or <code>train run --weights ${esc(w.name)}</code></span></div>
        ${w.mode !== "folder-mix" && w.cluster_table && w.cluster_table.length ? `<details><summary class="dim small">clusters</summary>${clusterTableHtml(w.cluster_table, { label: name })}</details>` : ""}
        <div class="form-actions"><button class="ghost danger" data-delete-weights="${esc(w.name)}">Delete</button><span class="form-msg" data-delete-msg="${esc(w.name)}"></span></div>
      </div>`).join("") : `<p class="small">No saved weights: without them every train clip is drawn equally often. Saved weights never change the workspace; a run uses one only when you pick it.</p>`}
      <form id="rb-form" class="ab-form rb-form" autocomplete="off">
        <fieldset class="grid-2"><legend>${current ? "Compute new weights" : "Compute weights"} <em>data rebalance, then save them under a name</em></legend>
          <label><span>clusters k</span><input type="number" name="k" min="2" max="64" value="${esc(saved.k ?? 10)}"></label>
          <label><span>alpha <em>strength; 0.5 validated, 1.0 overshoots</em></span><input type="number" name="alpha" min="0.05" max="2" step="0.05" value="${esc(saved.alpha ?? 0.5)}"></label>
          <label><span>cap <em>max weight ratio</em></span><input type="number" name="cap" min="1.1" max="20" step="0.1" value="${esc(saved.cap ?? 4)}"></label>
          <label><span>seed</span><input type="number" name="seed" value="${esc(saved.seed ?? 0)}"></label>
          <div class="span-all gens-pick">${gens.length ? `<span class="dim small">coverage mode: tick generations of models trained here (none ticked = flatten the clusters)</span>
            <div class="gens-list">${gens.slice(0, 40).map((g) => { const x = NR.state.bySig.get(g.sig); return `<label class="check"><input type="checkbox" name="gens" value="${esc(g.path)}"> <span>${esc(x ? NR.title(x) : g.sig)} <span class="dim">${esc(g.kind)} · ${g.wavs} wav · ${NR.ago(g.when)} · ${esc(g.path)}</span></span></label>`; }).join("")}</div>`
            : `<span class="dim small">No generations from models trained here, so the weights flatten the clusters.</span>`}</div>
        </fieldset>
        <div class="form-actions"><button class="solid big" type="submit">Compute</button><span class="form-msg" id="rb-msg"></span></div>
      </form>
      <form id="mix-form" class="ab-form rb-form" autocomplete="off">
        <fieldset><legend>Mix by source <em>the share of training draws each source gets; saved at once, nothing is computed</em></legend>
          <p class="dim small intro">Give each source the share you want, as relative numbers: 3 and 1 mean three quarters and a quarter. With the default per-file sampling each source's share of draws comes out exact; with "in proportion to length" a clip's length counts too.</p>
          <table class="groups-table mix-table">${mixable.map(([g, t]) => `<tr>
            <td class="gname">${esc(g)}</td>
            <td class="r num dim">${t.train} train clips</td>
            <td><input type="number" min="0" step="any" name="share" data-group="${esc(g)}" value="1" aria-label="share of ${esc(g)}"></td>
            <td class="r num" data-share="${esc(g)}"></td></tr>`).join("")}</table>
        </fieldset>
        <div class="form-actions"><label><span class="dim small">name</span> <input name="name" pattern="[a-z0-9][a-z0-9_.\\-]*" size="18" required></label><button class="solid big" type="submit">Save mix</button><span class="form-msg" id="mix-msg"></span></div>
      </form>
      <div id="rb-job"></div>`;

    const reset = document.getElementById("rb-reset");
    if (reset) reset.addEventListener("click", async () => {
      if (!confirm(`Clear the weights of ${name}? Every clip is then drawn equally often, and training here gets its old signature back.`)) return;
      try {
        await NR.postJSON(`/api/corpus/workspaces/${encodeURIComponent(name)}/rebalance/reset`, { dataset: dataset.name });
        viewWorkspace(name);
      } catch (err) { document.getElementById("rb-reset-msg").textContent = err.message; }
    });

    el.querySelectorAll("[data-delete-weights]").forEach((button) => button.addEventListener("click", async () => {
      const weights = button.dataset.deleteWeights;
      if (!confirm(`Delete the saved weights ${weights}? Runs trained with them read their manifest from there: resuming one would then refuse.`)) return;
      try {
        await NR.postJSON(`/api/corpus/workspaces/${encodeURIComponent(name)}/weights/delete`, { name: weights });
        viewWorkspace(name);
      } catch (err) { el.querySelector(`[data-delete-msg="${CSS.escape(weights)}"]`).textContent = err.message; }
    }));

    const form = document.getElementById("rb-form");
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const f = form.elements;
      const body = {
        dataset: dataset.name, k: +f.k.value, alpha: +f.alpha.value, cap: +f.cap.value, seed: +f.seed.value,
        gens_dirs: [...form.querySelectorAll("input[name=gens]:checked")].map((i) => i.value),
      };
      NR.storage(() => localStorage.setItem(FORM_KEY + ".rebalance", JSON.stringify({ k: body.k, alpha: body.alpha, cap: body.cap, seed: body.seed })));
      const msg = document.getElementById("rb-msg");
      msg.classList.remove("err");
      msg.textContent = "starting…";
      try {
        const started = await NR.postJSON(`/api/corpus/workspaces/${encodeURIComponent(name)}/rebalance`, body);
        msg.textContent = "";
        watch({ ...started, dataset: dataset.name });
      } catch (err) { msg.textContent = err.message; msg.classList.add("err"); }
    });

    const mixForm = document.getElementById("mix-form");
    const refreshShares = () => {
      const rows = [...mixForm.querySelectorAll("input[name=share]")].map((i) => [i.dataset.group, Math.max(0, +i.value || 0)]);
      const total = rows.reduce((sum, [, share]) => sum + share, 0);
      rows.forEach(([group, share]) => { mixForm.querySelector(`[data-share="${CSS.escape(group)}"]`).textContent = total ? `${+(share / total * 100).toFixed(1)}%` : "—"; });
    };
    mixForm.addEventListener("input", refreshShares);
    refreshShares();
    mixForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const msg = document.getElementById("mix-msg");
      const mix = Object.fromEntries([...mixForm.querySelectorAll("input[name=share]")].map((i) => [i.dataset.group, +i.value || 0]));
      msg.classList.remove("err");
      msg.textContent = "saving…";
      try {
        await NR.postJSON(`/api/corpus/workspaces/${encodeURIComponent(name)}/mix`, { dataset: dataset.name, mix, name: mixForm.querySelector("input[name=name]").value.trim() });
        viewWorkspace(name);
      } catch (err) { msg.textContent = err.message; msg.classList.add("err"); }
    });

    function watch(j) {
      const box = document.getElementById("rb-job");
      if (!box || !j) return;
      const [done, total] = (j.progress || "").split(" ").pop().split("/").map(Number);
      const result = j.result;
      box.innerHTML = `<div class="plan">
        <div>${stateHtml(j)} <span class="dim">weights computed ${NR.ago(j.created_at)} · k=${esc(j.params && j.params.k)}, alpha=${esc(j.params && j.params.alpha)}, cap=${esc(j.params && j.params.cap)}${j.gens_dirs && j.gens_dirs.length ? ` · coverage against ${j.gens_dirs.length} folder${j.gens_dirs.length > 1 ? "s" : ""}` : ""}</span>
          <a class="linkish" href="#/corpus/job/${encodeURIComponent(j.id)}">log</a></div>
        ${j.active ? `<div class="progress"><span style="width:${total ? (done / total) * 100 : 0}%"></span></div><div class="progress-label num">${esc(j.progress || "starting")} clips read</div>` : ""}
        ${j.state === "failed" || j.state === "exited" ? `<p class="err">It did not finish; the log says why.</p>` : ""}
        ${result && j.saved_as ? `<p class="dim">Saved ${NR.ago(j.applied_at)} as <b>${esc(j.saved_as)}</b>, above.</p>` : ""}
        ${result && !j.saved_as ? `<div class="plan-summary">${esc(result.mode)}: ${result.readable_segments} of ${result.segments} clips clustered${result.generations ? `, ${result.generations} generations placed` : ""} · digest <code>${esc(result.digest)}</code></div>
          ${clusterTableHtml(result.cluster_table, { label: name })}
          ${result.stale ? `<p class="dim">The manifest changed since these were computed; compute again.</p>`
            : `<div class="form-actions"><label><span class="dim small">name</span> <input id="rb-name" value="${esc(`${result.mode === "coverage" ? "cov" : "flat"}_a${String(j.params && j.params.alpha).replace(".", "")}`)}" pattern="[a-z0-9][a-z0-9_.\\-]*" size="18"></label><button class="solid big" id="rb-apply">Save these weights</button><span class="form-msg" id="rb-apply-msg"></span></div>`}` : ""}
      </div>`;
      const apply = document.getElementById("rb-apply");
      if (apply) apply.addEventListener("click", async () => {
        try {
          await NR.postJSON(`/api/corpus/workspaces/${encodeURIComponent(name)}/rebalance/save`, { digest: result.digest, name: document.getElementById("rb-name").value.trim() });
          viewWorkspace(name);
        } catch (err) { document.getElementById("rb-apply-msg").textContent = err.message; }
      });
      if (j.active) later(2000, async () => {
        const fresh = (await NR.getJSON(`/api/corpus/workspaces/${encodeURIComponent(name)}/rebalance`)).job;
        watch(fresh);
      });
    }
    if (job) watch(job);
  }

  // ---------- a new corpus ----------
  async function viewNew() {
    NR.setNav("corpus");
    NR.view.innerHTML = `<div class="loading">reading the library (the first time takes a few seconds)…</div>`;
    const lib = await NR.getJSON("/api/corpus/library");
    const saved = NR.storage(() => JSON.parse(localStorage.getItem(FORM_KEY) || "{}"), {});
    const picked = new Set((saved.sources || []).filter((p) => lib.folders.some((f) => f.path === p)));
    // files left out by hand, kept even while their folder is unticked, sent only for ticked ones
    const excluded = new Set(saved.exclude || []);
    let selections = { saved: [], corpora: [] };
    const libPrefs = NR.storage(() => JSON.parse(localStorage.getItem(FORM_KEY + ".library") || "{}"), {});
    const ui = { q: "", artist: "", assembled: false, unused: false, tickedOnly: false, sort: libPrefs.sort || "folder", dir: libPrefs.dir || 1, open: new Set() };
    const tracksOf = new Map();
    const datasets = lib.datasets;
    // the dataset most recipes train on, unless one was picked before
    const datasetPicked = datasets.some((d) => d.name === saved.dataset) ? saved.dataset
      : [...datasets].sort((a, b) => b.profiles.length - a.profiles.length)[0]?.name;
    const segs = (list) => list.map((s) => `${s} s`).join(", ");
    // tags spell one artist several ways ("DJ Lostboi", "dj lostboi", "Kali Malone featuring …"):
    // an artist matches every tag that holds its name as whole words, whatever the case
    const artistRe = new Map();
    const artistMatch = (name, tags) => {
      if (!artistRe.has(name)) artistRe.set(name, new RegExp(`(^|[^\\p{L}\\p{N}])${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}($|[^\\p{L}\\p{N}])`, "iu"));
      return tags.some((t) => artistRe.get(name).test(t));
    };
    const spellings = new Map();
    for (const x of lib.folders) for (const a of x.artists) if (!spellings.has(a.toLowerCase())) spellings.set(a.toLowerCase(), a);
    const artists = [...spellings.values()].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
    const byArtist = new Map(artists.map((a) => [a, lib.folders.filter((x) => artistMatch(a, x.artists)).length]));
    const libCols = [
      ["folder", (x) => x.name.toLowerCase()],
      ["artist", (x) => (x.artists[0] || "\uffff").toLowerCase()],
      ["year", (x) => (x.years ? x.years[0] : 0), "r"],
      ["files", (x) => x.files, "r"],
      ["audio", (x) => x.seconds, "r"],
      ["modified", (x) => x.modified_at || ""],
      ["prepared into", (x) => x.workspaces.length],
    ];
    NR.view.innerHTML = `<div class="corpus new-corpus">
      <div class="crumb"><a href="#/corpus">Corpus</a> / new</div>
      <h1 class="title-xl">Nouveau corpus</h1>
      <p class="dim intro">Pick source folders from the library; their audio is gathered, whole or cut into slices, and split into a new dataset, <code>datasets/&lt;name&gt;</code>, in one step (<code>data build</code>). Clips are named <code>&lt;source&gt;__&lt;file&gt;</code>, so every clip stays attributable to its source.</p>
      <div class="section"><h3>Bibliothèque <span class="dim">${lib.folders.length} folders in ${lib.roots.map(esc).join(" and ")}</span></h3>
        <div class="toolbar">
          <input type="search" id="lib-q" placeholder="folder, artist, album or genre" autocomplete="off">
          <label class="check small"><input type="checkbox" id="lib-ticked"> ticked only</label>
          <label class="check small"><input type="checkbox" id="lib-unused"> never prepared</label>
          <label class="check small"><input type="checkbox" id="lib-assembled"> show assembled corpora</label>
          <span class="count" id="lib-count"></span>
        </div>
        <div class="toolbar selections">
          <select id="sel-list" aria-label="selections"></select>
          <button type="button" class="ghost" id="sel-load" disabled>load</button>
          <button type="button" class="linkish" id="sel-delete" hidden>delete</button>
          <span class="sel-save"><input id="sel-name" placeholder="name the ticked folders" autocomplete="off"><button type="button" class="ghost" id="sel-save">save</button></span>
          <span class="form-msg" id="sel-msg"></span>
        </div>
        <div class="toolbar lib-actions">
          <button type="button" class="linkish" id="lib-artist" hidden></button>
          <button type="button" class="ghost" id="lib-tick-shown"></button>
          <button type="button" class="ghost" id="lib-untick"></button>
        </div>
        ${artists.length ? `<details class="facets"><summary>${artists.length} artists <span class="dim">from the files' tags; pick one to see only their folders</span></summary>
          <div class="facet-list">${artists.map((a) => `<button type="button" class="pill facet" data-artist="${esc(a)}">${esc(a)}${byArtist.get(a) > 1 ? ` <span class="dim">${byArtist.get(a)}</span>` : ""}</button>`).join("")}</div></details>` : ""}
        <div class="table-wrap lib-wrap"><table class="lib-table"><thead><tr><th></th>${libCols.map(([n, , c], i) => `<th data-col="${i}" class="${c || ""}">${n}</th>`).join("")}</tr></thead><tbody id="lib-rows"></tbody></table></div>
      </div>
      <form id="corpus-form" class="ab-form" autocomplete="off">
        <fieldset class="grid-2"><legend>Corpus</legend>
          <label><span>name <em>letters, digits, . _ -</em></span><input name="name" required placeholder="drones_10s" value="${esc(saved.name || "")}"></label>
          <div class="picked span-all" id="picked"></div>
        </fieldset>
        <fieldset class="grid-2"><legend>Clips</legend>
          <label class="radio"><input type="radio" name="mode" value="whole" ${saved.slice ? "" : "checked"}><span><b>whole files</b> <span class="dim">hard-linked, formats kept; training crops random segments from them</span></span></label>
          <label class="radio"><input type="radio" name="mode" value="slices" ${saved.slice ? "checked" : ""}><span><b>fixed-length slices</b> <span class="dim">FLAC, 32 kHz mono; every clip the same length</span></span></label>
          <label data-slices><span>slice length, s</span><input type="number" name="slice_seconds" min="0.5" step="0.5" value="${esc(saved.slice || 10)}"></label>
          <label data-slices><span>hop, s <em>blank = slice length, no overlap</em></span><input type="number" name="hop_seconds" min="0.1" step="0.5" value="${esc(saved.hop || "")}"></label>
          <label><span>shortest clip kept, s <em>blank = the slice length (whole files: any)</em></span><input type="number" name="min_clip_seconds" min="0" step="0.5" value="${esc(saved.min_clip || "")}"></label>
          <label class="check span-all"><input type="checkbox" name="skip_duplicates" ${saved.skip_duplicates === false ? "" : "checked"}> leave probable duplicates out <span class="dim">(the same recording in two ticked folders: keeps the lossless, else the larger copy)</span></label>
        </fieldset>
        <fieldset class="grid-2"><legend>Dataset <em>data build, into datasets/&lt;name&gt;</em></legend>
          <label class="check span-all"><input type="checkbox" name="prepare" ${saved.prepare === false ? "" : "checked"}> split it into a dataset <span class="dim">(untick to only gather the clips into training_datasets/&lt;name&gt;/clips)</span></label>
          <label data-prep><span>dataset <em>the recipes that can train on it</em></span><select name="dataset">${datasets.map((d) => `<option value="${esc(d.name)}" title="${esc(d.profiles.join(", "))}" ${d.name === datasetPicked ? "selected" : ""}>${esc(d.name)} · ${d.profiles.length} recipe${d.profiles.length > 1 ? "s" : ""}${d.segments.length ? ` at ${segs(d.segments)}` : ""}</option>`).join("")}</select></label>
          <label data-prep><span>validation share</span><input type="number" name="valid_fraction" min="0.01" max="0.49" step="0.01" value="${esc(saved.valid_fraction || 0.1)}"></label>
          <label data-prep><span>split seed</span><input type="number" name="seed" value="${esc(saved.seed ?? 42)}"></label>
          <label data-prep><span>drop clips more silent than <em>fraction of frames</em></span><input type="number" name="max_silence_fraction" min="0.05" max="1" step="0.05" value="${esc(saved.max_silence_fraction || 0.85)}"></label>
          <label data-prep><span>speed variants <em>optional, train only, e.g. 0.9,1.1</em></span><input name="speed_ratios" placeholder="none" value="${esc(saved.speed_ratios || "")}"></label>
        </fieldset>
        <div class="form-actions"><button class="solid big" type="submit">Preview</button><span class="form-msg" id="corpus-msg"></span></div>
      </form>
      <div id="corpus-plan"></div>
    </div>`;

    const form = document.getElementById("corpus-form");
    const f = form.elements;
    let preview = null;
    const invalidate = () => { if (preview) { preview = null; document.getElementById("corpus-plan").innerHTML = `<p class="dim">Changed — preview again.</p>`; } };
    const matches = (x) => {
      if (ui.tickedOnly && !picked.has(x.path)) return false;
      if (!ui.assembled && x.assembled && !picked.has(x.path)) return false;
      if (ui.unused && x.workspaces.length) return false;
      if (ui.artist && !artistMatch(ui.artist, x.artists)) return false;
      return !ui.q || [x.path, ...x.artists, ...x.albums, ...x.genres].join("\n").toLowerCase().includes(ui.q);
    };
    const shown = () => {
      const col = libCols.find(([n]) => n === ui.sort) || libCols[0];
      return lib.folders.filter(matches).sort((a, b) => { const va = col[1](a), vb = col[1](b); return (va < vb ? -1 : va > vb ? 1 : 0) * ui.dir; });
    };
    const savePicks = () => NR.storage(() => localStorage.setItem(FORM_KEY, JSON.stringify({ ...NR.storage(() => JSON.parse(localStorage.getItem(FORM_KEY) || "{}"), {}), sources: [...picked], exclude: [...excluded] })));
    const excludedIn = (path) => [...excluded].filter((p) => p.startsWith(`${path}/`));
    const nameOf = (path) => path.split("/").pop();
    const dupLine = (x) => (x.duplicates && x.duplicates.length
      ? `<div class="dim small sub dup-line">same recordings as ${x.duplicates.slice(0, 2).map((d) => `<button type="button" class="linkish" data-find="${esc(nameOf(d.path))}" title="show ${esc(d.path)}">${esc(nameOf(d.path))}</button> (${d.files})`).join(", ")}${x.duplicates.length > 2 ? ` and ${x.duplicates.length - 2} more` : ""}</div>`
      : "");
    const yearsText = (y) => (!y ? "" : y[0] === y[1] ? String(y[0]) : `${y[0]}–${y[1]}`);
    const albumLine = (x) => {
      const others = x.albums.filter((a) => !x.name.toLowerCase().includes(a.toLowerCase()));
      return others.length ? `<div class="dim small sub">${others.slice(0, 2).map(esc).join(" · ")}${x.albums.length > 2 ? ` · +${x.albums.length - 2}` : ""}</div>` : "";
    };
    const tracksHtml = (x) => {
      const t = tracksOf.get(x.path);
      if (!t) return `<p class="dim small">reading…</p>`;
      if (t.error) return `<p class="err small">${esc(t.error)}</p>`;
      const main = x.artists[0];
      const limit = 200;
      const out = excludedIn(x.path).length;
      return `<p class="dim small lib-tracks-note">Untick a file to leave it out of the corpus${out ? ` · ${out} left out <button type="button" class="linkish" data-keep-all="${esc(x.path)}">keep them all</button>` : ""}.</p>
        <div class="tracks lib-tracks">${t.files.slice(0, limit).map((f) => `<div class="lib-track ${excluded.has(f.path) ? "out" : ""}">
          <input type="checkbox" data-keep="${esc(f.path)}" ${excluded.has(f.path) ? "" : "checked"} aria-label="keep ${esc(f.name)}" title="keep this file in the corpus">
          <button type="button" class="track" ${f.playable ? "" : "disabled title=\"the browser cannot play this format\""} data-src="${esc(f.path)}" data-label="${esc(`${x.name} — ${f.title || f.name}`)}">
          <span class="ico"></span><span class="nm" title="${esc(f.name)}${f.duplicates.length ? `\nsame recording as:\n${esc(f.duplicates.join("\n"))}` : ""}">${esc(f.title || f.name.replace(/\.[^.]+$/, ""))}${f.artist && f.artist !== main ? ` <span class="dim">${esc(f.artist)}</span>` : ""}</span>${f.duplicates.length ? `<span class="dup-mark" title="same recording as ${esc(f.duplicates.join(", "))}">≈</span>` : ""}
          <span class="dur num dim">${f.duration != null ? secs(f.duration) : "unreadable"}</span></button></div>`).join("")}</div>
        ${t.files.length > limit ? `<p class="dim small">and ${t.files.length - limit} more files</p>` : ""}`;
    };
    const drawLib = () => {
      const list = shown();
      const shownPaths = list.map((x) => x.path);
      document.getElementById("lib-count").textContent = `${list.length} shown · ${picked.size} ticked`;
      document.querySelectorAll(".lib-table th[data-col]").forEach((th) => {
        const name = libCols[+th.dataset.col][0];
        th.classList.toggle("sorted", name === ui.sort);
        th.textContent = name + (name === ui.sort ? (ui.dir > 0 ? " ↑" : " ↓") : "");
      });
      const unticked = shownPaths.filter((p) => !picked.has(p)).length;
      const tickShown = document.getElementById("lib-tick-shown");
      tickShown.textContent = `tick the ${unticked} shown`;
      tickShown.hidden = !unticked || list.length === lib.folders.length;
      const untick = document.getElementById("lib-untick");
      untick.textContent = `untick all (${picked.size})`;
      untick.disabled = !picked.size;
      const artistBtn = document.getElementById("lib-artist");
      artistBtn.hidden = !ui.artist;
      artistBtn.textContent = ui.artist ? `artist: ${ui.artist} ✕` : "";
      document.querySelectorAll(".facet").forEach((b) => b.classList.toggle("on", b.dataset.artist === ui.artist));
      document.getElementById("lib-rows").innerHTML = list.map((x) => `<tr data-path="${esc(x.path)}" class="${picked.has(x.path) ? "ticked" : ""}">
        <td><input type="checkbox" ${picked.has(x.path) ? "checked" : ""} aria-label="pick ${esc(x.name)}"></td>
        <td class="folder"><button type="button" class="expand linkish" aria-expanded="${ui.open.has(x.path)}" title="list and play its files">${ui.open.has(x.path) ? "▾" : "▸"}</button> ${esc(x.name)} <span class="dim">${esc(x.root)}</span>${x.assembled ? ` <span class="tag" title="${x.corpus ? "built by data corpus" : "its files are named <source>__<file>: assembled from other folders"}">assembled</span>` : ""}${albumLine(x)}${dupLine(x)}</td>
        <td class="artists">${x.artists.slice(0, 2).map((a) => `<button type="button" class="linkish" data-artist="${esc(a)}">${esc(a)}</button>`).join(", ")}${x.artist_count > 2 ? ` <span class="dim">+${x.artist_count - 2}</span>` : ""}${x.artists.length && x.tagged < x.files ? ` <span class="dim" title="${x.files - x.tagged} files carry no artist tag">*</span>` : ""}${x.genres.length ? `<div class="dim small sub">${x.genres.slice(0, 2).map(esc).join(", ")}</div>` : ""}</td>
        <td class="r num dim">${yearsText(x.years)}</td>
        <td class="r num" title="${esc(Object.entries(x.formats).map(([k, n]) => `${k} ${n}`).join(", "))}">${excludedIn(x.path).length ? `<span title="${excludedIn(x.path).length} left out by hand">${x.files - excludedIn(x.path).length}<span class="dim">/${x.files}</span></span>` : x.files}${x.unreadable ? ` <span class="err" title="${x.unreadable} unreadable">!</span>` : ""}</td>
        <td class="r num">${hours(x.seconds)}</td>
        <td class="dim nowrap" title="${esc((x.modified_at || "").replace("T", " ").slice(0, 16))}">${NR.ago(x.modified_at)}</td>
        <td class="dim">${x.workspaces.map(esc).join(", ")}</td></tr>
        ${ui.open.has(x.path) ? `<tr class="lib-detail"><td></td><td colspan="${libCols.length}">${tracksHtml(x)}</td></tr>` : ""}`).join("");
      const chosen = lib.folders.filter((x) => picked.has(x.path));
      const outSeconds = (x) => {
        const files = (tracksOf.get(x.path) || {}).files || [];
        return files.filter((f) => excluded.has(f.path)).reduce((s, f) => s + (f.duration || 0), 0);
      };
      const kept = (x) => x.seconds - outSeconds(x);
      document.getElementById("picked").innerHTML = chosen.length
        ? `<span class="dim">sources:</span> ${chosen.map((x) => { const out = excludedIn(x.path).length; return `<span class="pill">${esc(x.name)} <span class="dim">${hours(kept(x))}${out ? ` · ${out} left out` : ""}</span> <button type="button" class="unpick" data-unpick="${esc(x.path)}" aria-label="untick ${esc(x.name)}">✕</button></span>`; }).join(" ")} <span class="dim">· ${hours(chosen.reduce((s, x) => s + kept(x), 0))} in all</span>`
        : `<span class="dim">Tick source folders in the library above.</span>`;
      drawSelections();
    };
    const changed = () => { savePicks(); invalidate(); drawLib(); };
    document.getElementById("lib-q").addEventListener("input", (ev) => { ui.q = ev.target.value.trim().toLowerCase(); drawLib(); });
    document.getElementById("lib-assembled").addEventListener("change", (ev) => { ev.stopPropagation(); ui.assembled = ev.target.checked; drawLib(); });
    document.getElementById("lib-ticked").addEventListener("change", (ev) => { ev.stopPropagation(); ui.tickedOnly = ev.target.checked; drawLib(); });
    document.getElementById("lib-unused").addEventListener("change", (ev) => { ev.stopPropagation(); ui.unused = ev.target.checked; drawLib(); });
    document.getElementById("lib-artist").addEventListener("click", () => { ui.artist = ""; drawLib(); });
    document.getElementById("lib-tick-shown").addEventListener("click", () => { shown().forEach((x) => picked.add(x.path)); changed(); });
    document.getElementById("lib-untick").addEventListener("click", () => { picked.clear(); changed(); });
    document.querySelectorAll(".facet").forEach((b) => b.addEventListener("click", () => { ui.artist = ui.artist === b.dataset.artist ? "" : b.dataset.artist; drawLib(); }));
    document.querySelectorAll(".lib-table th[data-col]").forEach((th) => th.addEventListener("click", () => {
      const name = libCols[+th.dataset.col][0];
      if (ui.sort === name) ui.dir *= -1; else { ui.sort = name; ui.dir = ["folder", "artist"].includes(name) ? 1 : -1; }
      NR.storage(() => localStorage.setItem(FORM_KEY + ".library", JSON.stringify({ sort: ui.sort, dir: ui.dir })));
      drawLib();
    }));
    document.getElementById("picked").addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-unpick]");
      if (btn) { picked.delete(btn.dataset.unpick); changed(); }
    });
    document.getElementById("lib-rows").addEventListener("click", async (ev) => {
      const keep = ev.target.closest("[data-keep]");
      if (keep) { keep.checked ? excluded.delete(keep.dataset.keep) : excluded.add(keep.dataset.keep); changed(); return; }
      const keepAll = ev.target.closest("[data-keep-all]");
      if (keepAll) { excludedIn(keepAll.dataset.keepAll).forEach((p) => excluded.delete(p)); changed(); return; }
      const find = ev.target.closest("[data-find]");
      if (find) {
        ui.q = find.dataset.find.toLowerCase();
        ui.artist = "";
        ui.assembled = true;
        document.getElementById("lib-q").value = find.dataset.find;
        document.getElementById("lib-assembled").checked = true;
        drawLib();
        return;
      }
      const row = ev.target.closest("tr[data-path]");
      if (!row) return;
      const path = row.dataset.path;
      const artist = ev.target.closest("[data-artist]");
      if (artist) { ui.artist = artist.dataset.artist; drawLib(); return; }
      if (ev.target.closest(".expand")) {
        if (ui.open.has(path)) { ui.open.delete(path); drawLib(); return; }
        ui.open.add(path);
        drawLib();
        if (!tracksOf.has(path)) {
          try { tracksOf.set(path, await NR.getJSON(`/api/corpus/library/folder?path=${encodeURIComponent(path)}`)); }
          catch (err) { tracksOf.set(path, { error: err.message }); }
          drawLib();
        }
        return;
      }
      const on = ev.target.tagName === "INPUT" ? ev.target.checked : !picked.has(path);
      on ? picked.add(path) : picked.delete(path);
      changed();
    });
    function drawSelections() {
      const list = document.getElementById("sel-list");
      if (!list) return;
      const current = list.value;
      const opt = (kind, x) => `<option value="${kind}:${esc(x.name)}">${esc(x.name)} · ${x.sources.length} folder${x.sources.length > 1 ? "s" : ""}${x.exclude.length ? `, ${x.exclude.length} left out` : ""}${x.missing.length ? ` (${x.missing.length} gone)` : ""}</option>`;
      list.innerHTML = `<option value="">${selections.saved.length || selections.corpora.length ? "a saved selection…" : "no saved selections yet"}</option>
        ${selections.saved.length ? `<optgroup label="saved">${selections.saved.map((x) => opt("saved", x)).join("")}</optgroup>` : ""}
        ${selections.corpora.length ? `<optgroup label="sources of a built corpus">${selections.corpora.map((x) => opt("corpus", x)).join("")}</optgroup>` : ""}`;
      list.value = [...list.options].some((o) => o.value === current) ? current : "";
      document.getElementById("sel-load").disabled = !list.value;
      document.getElementById("sel-delete").hidden = !list.value.startsWith("saved:");
      document.getElementById("sel-save").disabled = !picked.size;
    }
    const chosenSelection = () => {
      const value = document.getElementById("sel-list").value;
      const kind = value.slice(0, value.indexOf(":")), name = value.slice(value.indexOf(":") + 1);
      const sel = (kind === "saved" ? selections.saved : selections.corpora).find((x) => x.name === name);
      return sel ? { kind, sel } : null;
    };
    const selMsg = (text, err) => { const m = document.getElementById("sel-msg"); m.textContent = text; m.classList.toggle("err", !!err); };
    document.getElementById("sel-list").addEventListener("change", (ev) => { ev.stopPropagation(); drawSelections(); });
    document.getElementById("sel-load").addEventListener("click", () => {
      const chosen = chosenSelection();
      if (!chosen) return;
      const { kind, sel } = chosen;
      const known = sel.sources.filter((path) => lib.folders.some((x) => x.path === path));
      picked.clear();
      known.forEach((path) => picked.add(path));
      known.forEach((path) => excludedIn(path).forEach((e) => excluded.delete(e)));
      sel.exclude.forEach((path) => excluded.add(path));
      if (kind === "corpus" && sel.slicing) {
        const whole = sel.slicing.mode === "whole";
        form.querySelector(`input[name=mode][value=${whole ? "whole" : "slices"}]`).checked = true;
        if (!whole) { f.slice_seconds.value = sel.slicing.slice_seconds; f.hop_seconds.value = sel.slicing.hop_seconds !== sel.slicing.slice_seconds ? sel.slicing.hop_seconds : ""; }
        f.min_clip_seconds.value = sel.slicing.min_clip_seconds != null && sel.slicing.min_clip_seconds !== (whole ? 0 : sel.slicing.slice_seconds) ? sel.slicing.min_clip_seconds : "";
        f.skip_duplicates.checked = sel.skip_duplicates;
        sync();
      }
      ui.tickedOnly = true;
      document.getElementById("lib-ticked").checked = true;
      selMsg(`loaded ${sel.name}: ${known.length} folder${known.length === 1 ? "" : "s"}${sel.exclude.length ? `, ${sel.exclude.length} file${sel.exclude.length === 1 ? "" : "s"} left out` : ""}${kind === "corpus" ? ", and its slicing" : ""}${sel.sources.length > known.length ? ` · ${sel.sources.length - known.length} no longer in the library` : ""}`);
      changed();
    });
    document.getElementById("sel-delete").addEventListener("click", async () => {
      const chosen = chosenSelection();
      if (!chosen || !confirm(`Delete the saved selection ${chosen.sel.name}? The folders themselves are not touched.`)) return;
      try { selections = await NR.postJSON("/api/corpus/selections/delete", { name: chosen.sel.name }); selMsg(`deleted ${chosen.sel.name}`); drawSelections(); }
      catch (err) { selMsg(err.message, true); }
    });
    document.getElementById("sel-save").addEventListener("click", async () => {
      const name = document.getElementById("sel-name").value.trim();
      if (!name) { selMsg("give the selection a name", true); return; }
      if (selections.saved.some((x) => x.name === name) && !confirm(`Replace the saved selection ${name}?`)) return;
      try {
        selections = await NR.postJSON("/api/corpus/selections", { name, sources: [...picked], exclude: [...excluded] });
        document.getElementById("sel-name").value = "";
        drawSelections();
        document.getElementById("sel-list").value = `saved:${name}`;
        drawSelections();
        selMsg(`saved ${name}`);
      } catch (err) { selMsg(err.message, true); }
    });
    NR.getJSON("/api/corpus/selections").then((got) => { selections = got; drawSelections(); }).catch(() => {});
    const sync = () => {
      const slices = form.querySelector("input[name=mode]:checked").value === "slices";
      form.querySelectorAll("[data-slices]").forEach((x) => (x.hidden = !slices));
      form.querySelectorAll("[data-prep]").forEach((x) => (x.hidden = !f.prepare.checked));
    };
    form.addEventListener("input", () => { sync(); invalidate(); });
    form.addEventListener("change", () => { sync(); invalidate(); });
    sync();
    drawLib();

    const request = () => {
      const slices = form.querySelector("input[name=mode]:checked").value === "slices";
      const num = (v) => (v === "" ? null : +v);
      return {
        name: f.name.value.trim(), sources: [...picked],
        slice_seconds: slices ? num(f.slice_seconds.value) : null, hop_seconds: slices ? num(f.hop_seconds.value) : null,
        min_clip_seconds: num(f.min_clip_seconds.value),
        prepare: f.prepare.checked, dataset: f.dataset.value,
        valid_fraction: +f.valid_fraction.value, seed: +f.seed.value, max_silence_fraction: +f.max_silence_fraction.value,
        speed_ratios: f.speed_ratios.value.trim(),
        exclude: [...excluded].filter((path) => [...picked].some((folder) => path.startsWith(`${folder}/`))),
        skip_duplicates: f.skip_duplicates.checked,
      };
    };
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const req = request();
      NR.storage(() => localStorage.setItem(FORM_KEY, JSON.stringify({
        name: req.name, sources: req.sources, slice: req.slice_seconds, hop: req.hop_seconds, min_clip: req.min_clip_seconds,
        prepare: req.prepare, dataset: req.dataset, valid_fraction: req.valid_fraction, seed: req.seed,
        max_silence_fraction: req.max_silence_fraction, speed_ratios: req.speed_ratios,
        exclude: [...excluded], skip_duplicates: req.skip_duplicates,
      })));
      const msg = document.getElementById("corpus-msg");
      msg.classList.remove("err");
      msg.textContent = "planning…";
      document.getElementById("corpus-plan").innerHTML = "";
      try {
        preview = await NR.postJSON("/api/corpus/preview", { request: req });
        preview.request_sent = req;
        msg.textContent = "";
        drawPreview(preview);
      } catch (err) { msg.textContent = err.message; msg.classList.add("err"); }
    });

    function drawPreview(p) {
      const st = p.stats || {};
      const sources = Object.entries(p.per_source);
      document.getElementById("corpus-plan").innerHTML = `<div class="plan">
        <div class="plan-head">
          <div><div class="dim">clips</div><div class="plan-sig">${p.clips}</div></div>
          <div><div class="dim">audio</div><div>${hours(p.seconds)}</div></div>
          <div><div class="dim">clip length</div><div>${secs(st.min)} · median ${secs(st.median)} · ${secs(st.max)}</div></div>
          <div><div class="dim">into</div><div>${esc(p.output_dir)}${p.prepare ? "" : "/clips"}</div></div>
          ${p.padding.length ? `<div><div class="dim">padded with silence, by segment</div><div>${p.padding.map((a) => `<span title="${esc(a.profiles.join(", "))}">${a.segment_duration} s: <span class="num">${(a.padded_share * 100).toFixed(0)}%</span> <span class="dim">(${a.profiles.length} recipe${a.profiles.length > 1 ? "s" : ""})</span></span>`).join(" · ")}</div></div>` : ""}
        </div>
        ${p.blockers.map((b) => `<div class="note warn"><p><span class="k">blocked</span>${esc(b)}</p></div>`).join("")}
        ${p.warnings.map((w) => `<div class="note"><p><span class="k">note</span>${esc(w)}</p></div>`).join("")}
        <table class="plan-table"><thead><tr><th>source</th><th class="r">files</th><th class="r">audio</th><th class="r">clips</th><th class="r">clip audio</th><th>share of the clips</th><th class="r">skipped</th>${p.excluded_count ? '<th class="r">left out</th>' : ""}</tr></thead><tbody>
          ${sources.map(([g, c]) => `<tr><td>${esc(g)}</td><td class="r num">${c.files}</td><td class="r num">${hours(c.source_seconds)}</td><td class="r num">${c.clips}</td><td class="r num">${hours(c.seconds)}</td>
            <td><span class="sharebar"><span style="width:${p.clips ? (c.clips / p.clips) * 100 : 0}%"></span></span> <span class="num">${p.clips ? ((c.clips / p.clips) * 100).toFixed(1) : 0}%</span></td><td class="r num dim">${c.skipped || ""}</td>${p.excluded_count ? `<td class="r num dim">${c.excluded || ""}</td>` : ""}</tr>`).join("")}
        </tbody></table>
        ${p.duplicates.length ? `<details ${p.request_sent.skip_duplicates ? "" : "open"}><summary>${p.duplicates.reduce((s, d) => s + d.files, 0)} probable duplicates ${p.request_sent.skip_duplicates ? "left out" : "counted twice"}</summary>
          <div class="table-wrap"><table class="plan-table dup-table"><thead><tr><th>files in</th><th>same recordings as, ${p.request_sent.skip_duplicates ? "kept" : "kept if left out"}</th><th class="r">files</th><th>for example</th></tr></thead><tbody>
          ${p.duplicates.map((d) => `<tr><td>${esc(nameOf(d.folder))}</td><td>${esc(nameOf(d.kept_folder))}</td><td class="r num">${d.files}</td><td class="dim small">${d.examples.map((e) => `${esc(e.path)} ≈ ${esc(e.kept)}`).join("<br>")}</td></tr>`).join("")}
          </tbody></table></div></details>` : ""}
        ${p.skipped_count ? `<details><summary>${p.skipped_count} files skipped</summary><pre class="cmd">${p.skipped.map((s) => `${esc(s.path)}: ${esc(s.reason)}`).join("\n")}${p.skipped_count > p.skipped.length ? `\n… and ${p.skipped_count - p.skipped.length} more` : ""}</pre></details>` : ""}
        <details><summary>first clip names</summary><pre class="cmd">${p.examples.map(esc).join("\n")}</pre></details>
        <details><summary>the commands it runs</summary><pre class="cmd">${p.commands.map(esc).join("\n\n")}</pre></details>
        ${p.key ? `<div class="form-actions"><button class="solid big" id="corpus-launch">${p.prepare ? "Build and prepare" : "Build"}</button><span class="form-msg" id="launch-msg"></span></div>` : ""}
      </div>`;
      const launch = document.getElementById("corpus-launch");
      if (launch) launch.addEventListener("click", async () => {
        const msg = document.getElementById("launch-msg");
        launch.disabled = true;
        msg.textContent = "starting…";
        try {
          const job = await NR.postJSON("/api/corpus/launch", { request: p.request_sent, key: p.key });
          location.hash = `#/corpus/job/${encodeURIComponent(job.id)}`;
        } catch (err) { msg.textContent = err.message; msg.classList.add("err"); launch.disabled = false; }
      });
    }
  }

  // ---------- a job ----------
  async function viewJob(id) {
    NR.setNav("corpus");
    const url = `/api/corpus/jobs/${encodeURIComponent(id)}`;
    let job;
    try { job = await NR.getJSON(url); } catch (err) {
      NR.view.innerHTML = `<div class="empty">${esc(err.message)}<br><a href="#/corpus">Corpus</a></div>`;
      return;
    }
    NR.view.innerHTML = `<div class="run-page">
      <div class="crumb"><a href="#/corpus">Corpus</a> / job ${esc(id)}</div>
      <h1 class="title-xl" id="job-title"></h1>
      <div class="run-top"><div id="job-stats"></div><div class="run-actions" id="job-links"></div></div>
      <div class="section"><h3>Log <span class="dim">${esc(job.log_path)}</span></h3><pre class="log" id="job-log"></pre></div>
      <div class="section"><details><summary>the commands</summary><pre class="cmd">${job.commands.map(esc).join("\n\n")}</pre></details>
        <p class="dim store-line">It runs detached: closing the page or stopping the GUI does not stop it.</p></div>
    </div>`;
    const draw = (j) => {
      document.getElementById("job-title").textContent = j.title;
      document.getElementById("job-stats").innerHTML = `<div>${stateHtml(j)}${j.exit_code != null && j.exit_code !== 0 ? ` <span class="dim">exit ${j.exit_code}</span>` : ""}</div>
        ${j.progress ? `<div class="progress-label num">${esc(j.progress)}</div>` : ""}
        <p class="dim">started ${NR.ago(j.created_at)}</p>`;
      const links = [];
      if (j.workspace && (j.workspace_ready || j.kind === "rebalance")) links.push(`<a class="solid" href="${wsUrl(j.workspace)}">${j.kind === "rebalance" ? "back to" : "open"} ${esc(j.workspace)}</a>`);
      if (j.corpus && j.state === "completed" && !j.workspace) links.push(`<span class="dim">clips in ${esc(j.corpus)}/clips</span>`);
      document.getElementById("job-links").innerHTML = links.join("");
      const log = document.getElementById("job-log");
      const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
      log.textContent = j.log.join("\n") || "(no output yet)";
      if (stick) log.scrollTop = log.scrollHeight;
      if (j.active) later(2000, async () => draw(await NR.getJSON(url)));
    };
    draw(job);
  }

  NR.views.corpus = (args) => {
    if (args[0] === "new") return viewNew();
    if (args[0] === "ws" && args[1]) return viewWorkspace(args[1]);
    if (args[0] === "job" && args[1]) return viewJob(args[1]);
    return viewList();
  };
})();
