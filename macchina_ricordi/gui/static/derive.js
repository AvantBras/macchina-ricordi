// Merge & perturb: derived checkpoints from `model merge|perturb`, run as detached jobs.
// Reached from the Atelier; a finished one hands its path to the training form as a warm start.
(function () {
  "use strict";
  const NR = window.NR;
  const { esc } = NR;
  const FORM_KEY = "nr.derive.form";
  const TRAIN_FORM_KEY = "nr.train.form";
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

  const stateHtml = (job) => `<span class="state state-${esc(job.state)}">${job.active ? '<span class="live-dot"></span>' : ""}${esc(job.state)}</span>`;
  const xpLink = (sig) => (NR.state.bySig.has(sig) ? `<a href="#/xp/${encodeURIComponent(sig)}">${esc(sig)}</a>` : `<span class="dim" title="no longer in .dora/xps">${esc(sig)}</span>`);
  const sourceHtml = (s) => (s.sig ? `${xpLink(s.sig)} <span class="dim">${esc(s.file.replace(/^checkpoint_?|\.th$/g, "") || "last")}</span>` : `<span class="dim" title="${esc(s.path)}">…/${esc(s.path.split("/").slice(-2).join("/"))}</span>`);
  const params = (c) => (c.kind === "merge" ? `${esc(c.method || "?")} α ${esc(c.alpha ?? "?")}` : `σ ${esc(c.sigma ?? "?")} · ${esc(c.layer_scope || "?")} · seed ${esc(c.seed ?? "?")}`);
  const slug = (s) => String(s).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "");
  const pct = (v) => (v == null ? "?" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`);

  // what the sidecar's `movement` means, in words; the scale is a rough guide, not ear-validated
  function reading(c) {
    const t = c.movement.total;
    if (c.kind === "perturb") {
      const how = t.rel < 0.03 ? "a nudge" : t.rel < 0.15 ? "a shake" : t.rel < 0.4 ? "damage" : "wrecked: the noise is as loud as the weights";
      return `moved ${pct(t.rel)} from its source: ${how}${c.sigma != null && t.rms ? ` · σ ${c.sigma} is ${pct(c.sigma / t.rms)} of a typical weight (${t.rms})` : ""}`;
    }
    const kin = t.cos >= 0.98 ? "near twins: the blend will sound like either" : t.cos >= 0.8 ? "relatives" : "strangers: expect mush before it heals";
    const a = c.alpha ?? 0.5;
    // |(1-a)A + aB| / |A|, for A and B of equal size: what a linear blend keeps of the weights' scale
    const keeps = Math.sqrt((1 - a) ** 2 + a ** 2 + 2 * a * (1 - a) * t.cos);
    return `A and B are ${pct(t.rel)} apart (cos ${t.cos.toFixed(3)}): ${kin}${c.method === "linear" && keeps < 0.97 ? ` · a linear blend at α ${a} keeps about ${pct(keeps)} of the weights' size, slerp keeps all of it` : ""}`;
  }

  // a bar per group, input to output: embeddings, each transformer layer, the heads
  function movementHtml(c, big) {
    const m = c.movement;
    if (!m || !m.total) return big ? '<p class="dim small">No movement recorded: it was made before the sidecars kept one.</p>' : "";
    const max = Math.max(...m.groups.map((g) => g.rel || 0), 1e-9);
    const label = (n) => n.replace(/^layer /, "").replace(/^linears$/, "heads").replace(/^out_norm$/, "norm");
    const bars = m.groups.map((g) => `<i class="${c.kind === "perturb" && g.rel >= 0.4 ? "hot" : ""}" style="--h:${((g.rel || 0) / max) * 100}%"
      title="${esc(g.name)}: ${pct(g.rel)} · cos ${g.cos} · weight rms ${g.rms}"></i>`).join("");
    return `<div class="moves${big ? " big" : ""}">${bars}</div>
      ${big ? `<div class="moves-axis">${m.groups.map((g) => `<span>${esc(label(g.name))}</span>`).join("")}</div>` : ""}
      <div class="${big ? "" : "dim "}small">${esc(reading(c))}</div>`;
  }

  // the training form reads its last state from storage: leave it a warm start from this file
  function warmStart(path) {
    const saved = NR.storage(() => JSON.parse(localStorage.getItem(TRAIN_FORM_KEY) || "{}"), {});
    NR.storage(() => localStorage.setItem(TRAIN_FORM_KEY, JSON.stringify({ ...saved, init_mode: "file", init_file: path })));
    location.hash = "#/train";
  }

  // ---------- the page ----------
  async function viewDerive(prefill) {
    NR.setNav("derive");
    NR.view.innerHTML = `<div class="loading">checkpoints…</div>`;
    await NR.loadXps();
    const data = await NR.getJSON("/api/derive");
    const saved = NR.storage(() => JSON.parse(localStorage.getItem(FORM_KEY) || "{}"), {});
    if (prefill) Object.assign(saved, prefill);
    const derivedByPath = new Map(data.checkpoints.filter((c) => c.exists).map((c) => [c.path, c]));
    // Merge/perturb here works on legacy .th checkpoints only (the derive server accepts .th paths): a portable
    // bundle's single snapshot is a directory, so it is not offered as a source.
    const withCkpts = NR.state.xps.filter((x) => x.checkpoint_count && x.kind !== "bundle");

    NR.view.innerHTML = `<div class="atelier derive">
      <div class="crumb"><a href="#/train">Atelier</a> / merge &amp; perturb</div>
      <p class="dim intro">Make a new starting point from trained weights. <b>Merge</b> blends two models of the same shape; <b>perturb</b> adds seeded noise to one, for a degrade-then-heal run. The result keeps only the weights (no optimizer state), lands in <code>artifacts/checkpoints/merged|perturbed/&lt;name&gt;/</code> with a sidecar recording how it was made, and is warm-started like any checkpoint.</p>
      ${data.runtime ? `<div class="note warn"><p><span class="k">blocked</span>${esc(data.runtime)}</p></div>` : ""}
      <div class="section"><h3>Nouvelle greffe <span class="dim">preview first — nothing is written until you press the button</span></h3>
        <form id="derive-form" class="ab-form" autocomplete="off" novalidate>
          <fieldset><legend>What</legend>
            <label class="radio"><input type="radio" name="kind" value="merge" ${saved.kind !== "perturb" ? "checked" : ""}>
              <span><b>merge two checkpoints</b> <span class="dim">interpolate every weight between A and B. Both must have the same shape (width, depth, codebooks).</span></span></label>
            <label class="radio"><input type="radio" name="kind" value="perturb" ${saved.kind === "perturb" ? "checked" : ""}>
              <span><b>perturb one checkpoint</b> <span class="dim">add Gaussian noise, reproducible from its seed, to all weights or some layers.</span></span></label>
          </fieldset>
          <fieldset class="grid-2"><legend>From <em>an XP and one of its checkpoints, or a merged or perturbed file</em></legend>
            ${["a", "b"].map((s) => `<div class="src-pick" data-src="${s}" ${s === "b" ? "data-merge" : ""}>
              <label><span>${s === "a" ? '<b data-merge>A</b><span data-perturb>source</span>' : "<b>B</b>"} <em>sig, label or file</em></span><input name="src_${s}" placeholder="type a name, label, sig or file"></label>
              <label><span>checkpoint</span><select name="ckpt_${s}"></select></label>
              <div class="src-shape dim small" id="shape-${s}"></div>
            </div>`).join("")}
          </fieldset>
          <fieldset class="grid-2" data-merge><legend>How</legend>
            <label><span>method <em>slerp follows the arc between the two weight vectors, keeping their norm</em></span><select name="method">
              <option value="linear">linear</option><option value="slerp">slerp</option></select></label>
            <label><span>alpha <em>0 = all A, 1 = all B</em></span>
              <span class="alpha-row"><input type="range" name="alpha_range" min="0" max="1" step="0.05"><input type="number" name="alpha" min="0" max="1" step="any"></span></label>
          </fieldset>
          <fieldset class="grid-2" data-perturb><legend>How</legend>
            <label><span>sigma <em>noise scale; 0.005–0.02 is gentle, 0.05 and up is damage</em></span><input type="number" name="sigma" min="0" max="1" step="any"></label>
            <label><span>layers</span><select name="layer_scope">
              ${data.layer_scopes.map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("")}<option value="contains">keys containing…</option></select></label>
            <label data-contains><span>key fragment <em>e.g. transformer.layers.0.</em></span><input name="contains" placeholder="layers.0."></label>
            <label><span>seed</span><input type="number" name="seed" min="0" step="1"></label>
          </fieldset>
          <fieldset class="grid-2"><legend>Into</legend>
            <label class="span-all"><span>name <em id="out-hint"></em></span><input name="name" maxlength="80"></label>
          </fieldset>
          <div class="form-actions"><button class="solid big" type="submit">Preview</button><span class="form-msg" id="derive-msg"></span></div>
        </form>
        <div id="derive-plan"></div>
      </div>
      <div class="section"><h3>Travaux <span class="dim">merge and perturb jobs</span></h3><div id="derive-jobs"></div></div>
      <div class="section"><h3>Poids dérivés <span class="dim">${data.checkpoints.length} merged and perturbed checkpoints</span></h3>
        <div class="table-wrap"><table class="corpus-table derived-table"><thead><tr><th>kind</th><th>name</th><th>from</th><th>how</th><th>shape</th><th>trained from it</th><th></th></tr></thead>
        <tbody>${data.checkpoints.map(derivedRow).join("")}</tbody></table></div>
        <p class="dim small">Read from the <code>merged_from.json</code> and <code>perturbed_from.json</code> sidecars under <code>artifacts/checkpoints</code>, <code>artifacts/sessions</code> and <code>merged_checkpoints</code>. A folder that holds several outputs keeps one sidecar, for the last of them.</p>
      </div>
    </div>`;

    function derivedRow(c) {
      const actions = c.exists ? `<a href="" data-warm="${esc(c.path)}">warm start</a> · <a href="" data-use="perturb" data-path="${esc(c.path)}">perturb</a> · <a href="" data-use="merge" data-path="${esc(c.path)}">merge</a>` : '<span class="dim">file gone</span>';
      return `<tr>
        <td><span class="tag">${esc(c.kind)}</span></td>
        <td title="${esc(c.path)}">${esc(c.name)}${c.unrecorded.length ? ` <span class="dim" title="${esc(c.unrecorded.join("\n"))}">+${c.unrecorded.length} unrecorded</span>` : ""}${c.warnings.length ? ` <span class="dim" title="${esc(c.warnings.join("\n"))}">⚠</span>` : ""}
          <div class="dim small">${NR.ago(c.created_at)}${c.size_mb != null ? ` · ${Math.round(c.size_mb)} MB` : ""}</div></td>
        <td>${c.sources.map(sourceHtml).join("<br>")}</td>
        <td><span class="nowrap">${params(c)}</span>${movementHtml(c)}</td>
        <td class="dim small shape">${esc(c.shape || "—")}</td>
        <td>${c.children.map(xpLink).join(" ") || '<span class="dim">—</span>'}</td>
        <td class="small">${actions}</td></tr>`;
    }

    const form = document.getElementById("derive-form");
    const f = form.elements;
    let preview = null;
    let nameTouched = Boolean(saved.name);
    const kind = () => form.querySelector("input[name=kind]:checked").value;
    const picked = { a: null, b: null };  // {path, label} of each source, once known

    function sync() {
      const merge = kind() === "merge";
      form.querySelectorAll("[data-merge]").forEach((x) => (x.hidden = !merge));
      form.querySelectorAll("[data-perturb]").forEach((x) => (x.hidden = merge));
      form.querySelectorAll("[data-contains]").forEach((x) => (x.hidden = f.layer_scope.value !== "contains"));
      document.getElementById("out-hint").textContent = `→ artifacts/checkpoints/${merge ? "merged" : "perturbed"}/${f.name.value.trim() || "…"}/checkpoint.th`;
    }

    function scope() {
      return f.layer_scope.value === "contains" ? `contains:${f.contains.value.trim()}` : f.layer_scope.value;
    }

    function suggestName() {
      if (nameTouched) return;
      const a = picked.a && picked.a.label, b = picked.b && picked.b.label;
      if (kind() === "merge") f.name.value = a && b ? slug(`${a}_${b}_${f.method.value}_a${+f.alpha.value}`).slice(0, 80) : "";
      else f.name.value = a ? slug(`${a}_perturb_s${+f.sigma.value}${scope() === "all" ? "" : `_${scope()}`}${+f.seed.value ? `_seed${+f.seed.value}` : ""}`).slice(0, 80) : "";
      sync();
    }

    // a source: an XP sig (then one of its checkpoints), or the path of a derived or any other .th file
    async function fillSource(s, wanted) {
      const value = f[`src_${s}`].value.trim();
      const select = f[`ckpt_${s}`];
      select.innerHTML = "";
      if (NR.state.bySig.has(value)) {
        const d = await NR.getJSON(`/api/xps/${encodeURIComponent(value)}`);
        if (d.summary && d.summary.kind === "bundle") {
          // A portable bundle's weights are a directory of safetensors, not a .th file: this page (and the derive
          // server behind it) handles legacy .th checkpoints only, so there is nothing usable to pick here.
          select.innerHTML = "";
          document.getElementById(`shape-${s}`).textContent =
            "portable bundles can't be merged or perturbed here (this page needs legacy .th checkpoints). " +
            "A bundle has one snapshot, so there's nothing to merge; to branch one, use `model perturb models/" +
            ((d.summary.name || "").replace(/[^A-Za-z0-9._-]/g, "_")) + "` from a terminal.";
          picked[s] = null;
          suggestName();
          return;
        }
        if (!d.checkpoints.length) {
          document.getElementById(`shape-${s}`).textContent = "no checkpoint of it is left on disk";
          picked[s] = null;
          suggestName();
          return;
        }
        select.innerHTML = d.checkpoints.slice().reverse().map((c) => {
          const path = `.dora/xps/${value}/${c.name}`;
          return `<option value="${esc(path)}" data-label="${esc(`${value}-${c.epoch ?? "last"}`)}" ${path === wanted ? "selected" : ""}>${esc(c.name)}${c.epoch ? ` (epoch ${c.epoch})` : ""} · ${Math.round(c.size_mb)} MB</option>`;
        }).join("");
      } else if (value.endsWith(".th")) {
        const derived = derivedByPath.get(value);
        const label = derived ? derived.name : value.split("/").pop().replace(/\.th$/, "");
        select.innerHTML = `<option value="${esc(value)}" data-label="${esc(label)}">${esc(derived ? `${derived.kind} ${derived.name}` : value.split("/").pop())}</option>`;
      }
      await describe(s);
    }

    async function describe(s) {
      const box = document.getElementById(`shape-${s}`);
      const option = f[`ckpt_${s}`].selectedOptions[0];
      picked[s] = null;
      box.textContent = "";
      if (!option) { suggestName(); return; }
      try {
        const d = await NR.getJSON(`/api/derive/checkpoint?path=${encodeURIComponent(option.value)}`);
        picked[s] = { path: d.path, label: option.dataset.label, shape: d.shape_label };
        const xp = d.sig && NR.state.bySig.get(d.sig);
        box.innerHTML = `${esc(d.shape_label || "shape unknown")} · ${Math.round(d.size_mb)} MB${xp && xp.label ? ` · ${esc(xp.label)}` : ""}`;
      } catch (err) { box.textContent = err.message; }
      suggestName();
    }

    const invalidate = () => {
      if (preview) { preview = null; document.getElementById("derive-plan").innerHTML = '<p class="dim">The form changed: preview again.</p>'; }
    };

    // restore, or start from what was asked for
    f.method.value = saved.method || "linear";
    f.alpha.value = f.alpha_range.value = saved.alpha ?? 0.5;
    f.sigma.value = saved.sigma ?? 0.005;
    f.seed.value = saved.seed ?? 0;
    if (saved.layer_scope && saved.layer_scope.startsWith("contains:")) { f.layer_scope.value = "contains"; f.contains.value = saved.layer_scope.slice(9); }
    else if (saved.layer_scope) f.layer_scope.value = saved.layer_scope;
    f.name.value = saved.name || "";
    for (const s of ["a", "b"]) {
      const path = saved[`path_${s}`];
      if (!path) continue;
      const sig = (path.match(/^\.dora\/xps\/([^/]+)\//) || [])[1];
      f[`src_${s}`].value = sig || path;
      fillSource(s, path);
    }
    sync();

    const files = [...derivedByPath.values()].map((c) => ({ value: c.path, title: `${c.kind} ${c.name}`, meta: c.path }));
    for (const s of ["a", "b"]) {
      NR.modelPicker(f[`src_${s}`], { models: withCkpts, extra: files });
      f[`src_${s}`].addEventListener("change", () => fillSource(s));
      f[`ckpt_${s}`].addEventListener("change", () => describe(s));
    }
    f.alpha_range.addEventListener("input", () => { f.alpha.value = f.alpha_range.value; suggestName(); });
    f.alpha.addEventListener("input", () => { f.alpha_range.value = f.alpha.value; suggestName(); });
    f.name.addEventListener("input", () => { nameTouched = f.name.value.trim() !== ""; });
    form.addEventListener("change", (ev) => { if (ev.target.name !== "name") suggestName(); sync(); invalidate(); });
    form.addEventListener("input", () => { sync(); invalidate(); });

    const request = () => ({
      kind: kind(),
      sources: (kind() === "merge" ? ["a", "b"] : ["a"]).map((s) => f[`ckpt_${s}`].value).filter(Boolean),
      name: f.name.value.trim(), method: f.method.value, alpha: +f.alpha.value,
      sigma: +f.sigma.value, layer_scope: scope(), seed: +f.seed.value,
    });

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const req = request();
      NR.storage(() => localStorage.setItem(FORM_KEY, JSON.stringify({
        kind: req.kind, path_a: f.ckpt_a.value, path_b: f.ckpt_b.value, method: req.method, alpha: req.alpha,
        sigma: req.sigma, layer_scope: req.layer_scope, seed: req.seed,
      })));
      const msg = document.getElementById("derive-msg");
      msg.classList.remove("err");
      msg.textContent = "checking…";
      try {
        preview = await NR.postJSON("/api/derive/preview", { request: req });
        preview.request_sent = req;
        msg.textContent = "";
        drawPreview(preview);
      } catch (err) { msg.textContent = err.message; msg.classList.add("err"); }
    });

    function drawPreview(p) {
      const merge = p.kind === "merge";
      const req = p.request_sent;
      document.getElementById("derive-plan").innerHTML = `<div class="plan">
        <div class="plan-head">
          <div><div class="dim">${merge ? "merge" : "perturb"}</div><div class="plan-sig">${merge ? `${esc(req.method)} α ${req.alpha}` : `σ ${req.sigma}`}</div></div>
          ${merge ? "" : `<div><div class="dim">layers · seed</div><div>${esc(req.layer_scope)} · ${req.seed}</div></div>`}
          <div><div class="dim">into</div><div>${esc(p.output)}</div></div>
          <div><div class="dim">memory</div><div>about ${p.memory_gib} GiB, on the CPU</div></div>
        </div>
        ${p.blockers.map((b) => `<div class="note warn"><p><span class="k">blocked</span>${esc(b)}</p></div>`).join("")}
        ${p.warnings.map((w) => `<div class="note"><p><span class="k">note</span>${esc(w)}</p></div>`).join("")}
        <table class="plan-table"><thead><tr><th></th><th>checkpoint</th><th>shape</th><th class="r">size</th></tr></thead><tbody>
          ${p.sources.map((s, i) => `<tr><td>${merge ? "AB"[i] : "source"}</td><td>${sourceHtml(s)} <span class="dim small">${esc(s.path)}</span></td><td class="dim">${esc(s.shape || "unknown")}</td><td class="r num">${Math.round(s.size_mb)} MB</td></tr>`).join("")}
        </tbody></table>
        <details><summary>the command it runs</summary><pre class="cmd">${esc(p.command)}</pre></details>
        ${p.key ? `<div class="form-actions"><button class="solid big" id="derive-launch">${merge ? "Merge" : "Perturb"}</button><span class="form-msg" id="launch-msg"></span></div>` : ""}
      </div>`;
      const launch = document.getElementById("derive-launch");
      if (launch) launch.addEventListener("click", async () => {
        const msg = document.getElementById("launch-msg");
        launch.disabled = true;
        msg.textContent = "starting…";
        try {
          const job = await NR.postJSON("/api/derive/launch", { request: req, key: p.key });
          location.hash = `#/derive/job/${encodeURIComponent(job.id)}`;
        } catch (err) { msg.textContent = err.message; msg.classList.add("err"); launch.disabled = false; }
      });
    }

    const jobs = document.getElementById("derive-jobs");
    jobs.innerHTML = data.jobs.length ? `<div class="table-wrap"><table class="kv runs-table">${data.jobs.slice(0, 8).map((j) => `<tr data-href="#/derive/job/${encodeURIComponent(j.id)}">
      <td>${stateHtml(j)}</td><td>${esc(j.title)}</td><td class="dim">${NR.ago(j.created_at)}</td></tr>`).join("")}</table></div>` : '<p class="dim">None yet.</p>';

    NR.view.querySelector(".derived-table").addEventListener("click", (ev) => {
      if (ev.target.closest("a[data-warm], a[data-use]")) ev.preventDefault();
      const warm = ev.target.closest("[data-warm]");
      if (warm) return warmStart(warm.dataset.warm);
      const use = ev.target.closest("[data-use]");
      if (!use) return;
      form.querySelector(`input[name=kind][value=${use.dataset.use}]`).checked = true;
      f.src_a.value = use.dataset.path;
      nameTouched = false;
      fillSource("a");
      sync();
      invalidate();
      form.scrollIntoView({ behavior: "smooth" });
    });
  }

  // ---------- a job ----------
  async function viewJob(id) {
    NR.setNav("derive");
    const url = `/api/derive/jobs/${encodeURIComponent(id)}`;
    let job;
    try { job = await NR.getJSON(url); } catch (err) {
      NR.view.innerHTML = `<div class="empty">${esc(err.message)}<br><a href="#/derive">Merge &amp; perturb</a></div>`;
      return;
    }
    NR.view.innerHTML = `<div class="run-page">
      <div class="crumb"><a href="#/train">Atelier</a> / <a href="#/derive">merge &amp; perturb</a> / ${esc(id)}</div>
      <h1 class="title-xl">${esc(job.title)}</h1>
      <div class="run-top"><div id="job-stats"></div><div class="run-actions" id="job-links"></div></div>
      <div class="section" id="job-moves" hidden><h3>How far it moved <span class="dim">relative change per group, embeddings → layers → heads; hover a bar</span></h3><div id="job-moves-body"></div></div>
      <div class="section"><h3>Log <span class="dim">${esc(job.log_path)}</span></h3><pre class="log" id="job-log"></pre></div>
      <div class="section"><details><summary>the command</summary><pre class="cmd">${job.commands.map(esc).join("\n\n")}</pre></details>
        <p class="dim store-line">It runs detached: closing the page or stopping the GUI does not stop it. Loading and hashing the checkpoints takes most of the time.</p></div>
    </div>`;
    const draw = (j) => {
      document.getElementById("job-stats").innerHTML = `<div>${stateHtml(j)}${j.exit_code ? ` <span class="dim">exit ${j.exit_code}</span>` : ""}</div>
        <p class="dim">started ${NR.ago(j.created_at)}${j.output ? ` · ${esc(j.output)}` : ""}</p>`;
      const links = document.getElementById("job-links");
      links.innerHTML = j.ready ? `<button class="solid" id="job-warm">warm start a run from it</button>` : "";
      if (j.ready) {
        document.getElementById("job-warm").addEventListener("click", () => warmStart(j.output));
        NR.getJSON("/api/derive").then((d) => {
          const c = d.checkpoints.find((x) => x.path === j.output);
          if (!c) return;
          document.getElementById("job-moves-body").innerHTML = movementHtml(c, true);
          document.getElementById("job-moves").hidden = false;
        }).catch(() => {});
      }
      const log = document.getElementById("job-log");
      log.textContent = j.log.join("\n") || "(no output yet)";
      if (j.active) later(2000, async () => draw(await NR.getJSON(url)));
    };
    draw(job);
  }

  NR.views.derive = (args) => {
    if (args[0] === "job" && args[1]) return viewJob(args[1]);
    // #/derive/merge/<path> or #/derive/perturb/<path>: start from that checkpoint
    if ((args[0] === "merge" || args[0] === "perturb") && args[1]) return viewDerive({ kind: args[0], path_a: args.slice(1).join("/"), path_b: "", name: "" });
    return viewDerive();
  };
})();
