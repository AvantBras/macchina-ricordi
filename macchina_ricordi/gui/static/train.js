// Atelier: queue, watch and stop training. Every training is a job (artifacts/train_jobs/<id>/), the same
// as `train queue add` makes; a worker starts queued jobs one at a time, detached, so they outlive
// this page and the server. The page only ever previews first.
(function () {
  "use strict";
  const NR = window.NR;
  const { esc } = NR;
  const FORM_KEY = "nr.train.form";
  let poll = null;

  function stopPolling() {
    if (poll) { clearTimeout(poll); poll = null; }
  }
  window.addEventListener("hashchange", stopPolling);

  function every(ms, fn) {
    stopPolling();
    const tick = async () => {
      try { await fn(); } catch (_) { /* keep polling through blips */ }
      if (document.visibilityState !== "hidden" && location.hash.startsWith("#/train")) poll = setTimeout(tick, ms);
      else poll = setTimeout(tick, ms * 3);
    };
    poll = setTimeout(tick, ms);
  }

  function duration(seconds) {
    if (seconds == null || !isFinite(seconds)) return "—";
    const s = Math.round(seconds);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h} h ${String(m).padStart(2, "0")}` : m ? `${m} min` : `${s} s`;
  }

  function gpuHtml(g) {
    if (!g || !g.available) return `<div class="gpu dim">GPU: ${esc((g && g.error) || "nvidia-smi unavailable")}</div>`;
    const bar = (v, max) => `<span class="gbar"><span style="width:${Math.min(100, (v / max) * 100)}%"></span></span>`;
    return `<div class="gpu">
      <span><b>GPU</b></span>
      <span>util ${bar(g.utilization_pct || 0, 100)} ${g.utilization_pct ?? "—"}%</span>
      <span>memory ${bar(g.memory_used_mib || 0, 12288)} ${g.memory_used_mib != null ? (g.memory_used_mib / 1024).toFixed(1) : "—"} GiB</span>
      <span>${g.temperature_c ?? "—"} °C</span>
      <span>${g.power_w != null ? Math.round(g.power_w) + " W" : ""}</span>
      ${g.gpu_lost ? `<span class="err">GPU LOST</span>` : ""}
    </div>`;
  }

  function stateHtml(run) {
    const live = run.active ? '<span class="live-dot"></span>' : "";
    return `<span class="state state-${esc(run.state)}">${live}${esc(run.state)}</span>`;
  }

  function progressHtml(run) {
    const total = run.epochs || 0;
    const done = run.epochs_done || 0;
    const pct = total ? Math.min(100, (done / total) * 100) : 0;
    return `<div class="progress"><span style="width:${pct}%"></span></div>
      <div class="progress-label num">${done}${total ? ` / ${total}` : ""} epochs</div>`;
  }

  const jobHref = (run) => `#/train/job/${encodeURIComponent(run.id)}`;
  const origin = (run) => ({ cli: "started in a terminal", queue: "queued from a terminal", gui: "" })[run.source] ?? "";

  function runCard(run) {
    const xp = NR.state.bySig.get(run.xp_sig);
    return `<a class="run-card" href="${jobHref(run)}">
      ${xp ? `<div class="mini">${NR.coverHtml(xp, { subtitle: false })}</div>` : `<div class="mini"><div class="cover" style="background:var(--ink)"></div></div>`}
      <div class="run-card-body">
        <div>${stateHtml(run)} <span class="dim">${esc(origin(run))}</span></div>
        <div class="run-card-title">${esc(run.effective_label || run.xp_sig || "—")}</div>
        <div class="dim">${esc(run.xp_sig || "")} · ${esc(run.description)}</div>
        ${progressHtml(run)}
      </div>
    </a>`;
  }

  // ---------- atelier ----------
  async function viewAtelier() {
    NR.setNav("train");
    NR.view.innerHTML = `<div class="loading">atelier…</div>`;
    await NR.loadXps();
    const [options, status] = await Promise.all([NR.getJSON("/api/train/options"), NR.getJSON("/api/train/status")]);
    const saved = NR.storage(() => JSON.parse(localStorage.getItem(FORM_KEY) || "{}"), {});
    const profiles = options.profiles;
    const profileOf = (name) => profiles.find((p) => p.name === name) || profiles[0];
    function hasWorkspace() {
      return options.workspaces.length > 0;
    }
    // Without a remembered form, start from whatever was trained last.
    if (!saved.profile) {
      const latest = NR.state.xps.filter((x) => x.profile && x.workspace).sort((a, b) => (b.written_at || "").localeCompare(a.written_at || ""))[0];
      if (latest) { saved.profile = latest.profile; saved.workspace = latest.workspace; }
    }

    NR.view.innerHTML = `<div class="atelier">
      <div id="gpu-strip">${gpuHtml(status.gpu)}</div>
      <div class="section"><h3>En cours <span class="dim" id="active-count"></span></h3><div id="active-runs"></div></div>
      <div class="section"><h3>File d'attente <span class="dim" id="queue-note"></span></h3><div id="queued-runs"></div></div>
      <div class="section"><h3>Nouvel entraînement <span class="dim">preview first — nothing is queued until you press Add</span></h3>
        <form id="train-form" class="ab-form train-form" autocomplete="off">
          <fieldset class="grid-2"><legend>What</legend>
            <label><span>recipe</span><select name="profile">${profiles.map((p) => `<option value="${esc(p.name)}">${esc(p.name)}${hasWorkspace() ? "" : " — no prepared workspace"}</option>`).join("")}</select></label>
            <label><span>workspace <em id="ws-hint"></em></span><select name="workspace"></select></label>
            <label><span>weights <em>saved on the workspace's page</em></span><select name="weights"></select></label>
            <div class="profile-note" id="profile-note"></div>
          </fieldset>
          <fieldset class="grid-2"><legend>From</legend>
            <label><span>start</span><select name="init_mode">
              <option value="">from scratch</option><option value="xp">warm start from an XP checkpoint or portable bundle</option><option value="file">warm start from a file or a local bundle directory</option>
            </select></label>
            <label data-init="xp"><span>XP <em>sig or label</em></span><input name="init_xp" placeholder="type a name, label, dataset or sig"></label>
            <label data-init="xp"><span>checkpoint</span><select name="init_ckpt"></select></label>
            <label data-init="file"><span>path or bundle</span><input name="init_file" list="ckpt-list" placeholder="artifacts/checkpoints/….th or models/<portable bundle>"></label>
            <p class="dim small span-all">Merged and perturbed checkpoints are files too: <a href="#/derive">merge or perturb one</a>, then warm start from it.</p>
            <datalist id="ckpt-list">${options.loose_checkpoints.map((c) => `<option value="${esc(c)}">`).join("")}</datalist>
          </fieldset>
          <fieldset class="sampling"><legend>Sampling <em>how training crops are drawn from the workspace</em></legend>
            <label class="radio"><input type="radio" name="sampling" value="file" checked>
              <span><b>each file equally often</b> <span class="dim">every file is drawn as often as any other, whatever its length. What the recipes do.</span></span></label>
            <label class="radio"><input type="radio" name="sampling" value="duration">
              <span><b>in proportion to length</b> <span class="dim">every second of audio equally often: a 10-minute track is drawn five times as often as a 2-minute one.</span></span></label>
            <p class="sampling-note" id="sampling-note"></p>
          </fieldset>
          <fieldset class="grid-2"><legend>Crops <em>encoded fresh every step, or once into a token pool</em></legend>
            <label class="check span-all"><input type="checkbox" name="pool"> train from a token pool
              <em>(crops encoded once and reused, each up to 8 times for the nano and pico families, 4 above: about 8× faster. The pool is built first, or reused, under artifacts/token_pools/)</em></label>
            <label data-pool><span>pool seed <em>another seed draws other crops, e.g. to continue a model trained on a pool</em></span><input type="number" min="0" step="1" name="pool_seed" placeholder="0"></label>
          </fieldset>
          <fieldset class="grid-2"><legend>How</legend>
            <label><span>run label <em>blank = dataset fingerprint label</em></span><input name="run_label" placeholder="king_night_warm_40ep"></label>
            <label><span>epochs <em id="epochs-hint"></em></span><input type="number" min="1" name="epochs"></label>
            <label><span>segment duration, s <em id="seg-hint"></em></span><input type="number" min="1" step="1" name="segment_duration"></label>
            <label class="span-all"><span>more overrides <em>one key=value per line, appended after the recipe's</em>
              <button type="button" class="hint-btn" id="guide-open" title="Which keys can be overridden?" aria-label="Override guide">?</button></span>
              <textarea name="overrides" rows="3" placeholder="optim.lr=5e-5"></textarea></label>
            <label class="check"><input type="checkbox" name="gpu_watchdog" checked> GPU watchdog <em>(kills the run if the GPU is lost or overheats)</em></label>
            <label><span>watchdog temperature trip, °C <em>optional</em></span><input type="number" name="gpu_watchdog_temp_c" placeholder="none"></label>
            <label class="check"><input type="checkbox" name="allow_segment_padding"> allow segment padding <em>(segment longer than the clips)</em></label>
          </fieldset>
          <div class="form-actions"><button class="solid big" type="submit">Preview</button><span class="form-msg" id="train-msg"></span></div>
        </form>
        <div id="plan"></div>
        <dialog id="override-guide" class="guide">
          <div class="guide-head"><h3>Overrides</h3><input type="search" id="guide-filter" placeholder="filter…"><button type="button" class="ghost" id="guide-close">Close</button></div>
          <p class="dim">Dora/Hydra keys, one <code>key=value</code> per line. They come after the recipe's, so they win. Click a key to add it.
            Changing anything gives a new XP signature. <b>recipe</b> is the selected recipe's value; <b>base</b> is AudioCraft's when the recipe leaves it alone.</p>
          <div id="guide-body"></div>
        </dialog>
      </div>
      <div class="section"><h3>Historique <span class="dim">every training: this page, the terminal, agents</span></h3><div id="past-runs"></div></div>
    </div>`;

    const form = document.getElementById("train-form");
    const f = form.elements;
    let preview = null;

    const fillWorkspaces = () => {
      const profile = profileOf(f.profile.value);
      const matching = options.workspaces;
      const current = f.workspace.value || saved.workspace;
      f.workspace.innerHTML = matching.length
        ? matching.map((w) => `<option value="${esc(w.name)}" ${w.name === current ? "selected" : ""}>${esc(w.name)} · ${w.clips ?? "?"} clips${w.slice_length_s ? ` · ${w.slice_length_s.toFixed(0)} s` : ""}</option>`).join("")
        : `<option value="">no prepared workspace — run data prepare</option>`;
      document.getElementById("profile-note").innerHTML = `<details><summary>${esc(profile.name)}${profile.extends ? ` extends ${esc(profile.extends)}` : ""}: ${profile.overrides.length} overrides</summary>
        <pre>${profile.overrides.map(esc).join("\n")}</pre></details>`;
      document.getElementById("epochs-hint").textContent = profile.epochs ? `recipe: ${profile.epochs}` : "";
      f.epochs.placeholder = profile.epochs || "";
      f.segment_duration.placeholder = profile.segment_duration || "";
      describeWorkspace();
    };
    const fillWeights = () => {
      const w = options.workspaces.find((x) => x.name === f.workspace.value);
      const saved_weights = (w && w.weights) || [];
      const current = f.weights.value || saved.weights || "";
      f.weights.innerHTML = `<option value="">none: every clip equally often</option>` + saved_weights.map((x) =>
        `<option value="${esc(x.name)}" ${x.name === current ? "selected" : ""} ${x.stale ? "disabled" : ""}>${esc(x.name)} · ${esc(x.mode)} a=${x.alpha}${x.stale ? " (stale)" : ""}</option>`).join("");
    };
    const describeWorkspace = () => {
      fillWeights();
      const w = options.workspaces.find((x) => x.name === f.workspace.value);
      const profile = profileOf(f.profile.value);
      const hint = document.getElementById("ws-hint");
      hint.textContent = w && w.last_run ? `last: ${w.last_run.effective_label || w.last_run.xp_sig} (${w.last_run.status})` : "";
      const seg = document.getElementById("seg-hint");
      seg.textContent = w && w.slice_length_s ? `clips are ${w.slice_length_s.toFixed(1)} s${profile.segment_duration ? ` · recipe: ${profile.segment_duration}` : ""}` : "";
      describeSampling();
    };
    const describeSampling = () => {
      const w = options.workspaces.find((x) => x.name === f.workspace.value);
      const profile = profileOf(f.profile.value);
      const note = document.getElementById("sampling-note");
      if (!w || !w.min_s) { note.textContent = ""; return; }
      const span = w.max_s / Math.max(w.min_s, 0.1);
      const secs = (v) => v >= 120 ? `${(v / 60).toFixed(v >= 600 ? 0 : 1)} min` : `${v.toFixed(0)} s`;
      let text = `${w.clips} files, ${secs(w.min_s)} to ${secs(w.max_s)} (median ${secs(w.median_s)}). `;
      text += span < 1.5
        ? "They are all about the same length, so both choices draw nearly the same crops."
        : `The longest is ${span.toFixed(span < 10 ? 1 : 0)}× the shortest: drawn per file, each second of the short ones comes up ${span.toFixed(span < 10 ? 1 : 0)}× as often as each second of the long ones.`;
      text += " Only the training split changes; validation keeps drawing every file equally, so valid ce stays comparable across runs.";
      const rb = Object.values(w.rebalance || {})[0];
      note.innerHTML = esc(text) + (rb ? `<br><b>Cluster weights are active</b> in this workspace (data rebalance, ${esc(rb.mode)} mode, k=${esc(rb.k)}, alpha=${esc(rb.alpha)}): they multiply into either choice. Clear them with <code>data rebalance --reset</code>.` : "");
    };
    const syncInit = () => {
      form.querySelectorAll("[data-init]").forEach((el) => (el.hidden = el.dataset.init !== f.init_mode.value));
    };
    const loadCheckpoints = async () => {
      const sig = f.init_xp.value.trim();
      f.init_ckpt.innerHTML = "";
      if (!NR.state.bySig.has(sig)) return;
      const d = await NR.getJSON(`/api/xps/${encodeURIComponent(sig)}`);
      const bundle = d.summary && d.summary.kind === "bundle";
      // A bundle's model page carries a single selected snapshot; its warm-start handle is the bundle directory
      // (models/<name>), not a fabricated .dora/xps path or an @selector on the safetensors file.
      f.init_ckpt.innerHTML = d.checkpoints.slice().reverse().map((c) => {
        const path = bundle ? d.summary.folder : `.dora/xps/${sig}/${c.name}`;
        return `<option value="${esc(path)}" ${c.name === saved.init_ckpt_name ? "selected" : ""}>${esc(bundle ? "portable bundle — single selected snapshot" : c.name)}${c.epoch ? ` (epoch ${c.epoch})` : ""}</option>`;
      }).join("");
    };

    const guide = document.getElementById("override-guide");
    const drawGuide = () => {
      const profile = profileOf(f.profile.value);
      const set = Object.fromEntries(profile.overrides.map((o) => o.split("=")).map(([k, ...v]) => [k, v.join("=")]));
      const q = document.getElementById("guide-filter").value.toLowerCase();
      const groups = new Map();
      for (const item of options.override_guide) {
        if (q && !`${item.key} ${item.what} ${item.group}`.toLowerCase().includes(q)) continue;
        if (!groups.has(item.group)) groups.set(item.group, []);
        groups.get(item.group).push(item);
      }
      document.getElementById("guide-body").innerHTML = [...groups].map(([group, items]) => `<h4>${esc(group)}</h4>
        <table class="guide-table">${items.map((item) => `<tr>
          <td><button type="button" class="guide-key" data-key="${esc(item.key)}">${esc(item.key)}</button></td>
          <td class="num">${item.key in set ? `<b>${esc(set[item.key])}</b> <span class="dim">recipe</span>` : `${esc(item.default)} <span class="dim">base</span>`}</td>
          <td>${esc(item.what)}</td></tr>`).join("")}</table>`).join("") || `<p class="dim">no key matches</p>`;
    };
    document.getElementById("guide-open").addEventListener("click", (ev) => {
      ev.preventDefault();
      drawGuide();
      guide.showModal();
    });
    document.getElementById("guide-close").addEventListener("click", () => guide.close());
    guide.addEventListener("click", (ev) => { if (ev.target === guide) guide.close(); });
    document.getElementById("guide-filter").addEventListener("input", drawGuide);
    document.getElementById("guide-body").addEventListener("click", (ev) => {
      const key = ev.target.closest("[data-key]");
      if (!key) return;
      const lines = f.overrides.value.split("\n").filter((line) => line.trim() && !line.startsWith(`${key.dataset.key}=`));
      lines.push(`${key.dataset.key}=`);
      f.overrides.value = lines.join("\n");
      guide.close();
      f.overrides.focus();
      f.overrides.setSelectionRange(f.overrides.value.length, f.overrides.value.length);
      invalidate();
    });

    // restore the last form
    if (saved.profile && profiles.some((p) => p.name === saved.profile)) f.profile.value = saved.profile;
    for (const name of ["init_mode", "init_xp", "init_file", "run_label", "epochs", "segment_duration", "overrides", "gpu_watchdog_temp_c", "pool_seed"]) {
      if (saved[name] != null) f[name].value = saved[name];
    }
    if (saved.gpu_watchdog === false) f.gpu_watchdog.checked = false;
    if (saved.pool) f.pool.checked = true;
    const syncPool = () => form.querySelectorAll("[data-pool]").forEach((el) => { el.hidden = !f.pool.checked; });
    f.pool.addEventListener("change", syncPool);
    syncPool();
    if (saved.sampling === "duration") form.querySelector("input[name=sampling][value=duration]").checked = true;
    fillWorkspaces();
    syncInit();
    if (f.init_mode.value === "xp") loadCheckpoints();
    NR.modelPicker(f.init_xp, { models: NR.state.xps.filter((x) => x.checkpoint_count) });

    const request = () => {
      const overrides = f.overrides.value.split("\n").map((s) => s.trim()).filter(Boolean);
      if (f.epochs.value) overrides.unshift(`optim.epochs=${+f.epochs.value}`);
      if (f.segment_duration.value) overrides.unshift(`dataset.segment_duration=${+f.segment_duration.value}`);
      const init = f.init_mode.value === "xp" ? f.init_ckpt.value : f.init_mode.value === "file" ? f.init_file.value.trim() : "";
      if (f.init_mode.value && !init) throw new Error("Select a valid warm-start snapshot or bundle first; no training has been queued.");
      return {
        profile: f.profile.value, workspace: f.workspace.value, overrides, run_label: f.run_label.value.trim(),
        init_from_checkpoint: init, gpu_watchdog: f.gpu_watchdog.checked, gpu_watchdog_temp_c: f.gpu_watchdog_temp_c.value,
        allow_segment_padding: f.allow_segment_padding.checked,
        sampling: form.querySelector("input[name=sampling]:checked").value,
        weights: f.weights.value,
        pool: f.pool.checked, pool_seed: f.pool.checked ? f.pool_seed.value : "",
      };
    };
    const remember = () => NR.storage(() => localStorage.setItem(FORM_KEY, JSON.stringify({
      profile: f.profile.value, workspace: f.workspace.value, init_mode: f.init_mode.value, init_xp: f.init_xp.value,
      init_ckpt_name: (f.init_ckpt.value || "").split("/").pop(), init_file: f.init_file.value, run_label: f.run_label.value,
      epochs: f.epochs.value, segment_duration: f.segment_duration.value, overrides: f.overrides.value,
      gpu_watchdog: f.gpu_watchdog.checked, gpu_watchdog_temp_c: f.gpu_watchdog_temp_c.value,
      sampling: form.querySelector("input[name=sampling]:checked").value, weights: f.weights.value,
      pool: f.pool.checked, pool_seed: f.pool_seed.value,
    })));

    const invalidate = () => {
      if (preview) {
        preview = null;
        document.getElementById("plan").innerHTML = `<p class="dim">The form changed — preview again before launching.</p>`;
      }
    };
    f.profile.addEventListener("change", fillWorkspaces);
    f.workspace.addEventListener("change", describeWorkspace);
    f.init_mode.addEventListener("change", syncInit);
    f.init_xp.addEventListener("change", loadCheckpoints);
    form.addEventListener("input", invalidate);
    form.addEventListener("change", invalidate);

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      remember();
      const msg = document.getElementById("train-msg");
      msg.classList.remove("err");
      msg.textContent = "planning (resolving the Dora signature takes a few seconds)…";
      document.getElementById("plan").innerHTML = "";
      try {
        const req = request();
        preview = await NR.postJSON("/api/train/preview", { request: req });
        preview.request_sent = req;
        msg.textContent = "";
        drawPlan(preview);
      } catch (err) {
        msg.textContent = err.message;
        msg.classList.add("err");
      }
    });

    function drawPlan(p) {
      const existing = p.existing_xp;
      const plan = document.getElementById("plan");
      plan.innerHTML = `<div class="plan">
        <div class="plan-head">
          <div><div class="dim">Dora signature</div><div class="plan-sig">${esc(p.xp_sig)}</div></div>
          <div><div class="dim">label</div><div>${esc(p.effective_label || "—")}</div></div>
          <div><div class="dim">epochs</div><div>${esc(p.epochs ?? "—")}</div></div>
          <div><div class="dim">segment</div><div>${p.segment_duration ?? "—"} s <span class="dim">(clips ${p.clip_slice_length_s ? p.clip_slice_length_s.toFixed(1) : "?"} s)</span></div></div>
          <div><div class="dim">sampling</div><div>${p.sampling === "duration" ? "by length" : "per file"}${p.rebalance ? ` <span class="dim">× cluster weights</span>` : ""}</div></div>
        </div>
        ${existing ? `<div class="note warn"><p><span class="k">exists</span>XP ${esc(p.xp_sig)} is already on disk${existing.label ? ` as <b>${esc(existing.label)}</b>` : ""}, ${existing.epochs_done} epochs done. Dora will <b>continue that XP</b> rather than start a new one. Change the label or an override for a fresh run.</p></div>` : `<div class="note"><p><span class="k">new</span>A new XP folder .dora/xps/${esc(p.xp_sig)} will be created.</p></div>`}
        ${p.pool ? `<div class="note"><p><span class="k">pool</span>${esc(p.pool.name)}: ${p.pool.crops.toLocaleString()} crops, each used ${p.pool.repeats} times over ${p.pool.epochs} epochs.
          ${p.pool.built ? "Already built, reused." : `Built first: about ${p.pool.build_hours.toFixed(1)} h of encoding and ${(p.pool.bytes / 1e9).toFixed(1)} GB on disk; the job shows <b>building</b> meanwhile.`}</p>
          <p class="dim">Resuming past ${p.pool.epochs} epochs needs a new pool.</p></div>` : ""}
        ${p.segment_warnings.map((w) => `<div class="note warn"><p><span class="k">segment</span>${esc(w)}</p></div>`).join("")}
        ${p.init_from_checkpoint ? `<div class="note"><p><span class="k">warm start</span>${esc(p.init_from_checkpoint.resolved_local_path || p.init_from_checkpoint.user_input)}</p>
          ${(p.init_from_checkpoint.warnings || []).map((w) => `<p class="dim">${esc(w)}</p>`).join("")}</div>` : ""}
        ${p.active_runs.length || p.queued_runs.length ? `<div class="note"><p><span class="k">queue</span>${p.active_runs.length ? `Training now: ${p.active_runs.map(esc).join(", ")}. ` : ""}${p.queued_runs.length ? `${p.queued_runs.length} job${p.queued_runs.length > 1 ? "s" : ""} queued before this one. ` : ""}It starts when they are done.</p>
          ${p.active_runs.length ? `<label class="check"><input type="checkbox" id="start-now"> start now, beside it <em>(two runs on one 12 GB GPU usually means both crawl or one dies)</em></label>` : ""}</div>` : ""}
        <details><summary>the same job from a terminal</summary><pre class="cmd">${esc(p.cli.join(" "))}</pre></details>
        <details><summary>the Dora command it runs</summary><pre class="cmd">${esc(p.command.join(" \\\n  "))}</pre></details>
        <div class="form-actions"><button class="solid big" id="launch">${p.active_runs.length || p.queued_runs.length ? "Add to the queue" : "Add and start"}</button><span class="form-msg" id="launch-msg"></span></div>
      </div>`;
      document.getElementById("launch").addEventListener("click", async () => {
        const now = document.getElementById("start-now");
        if (!confirm(`Queue GPU training of ${p.xp_sig}${p.effective_label ? ` (${p.effective_label})` : ""}? It runs on its own, even if you close this page or stop the GUI.`)) return;
        const msg = document.getElementById("launch-msg");
        msg.textContent = "queueing…";
        try {
          const run = await NR.postJSON("/api/train/launch", {
            request: p.request_sent, preview_key: p.key, start_now: !!(now && now.checked),
          });
          location.hash = jobHref(run);
        } catch (err) {
          msg.textContent = err.message;
          msg.classList.add("err");
        }
      });
    }

    const drawRuns = (st) => {
      document.getElementById("gpu-strip").innerHTML = gpuHtml(st.gpu);
      const active = st.runs.filter((r) => r.active);
      document.getElementById("active-count").textContent = active.length ? `${active.length} active` : "";
      document.getElementById("active-runs").innerHTML = active.length
        ? `<div class="run-cards">${active.map(runCard).join("")}</div>`
        : `<p class="dim">Nothing is training.</p>`;
      const queued = st.runs.filter((r) => r.state === "queued").sort((a, b) => a.queue_position - b.queue_position);
      document.getElementById("queue-note").textContent = queued.length
        ? (st.worker_pid ? "the worker starts each when the GPU is free" : "no worker is running: add a job here, or run train queue run")
        : "";
      document.getElementById("queued-runs").innerHTML = queued.length
        ? `<table class="kv runs-table">${queued.map((r) => `<tr data-href="${jobHref(r)}">
            <td class="num">#${r.queue_position}</td><td>${esc(r.effective_label || "—")}</td>
            <td class="dim">${esc(r.description)}</td><td class="dim">${esc(origin(r))}</td>
            <td><button class="ghost small" data-cancel="${esc(r.id)}">Cancel</button></td></tr>`).join("")}</table>`
        : `<p class="dim">Empty. Jobs added here, or with <code>train queue add</code>, wait here for the GPU.</p>`;
      const past = st.runs.filter((r) => !r.active && r.state !== "queued");
      document.getElementById("past-runs").innerHTML = past.length
        ? `<table class="kv runs-table">${past.map((r) => `<tr data-href="${jobHref(r)}">
            <td>${stateHtml(r)}</td><td>${esc(r.effective_label || "—")} <span class="dim">${esc(r.xp_sig || "")}</span></td>
            <td class="dim">${esc(r.description)}</td><td class="num">${r.epochs_done}${r.epochs ? ` / ${r.epochs}` : ""}</td><td class="dim">${NR.ago(r.started_at)}</td></tr>`).join("")}</table>`
        : `<p class="dim">No training jobs yet.</p>`;
    };
    document.getElementById("queued-runs").addEventListener("click", async (ev) => {
      const button = ev.target.closest("[data-cancel]");
      if (!button) return;
      ev.preventDefault();
      ev.stopPropagation();
      button.disabled = true;
      try {
        await NR.postJSON(`/api/train/jobs/${encodeURIComponent(button.dataset.cancel)}/stop`, {});
        drawRuns(await NR.getJSON("/api/train/status"));
      } catch (err) { alert(err.message); button.disabled = false; }
    });
    drawRuns(status);
    every(5000, async () => drawRuns(await NR.getJSON("/api/train/status")));
  }

  // ---------- a run ----------
  async function viewRun(jobId) {
    NR.setNav("train");
    await NR.loadXps();
    const url = `/api/train/jobs/${encodeURIComponent(jobId)}`;
    let run;
    try {
      run = await NR.getJSON(url);
    } catch (err) {
      NR.view.innerHTML = `<div class="empty">${esc(err.message)}<br><a href="#/train">Atelier</a></div>`;
      return;
    }
    let metric = "ce";
    NR.view.innerHTML = `<div class="run-page">
      <div class="crumb"><a href="#/train">Atelier</a> / ${esc(run.workspace || "")} / ${esc(run.id)}${origin(run) ? ` <span class="dim">· ${esc(origin(run))}</span>` : ""}</div>
      <h1 class="title-xl" id="run-title"></h1>
      <div class="byline" id="run-byline"></div>
      <div class="run-top">
        <div class="run-stats" id="run-stats"></div>
        <div class="run-actions" id="run-actions"></div>
      </div>
      <div id="run-gpu"></div>
      <div class="section"><h3>Curves <span class="seg" id="run-metric"></span></h3><div id="run-chart"></div></div>
      <div class="section"><h3>Log <span class="dim" id="log-name"></span></h3><pre class="log" id="run-log"></pre></div>
      <div class="section" id="run-meta"></div>
    </div>`;

    const draw = (r, gpu) => {
      const xp = NR.state.bySig.get(r.xp_sig);
      document.getElementById("run-title").textContent = (r.effective_label || r.xp_sig || "run").replace(/_/g, " ");
      document.getElementById("run-byline").innerHTML = `${esc(r.workspace)}${r.profile ? ` · ${esc(r.profile)}` : ""} · ${r.xp_exists ? `<a href="#/xp/${encodeURIComponent(r.xp_sig)}">${esc(r.xp_sig)}</a>` : esc(r.xp_sig || "")}`;
      const elapsedFrom = r.run_status && r.run_status.started_at ? r.run_status.started_at : r.started_at;
      const end = r.finished_at ? Date.parse(r.finished_at) : Date.now();
      document.getElementById("run-stats").innerHTML = `
        <div>${stateHtml(r)}${r.returncode != null && !r.active ? ` <span class="dim">exit ${esc(r.returncode)}</span>` : ""}</div>
        ${progressHtml(r)}
        <dl class="facts">
          <dt>elapsed</dt><dd>${elapsedFrom ? duration((end - Date.parse(elapsedFrom)) / 1000) : "—"}</dd>
          <dt>per epoch</dt><dd>${duration(r.seconds_per_epoch)}</dd>
          <dt>remaining</dt><dd>${r.active ? duration(r.eta_seconds) : "—"}</dd>
          ${r.queue_position ? `<dt>queue</dt><dd>#${r.queue_position}</dd>` : ""}
          ${r.stop && r.stop.requested_at ? `<dt>stop</dt><dd>requested ${NR.ago(r.stop.requested_at)}${r.stop.sigkill_sent ? " · had to SIGKILL" : ""}</dd>` : ""}
          ${r.error || (r.run_status && r.run_status.error_message) ? `<dt>error</dt><dd class="err">${esc(r.error || r.run_status.error_message)}</dd>` : ""}
        </dl>`;
      document.getElementById("run-actions").innerHTML = `
        ${xp ? `<a class="mini" href="#/xp/${encodeURIComponent(r.xp_sig)}">${NR.coverHtml(xp, { subtitle: false })}</a>` : ""}
        ${r.stoppable ? `<button class="ghost danger" id="stop-run">${r.state === "queued" ? "Cancel" : "Stop training"}</button>` : ""}`;
      const stopBtn = document.getElementById("stop-run");
      if (stopBtn) stopBtn.addEventListener("click", async () => {
        if (r.state !== "queued" && !confirm("Stop this training? Dora keeps the checkpoints it has written; progress since the last one is lost.")) return;
        stopBtn.disabled = true;
        try { draw(await NR.postJSON(`${url}/stop`), gpu); } catch (err) { alert(err.message); stopBtn.disabled = false; }
      });
      if (gpu) document.getElementById("run-gpu").innerHTML = gpuHtml(gpu);

      const metrics = [...new Set([...Object.keys(r.history.valid || {}), ...Object.keys(r.history.train || {})])].filter((k) => !["duration", "grad_scale"].includes(k));
      const seg = document.getElementById("run-metric");
      seg.innerHTML = metrics.map((k) => `<button class="${k === metric ? "on" : ""}" data-m="${esc(k)}">${esc(k)}</button>`).join("");
      seg.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => { metric = b.dataset.m; draw(r, gpu); }));
      const series = [];
      if (r.history.train && r.history.train[metric]) series.push({ name: `train ${metric}`, short: "train", values: r.history.train[metric], color: "var(--ink-3)", dash: true, width: 1.2 });
      if (r.history.valid && r.history.valid[metric]) series.push({ name: `valid ${metric}`, short: "valid", values: r.history.valid[metric], color: NR.chartColor(r.xp_sig || "x"), width: 2 });
      const chart = document.getElementById("run-chart");
      if (series.length) NR.drawChart(chart, series, { clip: /ppl|grad/.test(metric), sci: metric === "lr" });
      else chart.innerHTML = `<p class="dim">No epoch has finished yet${r.epochs ? ` — ${r.epochs} planned` : ""}.</p>`;

      const logEl = document.getElementById("run-log");
      const stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 8;
      const lines = r.log.length ? r.log : r.xp_log;
      document.getElementById("log-name").textContent = r.log.length ? r.log_path : r.xp_log.length ? "solver log" : "";
      logEl.textContent = lines.join("\n") || "(no output yet)";
      if (stick) logEl.scrollTop = logEl.scrollHeight;

      document.getElementById("run-meta").innerHTML = `<h3>Job</h3>
        <details><summary>the command it runs</summary><pre class="cmd">${esc(r.record.argv.join(" "))}</pre></details>
        <p class="dim store-line">record: ${esc(r.record.path)}/ (job.json, job_status.json, train_plan.json, train_run_status.json, train.log)</p>`;
    };
    draw(run, null);
    every(4000, async () => {
      const [r, st] = await Promise.all([NR.getJSON(url), NR.getJSON("/api/train/status")]);
      draw(r, st.gpu);
    });
  }

  NR.gpuHtml = gpuHtml; // the Scène launch form shows the same strip
  NR.views.train = (args) => (args[0] === "job" && args[1] ? viewRun(args[1]) : viewAtelier());
})();
