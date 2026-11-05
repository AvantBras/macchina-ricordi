// Réserve: what .dora/xps holds on disk, and the admin commands that trim it (clean-checkpoints,
// prune, archive-checkpoints). Every action is previewed as a table; a commit only goes through
// if what is on disk still matches the preview.
(function () {
  "use strict";
  const NR = window.NR;
  const { esc } = NR;
  const SEL_KEY = "nr.house.sel";
  const FORM_KEY = "nr.house.form";
  let poll = null;

  function stopPolling() {
    if (poll) { clearTimeout(poll); poll = null; }
  }
  window.addEventListener("hashchange", stopPolling);

  function size(bytes) {
    if (bytes == null) return "—";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let v = bytes, i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return i === 0 ? `${v} B` : `${v.toFixed(v >= 100 || i < 3 ? 0 : 1)} ${units[i]}`;
  }

  const loadSelection = () => new Set(NR.storage(() => JSON.parse(sessionStorage.getItem(SEL_KEY) || "[]"), []));
  const saveSelection = (sel) => NR.storage(() => sessionStorage.setItem(SEL_KEY, JSON.stringify([...sel])));

  function diskHtml(inv) {
    const fs = inv.filesystem, t = inv.totals;
    const pct = (v) => `${Math.max(0, (v / fs.total) * 100)}%`;
    const elsewhere = Math.max(0, fs.used - t.all);
    const parts = [
      ["latest", t.latest, "checkpoint.th"],
      ["epoch", t.epoch, "epoch checkpoints"],
      ["other", t.other, "other .th"],
      ["rest", t.rest, "samples, logs, history"],
      ["elsewhere", elsewhere, "the rest of the disk"],
    ];
    const share = t.all ? Math.round((t.latest / t.all) * 100) : 0;
    return `<div class="disk">
      <div class="disk-head">
        <div><span class="disk-big">${size(t.all)}</span> <span class="dim">in ${esc(inv.dora_dir)}/xps, ${inv.folders.length} folders</span></div>
        <div class="dim">${size(fs.free)} free of ${size(fs.total)} · ${Math.round((fs.used / fs.total) * 100)}% used</div>
      </div>
      <div class="disk-bar">${parts.map(([k, v, label]) => `<span class="seg-${k}" style="width:${pct(v)}" title="${esc(label)}: ${size(v)}"></span>`).join("")}</div>
      <div class="disk-legend">${parts.map(([k, v, label]) => `<span><i class="seg-${k}"></i>${esc(label)} <b class="num">${size(v)}</b></span>`).join("")}<span><i class="seg-free"></i>free <b class="num">${size(fs.free)}</b></span></div>
      ${share >= 50 ? `<p class="dim disk-hint">Final <code>checkpoint.th</code> files are ${share}% of it. Deleting whole XPs, or archiving them to the other disk first, is what frees real space; epoch cleanup only takes what is left between them.</p>` : ""}
    </div>`;
  }

  function epochChips(folder) {
    return folder.epochs.map((e) => {
      const tip = [`${e.name}, ${size(e.bytes)}`, e.best ? "best valid ce of the epoch checkpoints on disk" : "", e.refs ? `referenced by ${e.ref}${e.refs > 1 ? ` and ${e.refs - 1} more` : ""}` : ""].filter(Boolean).join("\n");
      return `<span class="ep${e.refs ? " ref" : ""}${e.best ? " best" : ""}" title="${esc(tip)}">${e.epoch}${e.best ? "★" : ""}${e.refs ? "↗" : ""}</span>`;
    }).join("");
  }

  function folderTitle(folder) {
    if (!folder.is_xp) return folder.bytes < 4096 ? "empty folder" : "not an XP";
    const xp = NR.state.bySig.get(folder.sig);
    return xp ? NR.title(xp) : folder.sig;
  }

  const COLUMNS = [
    ["", null],
    ["", null],
    ["label", (f) => folderTitle(f).toLowerCase()],
    ["sig", (f) => f.sig],
    ["epochs", (f) => (NR.state.bySig.get(f.sig) || {}).epochs_done || 0, "r"],
    ["best ce", (f) => { const x = NR.state.bySig.get(f.sig); return x && x.best_valid_ce != null ? x.best_valid_ce : 1e9; }, "r"],
    ["size", (f) => f.bytes, "r"],
    ["checkpoint.th", (f) => f.latest_bytes, "r"],
    ["epoch checkpoints", (f) => f.epoch_bytes],
    ["other .th", (f) => f.other_bytes, "r"],
    ["refs", (f) => f.refs, "r"],
    ["last write", (f) => f.last_write || 0, "r"],
  ];

  async function viewHouse() {
    NR.setNav("house");
    stopPolling();
    NR.view.innerHTML = `<div class="loading">taking stock of .dora/xps…</div>`;
    const [inv] = await Promise.all([NR.getJSON("/api/house"), NR.loadXps()]);
    const sel = loadSelection();
    const known = new Set(inv.folders.map((f) => f.sig));
    for (const sig of [...sel]) if (!known.has(sig)) sel.delete(sig);
    const saved = NR.storage(() => JSON.parse(localStorage.getItem(FORM_KEY) || "{}"), {});
    const ui = { q: "", sort: saved.sort || "size", dir: saved.dir || -1, tab: saved.tab || "checkpoints" };
    const maxBytes = Math.max(1, ...inv.folders.map((f) => f.bytes));
    const bySig = new Map(inv.folders.map((f) => [f.sig, f]));
    let plan = null;

    NR.view.innerHTML = `<div class="house">
      ${diskHtml(inv)}
      <div class="section"><h3>Inventaire <span class="dim">tick folders to act on them · scanned in ${inv.scan_seconds} s</span></h3>
        <div class="toolbar">
          <input type="search" id="house-q" placeholder="filter by label, sig, profile, workspace" autocomplete="off">
          <div class="seg"><span>select</span>
            <button data-select="shown">all shown</button>
            <button data-select="unreferenced" title="nothing outside the folder mentions it">unreferenced</button>
            <button data-select="nockpt" title="an XP without checkpoint.th">no checkpoint</button>
            <button data-select="empty" title="folders that are not XPs">empty folders</button>
            <button data-select="none">none</button>
          </div>
          <span class="count" id="house-count"></span>
        </div>
        <div class="table-wrap"><table class="house-table"><thead><tr>${COLUMNS.map(([name, , cls], i) => `<th data-col="${i}" class="${cls || ""}">${name}</th>`).join("")}</tr></thead><tbody id="house-rows"></tbody></table></div>
        <p class="dim house-legend"><span class="ep best">20★</span> best valid ce of the epoch checkpoints on disk · <span class="ep ref">20↗</span> something outside the XP refers to it (hover for what) · <b>refs</b>: files outside the folder that mention the XP</p>
      </div>
      <div class="section"><h3>Ranger <span class="dim">every action is previewed as a table; nothing changes until you commit</span></h3>
        <div class="seg house-tabs" id="house-tabs">
          <button data-tab="checkpoints">Epoch checkpoints</button><button data-tab="xps">Whole XPs</button><button data-tab="archive">Archive</button>
        </div>
        <form id="house-form" class="ab-form house-form" autocomplete="off">
          <fieldset class="grid-2" data-tab="checkpoints"><legend>Epoch checkpoints <em>tidy clean</em></legend>
            <label class="radio"><input type="radio" name="scope_c" value="selected" checked><span>the ticked XPs <b class="sel-n"></b></span></label>
            <label class="radio"><input type="radio" name="scope_c" value="all"><span>every XP</span></label>
            <label><span>keep the last <em>per XP, by epoch</em></span><input type="number" name="keep_last" min="0" max="100" value="${esc(saved.keep_last ?? 1)}"></label>
            <label class="check"><input type="checkbox" name="keep_best" ${saved.keep_best === false ? "" : "checked"}> also keep each XP's best one <em>(★)</em></label>
            <p class="span-all dim">Deletes <code>checkpoint_N.th</code> files. <code>checkpoint.th</code> is never touched, and a checkpoint something refers to (↗: a warm start, a merge, a generation run, a sweep, a session, a note) is always kept.</p>
          </fieldset>
          <fieldset data-tab="xps"><legend>Whole XPs <em>tidy prune, limited to the ticked folders</em></legend>
            <p class="dim">Deletes the ticked folders entirely: checkpoints, samples, history. Generation runs and sessions made with them stay, but can no longer be replayed from, or continued. XPs that something refers to are kept unless you tick the box.</p>
            <p>ticked: <b class="sel-n"></b></p>
            <label class="check"><input type="checkbox" name="include_referenced"> also delete XPs that something refers to</label>
          </fieldset>
          <fieldset class="grid-2" data-tab="archive"><legend>Archive <em>tidy archive</em></legend>
            <label class="span-all"><span>destination <em>on the other disk, mounted</em></span><input name="destination" placeholder="/media/…/macchina_ricordi-checkpoints"></label>
            <label><span>checkpoints</span><select name="checkpoint_mode">
              <option value="latest">checkpoint.th</option><option value="epochs">checkpoint_N.th</option><option value="all">both</option></select></label>
            <label><span>older than <em>optional: 1h, 7d, 2w</em></span><input name="min_age" placeholder="any age"></label>
            <label class="radio"><input type="radio" name="scope_a" value="selected" checked><span>the ticked XPs <b class="sel-n"></b></span></label>
            <label class="radio"><input type="radio" name="scope_a" value="all"><span>every XP</span></label>
            <p class="span-all dim">Copies each checkpoint with its metadata and history into <code>&lt;destination&gt;/&lt;sig&gt;/versions/…</code>, verified by SHA-256. It never deletes or changes anything here, skips what is already archived, and refuses a destination on this disk, so an unmounted drive cannot quietly fill the system disk.</p>
          </fieldset>
          <div class="form-actions"><button class="solid big" type="submit">Preview</button><span class="form-msg" id="house-msg"></span></div>
        </form>
        <div id="house-plan"></div>
        <div id="house-job"></div>
      </div>
    </div>`;

    const form = document.getElementById("house-form");
    const f = form.elements;
    f.destination.value = saved.destination || inv.archive_default || "";
    if (saved.checkpoint_mode) f.checkpoint_mode.value = saved.checkpoint_mode;
    if (saved.min_age) f.min_age.value = saved.min_age;

    const matches = (folder) => {
      if (!ui.q) return true;
      const xp = NR.state.bySig.get(folder.sig) || {};
      return [folder.sig, folderTitle(folder), xp.profile, xp.workspace, xp.label].join(" ").toLowerCase().includes(ui.q);
    };
    const shown = () => {
      const col = COLUMNS.find(([n]) => n === ui.sort) || COLUMNS[6];
      const key = col[1] || (() => 0);
      return inv.folders.filter(matches).sort((a, b) => {
        const va = key(a), vb = key(b);
        return (va < vb ? -1 : va > vb ? 1 : 0) * ui.dir;
      });
    };
    const selectedBytes = () => [...sel].reduce((sum, sig) => sum + ((bySig.get(sig) || {}).bytes || 0), 0);

    const drawCount = () => {
      const n = sel.size;
      document.getElementById("house-count").textContent = `${n} ticked${n ? ` · ${size(selectedBytes())}` : ""} · ${shown().length} / ${inv.folders.length} folders`;
      document.querySelectorAll(".sel-n").forEach((el) => (el.textContent = n ? `(${n}, ${size(selectedBytes())})` : "(none ticked)"));
    };

    const drawRows = () => {
      document.querySelectorAll(".house-table th[data-col]").forEach((th) => {
        const name = COLUMNS[+th.dataset.col][0];
        th.classList.toggle("sorted", name === ui.sort);
        th.textContent = name + (name && name === ui.sort ? (ui.dir > 0 ? " ↑" : " ↓") : "");
      });
      document.getElementById("house-rows").innerHTML = shown().map((folder) => {
        const xp = NR.state.bySig.get(folder.sig);
        const palette = Cover.palette(folder.sig);
        const refTip = folder.refs ? `${folder.refs} file${folder.refs > 1 ? "s" : ""} mention it, e.g.\n${folder.ref_examples.join("\n")}${folder.children.length ? `\nchild XPs: ${folder.children.join(", ")}` : ""}` : "nothing outside the folder mentions it";
        return `<tr data-sig="${esc(folder.sig)}" class="${sel.has(folder.sig) ? "ticked" : ""}${folder.live ? " live" : ""}">
          <td><input type="checkbox" data-tick="${esc(folder.sig)}" ${sel.has(folder.sig) ? "checked" : ""} ${folder.live ? `disabled title="${esc(folder.live)}"` : ""} aria-label="tick ${esc(folder.sig)}"></td>
          <td><span class="chip" style="background:${palette.ground};--chip-ink:${palette.ink}"></span></td>
          <td>${folder.live ? `<span class="live-dot" title="${esc(folder.live)}"></span>` : ""}${xp && folder.is_xp ? `<a href="#/xp/${encodeURIComponent(folder.sig)}">${esc(folderTitle(folder))}</a>` : `<span class="dim">${esc(folderTitle(folder))}</span>`}${folder.keep_marker ? ` <span class="tag">keep</span>` : ""}</td>
          <td class="dim">${esc(folder.sig)}</td>
          <td class="r num">${xp ? xp.epochs_done || 0 : ""}</td>
          <td class="r num">${xp ? NR.fmt(xp.best_valid_ce) : ""}</td>
          <td class="r num"><span class="sizebar"><span style="width:${(folder.bytes / maxBytes) * 100}%"></span></span>${size(folder.bytes)}</td>
          <td class="r num">${folder.latest ? size(folder.latest.bytes) + (folder.latest.refs ? ' <span class="dim" title="referenced by name">↗</span>' : "") : '<span class="dim">—</span>'}</td>
          <td class="eps">${epochChips(folder)}${folder.epochs.length ? ` <span class="dim num">${size(folder.epoch_bytes)}</span>` : ""}</td>
          <td class="r num" title="${esc(folder.others.map((o) => `${o.name} ${size(o.bytes)}`).join("\n"))}">${folder.other_bytes ? size(folder.other_bytes) : ""}</td>
          <td class="r num" title="${esc(refTip)}">${folder.refs || '<span class="dim">0</span>'}</td>
          <td class="r dim">${NR.ago(folder.last_write)}</td>
        </tr>`;
      }).join("");
      drawCount();
    };

    const invalidate = () => {
      if (plan) {
        plan = null;
        document.getElementById("house-plan").innerHTML = `<p class="dim">Changed — preview again.</p>`;
      }
    };
    const setTicked = (sig, on) => {
      const folder = bySig.get(sig);
      if (!folder || folder.live) return;
      on ? sel.add(sig) : sel.delete(sig);
    };

    document.getElementById("house-q").addEventListener("input", (ev) => { ui.q = ev.target.value.trim().toLowerCase(); drawRows(); });
    document.querySelectorAll("[data-select]").forEach((btn) => btn.addEventListener("click", () => {
      const how = btn.dataset.select;
      if (how === "none") sel.clear();
      else {
        const pick = {
          shown: () => true,
          unreferenced: (x) => x.refs === 0,
          nockpt: (x) => x.is_xp && !x.latest,
          empty: (x) => !x.is_xp,
        }[how];
        shown().filter(pick).forEach((x) => setTicked(x.sig, true));
      }
      saveSelection(sel);
      invalidate();
      drawRows();
    }));
    document.querySelectorAll(".house-table th[data-col]").forEach((th) => th.addEventListener("click", () => {
      const name = COLUMNS[+th.dataset.col][0];
      if (!name) return;
      if (ui.sort === name) ui.dir *= -1;
      else { ui.sort = name; ui.dir = ["label", "sig", "best ce"].includes(name) ? 1 : -1; }
      remember();
      drawRows();
    }));
    document.getElementById("house-rows").addEventListener("click", (ev) => {
      if (ev.target.closest("a")) return;
      const row = ev.target.closest("tr[data-sig]");
      if (!row) return;
      const sig = row.dataset.sig;
      setTicked(sig, ev.target.tagName === "INPUT" ? ev.target.checked : !sel.has(sig));
      saveSelection(sel);
      invalidate();
      drawRows();
    });

    const drawTabs = () => {
      document.querySelectorAll("#house-tabs [data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === ui.tab));
      form.querySelectorAll("fieldset[data-tab]").forEach((fs) => (fs.hidden = fs.dataset.tab !== ui.tab));
    };
    document.querySelectorAll("#house-tabs [data-tab]").forEach((b) => b.addEventListener("click", () => {
      ui.tab = b.dataset.tab;
      remember();
      invalidate();
      drawTabs();
    }));

    const request = () => {
      const scope = ui.tab === "checkpoints" ? form.querySelector("[name=scope_c]:checked").value
        : ui.tab === "archive" ? form.querySelector("[name=scope_a]:checked").value : "selected";
      const sigs = scope === "all" ? [] : [...sel];
      return {
        action: ui.tab, sigs, keep_last: +f.keep_last.value, keep_best: f.keep_best.checked,
        include_referenced: f.include_referenced.checked, destination: f.destination.value.trim(),
        checkpoint_mode: f.checkpoint_mode.value, min_age: f.min_age.value.trim(),
        scope,
      };
    };
    const remember = () => NR.storage(() => localStorage.setItem(FORM_KEY, JSON.stringify({
      tab: ui.tab, sort: ui.sort, dir: ui.dir, keep_last: f.keep_last.value, keep_best: f.keep_best.checked,
      destination: f.destination.value.trim(), checkpoint_mode: f.checkpoint_mode.value, min_age: f.min_age.value.trim(),
    })));
    form.addEventListener("input", invalidate);
    form.addEventListener("change", invalidate);

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      remember();
      const msg = document.getElementById("house-msg");
      msg.classList.remove("err");
      const req = request();
      if (req.scope === "selected" && !req.sigs.length) {
        msg.textContent = "Tick some folders in the inventory first.";
        msg.classList.add("err");
        return;
      }
      msg.textContent = "planning…";
      document.getElementById("house-plan").innerHTML = "";
      try {
        plan = await NR.postJSON("/api/house/plan", { request: req });
        plan.request_sent = req;
        msg.textContent = "";
        drawPlan(plan);
      } catch (err) {
        msg.textContent = err.message;
        msg.classList.add("err");
      }
    });

    const label = (sig) => {
      const folder = bySig.get(sig);
      return `${esc(folder ? folderTitle(folder) : sig)} <span class="dim">${esc(sig)}</span>`;
    };
    const skippedHtml = (p) => (p.skipped || []).length
      ? `<div class="note"><p><span class="k">left out</span>${p.skipped.map((s) => `${esc(s.sig)}: ${esc(s.reason)}`).join("<br>")}</p></div>` : "";

    function drawPlan(p) {
      const el = document.getElementById("house-plan");
      let head = "", body = "", button = "";
      if (p.action === "checkpoints") {
        const xps = p.rows.filter((r) => r.remove.length).length;
        head = p.remove_count
          ? `Deletes <b>${p.remove_count}</b> epoch checkpoint${p.remove_count > 1 ? "s" : ""}, <b>${size(p.remove_bytes)}</b>, from ${xps} XP${xps > 1 ? "s" : ""}.`
          : "Nothing to delete: every epoch checkpoint in scope is kept.";
        const rows = p.rows.filter((r) => r.remove.length || r.keep.some((k) => !/^among the last \d+$/.test(k.reason)));
        body = rows.length ? `<table class="plan-table"><thead><tr><th>XP</th><th>deleted</th><th>kept</th></tr></thead><tbody>${rows.map((r) => `<tr>
            <td>${label(r.sig)}</td>
            <td>${r.remove.map((c) => `<span class="ep gone" title="${size(c.bytes)}">${c.epoch}</span>`).join("") || '<span class="dim">—</span>'}</td>
            <td>${r.keep.map((c) => `<span class="ep" title="${esc(c.reason)}">${c.epoch}</span> <span class="dim why">${esc(c.reason)}</span>`).join("<br>")}</td>
          </tr>`).join("")}</tbody></table>` : "";
        button = p.key ? `<button class="ghost danger" id="house-commit">Delete ${p.remove_count} file${p.remove_count > 1 ? "s" : ""}</button>` : "";
      } else if (p.action === "xps") {
        head = p.remove_count
          ? `Deletes <b>${p.remove_count}</b> XP folder${p.remove_count > 1 ? "s" : ""}, <b>${size(p.remove_bytes)}</b>.`
          : "Nothing to delete: every ticked XP is kept.";
        body = `<table class="plan-table"><thead><tr><th></th><th>XP</th><th class="r">size</th><th>why</th></tr></thead><tbody>${p.rows.map((r) => `<tr class="${r.action}">
            <td><span class="state ${r.action === "remove" ? "state-failed" : ""}">${r.action === "remove" ? "delete" : "keep"}</span></td>
            <td>${label(r.sig)}</td><td class="r num">${size(r.bytes)}</td>
            <td class="why">${[...r.reasons, ...r.notes].map(esc).join("<br>") || '<span class="dim">no keep rule applies</span>'}</td>
          </tr>`).join("")}</tbody></table>`;
        button = p.key ? `<button class="ghost danger" id="house-commit">Delete ${p.remove_count} folder${p.remove_count > 1 ? "s" : ""}</button>` : "";
      } else {
        if (p.destination_error) {
          el.innerHTML = `<div class="plan"><div class="note warn"><p><span class="k">destination</span>${esc(p.destination_error.replace(" or pass --allow-same-filesystem", ""))}</p>
            <p class="dim">Mount the archive disk, then preview again. The page never archives onto the disk it is freeing.</p></div></div>`;
          return;
        }
        const fits = p.free_bytes == null || p.free_bytes >= p.copy_bytes;
        head = p.copy_count
          ? `Copies <b>${p.copy_count}</b> checkpoint${p.copy_count > 1 ? "s" : ""}, <b>${size(p.copy_bytes)}</b>, to ${esc(p.destination)} <span class="dim">(${size(p.free_bytes)} free there)</span>.${p.already_count ? ` ${p.already_count} already archived.` : ""}`
          : `Nothing to copy${p.already_count ? `: ${p.already_count} already archived` : ""}.`;
        if (!fits) head += `<div class="note warn"><p><span class="k">space</span>That is more than the destination has free; the copy stops at the first checkpoint that does not fit.</p></div>`;
        const order = { archive: 0, already_archived: 1, skip: 2 };
        body = p.rows.length ? `<table class="plan-table"><thead><tr><th></th><th>XP</th><th>checkpoint</th><th class="r">size</th><th>why</th></tr></thead><tbody>${p.rows.slice().sort((a, b) => order[a.action] - order[b.action]).map((r) => `<tr>
            <td><span class="state ${r.action === "archive" ? "state-running" : ""}">${r.action === "archive" ? "copy" : r.action === "already_archived" ? "archived" : "skip"}</span></td>
            <td>${label(r.sig)}</td><td>${esc(r.checkpoint)}</td><td class="r num">${size(r.bytes)}</td><td class="why dim">${r.reasons.map(esc).join("; ")}</td>
          </tr>`).join("")}</tbody></table>` : "";
        button = p.key ? `<button class="solid big" id="house-commit">Copy ${p.copy_count} checkpoint${p.copy_count > 1 ? "s" : ""}</button>` : "";
      }
      el.innerHTML = `<div class="plan">
        <div class="plan-summary">${head}</div>
        ${skippedHtml(p)}
        ${body ? `<div class="plan-scroll">${body}</div>` : ""}
        ${button ? `<div class="form-actions">${button}<span class="form-msg" id="commit-msg"></span></div>` : ""}
      </div>`;
      const commit = document.getElementById("house-commit");
      if (commit) commit.addEventListener("click", () => doCommit(p));
    }

    async function doCommit(p) {
      const question = p.action === "checkpoints" ? `Delete ${p.remove_count} epoch checkpoint files (${size(p.remove_bytes)})? This cannot be undone.`
        : p.action === "xps" ? `Delete ${p.remove_count} XP folder${p.remove_count > 1 ? "s" : ""} (${size(p.remove_bytes)}) with everything in them? This cannot be undone.`
        : `Copy ${p.copy_count} checkpoints (${size(p.copy_bytes)}) to ${p.destination}? Keep the GUI running until it finishes.`;
      if (!confirm(question)) return;
      const msg = document.getElementById("commit-msg");
      const button = document.getElementById("house-commit");
      button.disabled = true;
      msg.textContent = p.action === "archive" ? "starting…" : "deleting…";
      try {
        const result = await NR.postJSON("/api/house/commit", { request: p.request_sent, key: p.key });
        plan = null;
        if (result.action === "archive") {
          document.getElementById("house-plan").innerHTML = "";
          watchJob(result.job);
          return;
        }
        if (result.action === "xps") for (const sig of result.removed) sel.delete(sig);
        saveSelection(sel);
        NR.invalidate();
        const done = result.action === "xps"
          ? `Deleted ${result.removed.length} folder${result.removed.length === 1 ? "" : "s"}, ${size(result.freed_bytes)} freed.${result.failed.length ? ` Failed: ${result.failed.join(", ")}.` : ""} Report: ${result.report}`
          : `Deleted ${result.removed_count} file${result.removed_count === 1 ? "" : "s"}, ${size(result.freed_bytes)} freed. Each XP touched has a checkpoint_cleanup_report.json.`;
        NR.storage(() => sessionStorage.setItem("nr.house.done", done));
        await viewHouse();
      } catch (err) {
        msg.textContent = err.message;
        msg.classList.add("err");
        button.disabled = false;
      }
    }

    function jobHtml(job) {
      const pct = job.bytes_total ? (job.bytes_copied / job.bytes_total) * 100 : job.state === "running" ? 0 : 100;
      const state = job.state === "running" ? "running" : job.state === "failed" || job.failed.length ? "failed" : "completed";
      return `<div class="plan job">
        <div><span class="state state-${state}">${job.state === "running" ? '<span class="live-dot"></span>' : ""}archive ${esc(job.state)}</span> <span class="dim">to ${esc(job.destination)} · started ${NR.ago(job.started_at)}</span></div>
        <div class="progress"><span style="width:${pct}%"></span></div>
        <div class="progress-label num">${job.copied} / ${job.to_copy} copied · ${size(job.bytes_copied)} / ${size(job.bytes_total)}${job.already ? ` · ${job.already} already there` : ""}</div>
        ${job.current ? `<p class="dim">copying ${esc(job.current)}…</p>` : ""}
        ${job.failed.length ? `<div class="note warn"><p><span class="k">failed</span>${job.failed.map((x) => `${esc(x.checkpoint)}: ${esc(x.reason)}`).join("<br>")}</p></div>` : ""}
        ${job.error ? `<p class="err">${esc(job.error)}</p>` : ""}
        ${job.report ? `<p class="dim store-line">report: ${esc(job.report)}</p>` : ""}
      </div>`;
    }
    function watchJob(job) {
      const el = document.getElementById("house-job");
      if (!el || !job) return;
      el.innerHTML = jobHtml(job);
      if (job.state !== "running") return;
      stopPolling();
      poll = setTimeout(async () => {
        try { watchJob((await NR.getJSON("/api/house/job")).job); } catch (_) { watchJob(job); }
      }, 2000);
    }

    drawTabs();
    drawRows();
    watchJob(inv.job);
    const done = NR.storage(() => sessionStorage.getItem("nr.house.done"));
    if (done) {
      NR.storage(() => sessionStorage.removeItem("nr.house.done"));
      document.getElementById("house-plan").innerHTML = `<div class="note"><p><span class="k">done</span>${esc(done)}</p></div>`;
    }
  }

  NR.views.house = () => viewHouse();
})();
