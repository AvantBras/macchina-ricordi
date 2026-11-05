// Experiment ledger: read, create, edit, and attach XPs. Every save rewrites
// research/training/experiment_ledger.toml, which is Git-tracked; Git is its history.
(function () {
  "use strict";
  const NR = window.NR;
  const { esc, state } = NR;
  const STATUSES = ["planned", "running", "paused", "completed", "abandoned"];

  async function loadExperiments(force) {
    if (!state.experiments || force) state.experiments = await NR.getJSON("/api/experiments");
    return state.experiments;
  }

  async function save(id, fields, create) {
    const result = await NR.postJSON("/api/ledger", { id, fields, create });
    NR.invalidate();
    return result;
  }

  function formHtml(e, opts) {
    const creating = !e;
    e = e || { experiment_id: "", title: "", status: "planned", hypothesis: "", verdict: "", next_action: "", tags: [], xp_sigs: opts.sigs || [] };
    return `<form class="ledger-form" data-create="${creating ? 1 : 0}" data-id="${esc(e.experiment_id)}">
      ${creating ? `<label><span>id</span><input name="id" required pattern="[a-z0-9][a-z0-9_\\-]*" placeholder="lowercase-with-dashes" autocomplete="off"></label>` : ""}
      <label><span>title</span><input name="title" value="${esc(e.title)}" ${creating ? "required" : ""}></label>
      <label><span>status</span><select name="status">${STATUSES.map((s) => `<option ${s === e.status ? "selected" : ""}>${s}</option>`).join("")}</select></label>
      <label><span>hypothesis</span><textarea name="hypothesis" rows="2">${esc(e.hypothesis)}</textarea></label>
      <label><span>verdict</span><textarea name="verdict" rows="3">${esc(e.verdict)}</textarea></label>
      <label><span>next action</span><textarea name="next_action" rows="2">${esc(e.next_action)}</textarea></label>
      <label><span>tags</span><input name="tags" value="${esc((e.tags || []).join(", "))}" placeholder="comma separated"></label>
      <label><span>xp sigs</span><textarea name="xp_sigs" rows="2" placeholder="one per line or comma separated">${esc((e.xp_sigs || []).join("\n"))}</textarea></label>
      <div class="form-actions">
        <button type="submit" class="solid">${creating ? "Create" : "Save"}</button>
        <button type="button" class="ghost" data-cancel>Cancel</button>
        <span class="form-msg"></span>
      </div>
    </form>`;
  }

  function readForm(form) {
    const get = (name) => form.elements[name] ? form.elements[name].value : undefined;
    const fields = {
      title: get("title"),
      status: get("status"),
      hypothesis: get("hypothesis"),
      verdict: get("verdict"),
      next_action: get("next_action"),
      tags: get("tags"),
      xp_sigs: get("xp_sigs"),
    };
    return { id: form.dataset.create === "1" ? get("id").trim() : form.dataset.id, fields };
  }

  function bindForm(form, onDone) {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const msg = form.querySelector(".form-msg");
      const { id, fields } = readForm(form);
      msg.textContent = "saving…";
      try {
        const result = await save(id, fields, form.dataset.create === "1");
        msg.textContent = result.changed_fields.length || result.created ? "saved" : "nothing changed";
        await onDone(id);
      } catch (err) {
        msg.textContent = err.message;
        msg.classList.add("err");
      }
    });
    form.querySelector("[data-cancel]").addEventListener("click", () => onDone(null));
  }

  function memberStrip(e) {
    const members = e.xp_sigs.map((s) => state.bySig.get(s) || { sig: s, missing: true });
    if (!members.length) return `<div class="dim">no XPs attached yet</div>`;
    return `<div class="family">${members.map((x, i) => `${i ? '<span class="arrow">·</span>' : ""}<a href="#/xp/${encodeURIComponent(x.sig)}" title="${esc(x.missing ? "not on disk" : NR.title(x))}">${x.missing ? `<div class="cover" style="background:var(--paper-2)"></div>` : NR.coverHtml(x, { subtitle: false })}<div>${esc(x.sig)}</div></a>`).join("")}</div>
      ${members.length > 1 ? `<a class="more" href="#/compare/${e.xp_sigs.map(encodeURIComponent).join(",")}">compare all ${members.length}</a>` : ""}`;
  }

  function entryHtml(e) {
    return `<section class="exp" id="exp-${esc(e.experiment_id)}">
      <div>
        <span class="status">${esc(e.status)}</span>
        <h2>${esc(e.title)}</h2>
        ${e.hypothesis ? `<p>${esc(e.hypothesis)}</p>` : ""}
        <div class="tags">${e.tags.map(esc).join(" · ")}${e.updated_at ? ` — updated ${esc(e.updated_at)}` : ""}</div>
        <div class="tags"><span class="dim">${esc(e.experiment_id)}</span> · <button class="linkish" data-edit="${esc(e.experiment_id)}">edit</button></div>
      </div>
      <div>
        ${e.verdict || e.next_action ? `<div class="note">${e.verdict ? `<p><span class="k">verdict</span>${esc(e.verdict)}</p>` : ""}${e.next_action ? `<p><span class="k">next</span>${esc(e.next_action)}</p>` : ""}</div>` : ""}
        ${memberStrip(e)}
      </div>
    </section>`;
  }

  async function viewExperiments(args) {
    NR.setNav("experiments");
    await NR.loadXps();
    const exps = await loadExperiments();
    const newFor = args[0] === "new" ? (args[1] || "").split(",").filter(Boolean) : null;
    NR.view.innerHTML = `<div class="toolbar"><button class="ghost" id="new-exp">+ New experiment</button>
        <span class="count">${exps.length} in research/training/experiment_ledger.toml</span></div>
      <div id="new-slot"></div>
      ${exps.length ? exps.map(entryHtml).join("") : `<div class="empty">no experiments yet — the ledger records the question, the verdict and the XPs that answer it</div>`}`;

    const openNew = (sigs) => {
      const slot = document.getElementById("new-slot");
      slot.innerHTML = `<section class="exp editing"><div><h2>New experiment</h2><p class="dim">Written to the ledger with today as updated_at. Commit it like any other file.</p></div>${formHtml(null, { sigs })}</section>`;
      bindForm(slot.querySelector("form"), async (id) => {
        if (!id) { slot.innerHTML = ""; if (newFor) location.hash = "#/experiments"; return; }
        await loadExperiments(true);
        location.hash = "#/experiments";
        viewExperiments([]);
      });
      slot.querySelector("input[name=id]").focus();
    };
    document.getElementById("new-exp").addEventListener("click", () => openNew([]));
    if (newFor) openNew(newFor);

    NR.view.querySelectorAll("[data-edit]").forEach((btn) => btn.addEventListener("click", () => {
      const e = exps.find((x) => x.experiment_id === btn.dataset.edit);
      const section = document.getElementById(`exp-${e.experiment_id}`);
      section.classList.add("editing");
      section.innerHTML = `<div><span class="status">${esc(e.status)}</span><h2>${esc(e.title)}</h2><p class="dim">${esc(e.experiment_id)}</p></div>${formHtml(e, {})}`;
      bindForm(section.querySelector("form"), async () => {
        await loadExperiments(true);
        viewExperiments([]);
      });
    }));
  }

  // On an XP page: attach it to an experiment, detach it, or start a new one from it.
  async function xpLedger(sig, detail) {
    const box = document.getElementById("ledger-controls");
    if (!box) return;
    const exps = await loadExperiments();
    const member = new Set((detail.experiments || []).map((e) => e.experiment_id));
    const others = exps.filter((e) => !member.has(e.experiment_id));
    box.innerHTML = `<div class="ledger-inline">
      ${others.length ? `<select id="attach-exp"><option value="">add to experiment…</option>${others.map((e) => `<option value="${esc(e.experiment_id)}">${esc(e.title)}</option>`).join("")}</select>` : ""}
      <a class="ghost" href="#/experiments/new/${encodeURIComponent(sig)}" style="text-decoration:none">+ new experiment from this XP</a>
      ${[...member].map((id) => `<button class="linkish" data-detach="${esc(id)}">remove from ${esc(id)}</button>`).join("")}
      <span class="form-msg"></span>
    </div>`;
    const msg = box.querySelector(".form-msg");
    const update = async (id, sigs) => {
      msg.textContent = "saving…";
      try {
        await save(id, { xp_sigs: sigs }, false);
        await loadExperiments(true);
        NR.route();
      } catch (err) {
        msg.textContent = err.message;
        msg.classList.add("err");
      }
    };
    const attach = box.querySelector("#attach-exp");
    if (attach) attach.addEventListener("change", () => {
      const e = exps.find((x) => x.experiment_id === attach.value);
      if (e) update(e.experiment_id, [...e.xp_sigs, sig]);
    });
    box.querySelectorAll("[data-detach]").forEach((btn) => btn.addEventListener("click", () => {
      const e = exps.find((x) => x.experiment_id === btn.dataset.detach);
      if (e) update(e.experiment_id, e.xp_sigs.filter((s) => s !== sig));
    }));
  }

  NR.views.experiments = viewExperiments;
  NR.hooks.xpLedger = xpLedger;
})();
