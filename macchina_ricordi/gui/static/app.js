(function () {
  "use strict";

  const view = document.getElementById("view");
  const state = { xps: null, bySig: new Map(), experiments: null, picks: loadPicks(), prefs: loadPrefs() };

  // ---------- utils ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmt = (v, d = 3) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—");
  const fileUrl = (p) => "/files/" + p.split("/").map(encodeURIComponent).join("/");

  function ago(ts) {
    if (!ts) return "—";
    const t = typeof ts === "number" ? ts * 1000 : Date.parse(ts);
    if (!isFinite(t)) return "—";
    const s = (Date.now() - t) / 1000;
    if (s < 90) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400 * 2) return `${Math.round(s / 3600)} h ago`;
    if (s < 86400 * 60) return `${Math.round(s / 86400)} days ago`;
    return new Date(t).toISOString().slice(0, 10);
  }

  function storage(fn, fallback) {
    try { return fn(); } catch (_) { return fallback; }
  }
  function loadPicks() {
    return new Set(storage(() => JSON.parse(sessionStorage.getItem("nr.picks") || "[]"), []));
  }
  function savePicks() {
    storage(() => sessionStorage.setItem("nr.picks", JSON.stringify([...state.picks])));
    renderTray();
  }
  function loadPrefs() {
    return Object.assign({ group: "profile", sort: "recent", q: "", tsort: "last_activity", tdir: -1 },
      storage(() => JSON.parse(localStorage.getItem("nr.prefs") || "{}"), {}));
  }
  function savePrefs() {
    storage(() => localStorage.setItem("nr.prefs", JSON.stringify(state.prefs)));
  }

  async function getJSON(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      let message = `${res.status} ${url}`;
      try { message = (await res.json()).error || message; } catch (_) { /* not JSON */ }
      throw new Error(message);
    }
    return res.json();
  }

  // Writes carry a header a cross-origin page cannot send without a preflight the server never approves.
  function postJSON(url, body) {
    return getJSON(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Macchina-Ricordi": "1" },
      body: JSON.stringify(body || {}),
    });
  }

  async function loadXps(force) {
    if (state.xps && !force) return state.xps;
    state.xps = await getJSON("/api/xps");
    state.bySig = new Map(state.xps.map((x) => [x.sig, x]));
    return state.xps;
  }

  function luminance(hex) {
    const c = hex.replace("#", "");
    const [r, g, b] = [0, 2, 4].map((i) => {
      const v = parseInt(c.slice(i, i + 2), 16) / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  function isDark() {
    const forced = document.documentElement.dataset.theme;
    if (forced) return forced === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  // The duotone colour that reads best on the page, for charts and chips.
  function chartColor(sig) {
    const { ground, ink } = Cover.palette(sig);
    const pageLum = isDark() ? 0.01 : 0.85;
    const contrast = (hex) => {
      const l = luminance(hex);
      return (Math.max(l, pageLum) + 0.05) / (Math.min(l, pageLum) + 0.05);
    };
    const best = contrast(ink) >= contrast(ground) ? ink : ground;
    return contrast(best) >= 2.2 ? best : isDark() ? "#ebe7dd" : "#161514";
  }

  function title(xp) {
    return xp.name || xp.label || xp.sig;
  }

  function coverHtml(xp, opts) {
    return `<div class="cover">${Cover.render(xp, opts)}</div>`;
  }

  // ---------- family / lineage ----------
  function rootOf(xp) {
    const seen = new Set();
    let cur = xp;
    while (cur && cur.parents && cur.parents.length && !seen.has(cur.sig)) {
      seen.add(cur.sig);
      const next = state.bySig.get(cur.parents[0]);
      if (!next) break;
      cur = next;
    }
    return cur;
  }

  function ancestry(xp) {
    const chain = [];
    const seen = new Set([xp.sig]);
    let cur = xp;
    while (cur && cur.parents && cur.parents.length) {
      const parents = cur.parents.map((p) => state.bySig.get(p) || { sig: p, missing: true });
      chain.unshift(parents);
      const next = parents.find((p) => !p.missing && !seen.has(p.sig));
      if (!next) break;
      seen.add(next.sig);
      cur = next;
    }
    return chain;
  }

  // ---------- nav / tray ----------
  function setNav(name) {
    document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === name));
  }

  function renderTray() {
    const tray = document.getElementById("compare-tray");
    const n = state.picks.size;
    tray.hidden = n === 0;
    if (!n) { tray.innerHTML = ""; return; }
    tray.innerHTML = `<span>${n} selected</span>
      <a class="solid" href="#/compare/${[...state.picks].map(encodeURIComponent).join(",")}">Compare</a>
      <button class="ghost" data-clear-picks>Clear</button>`;
  }

  function togglePick(sig) {
    if (state.picks.has(sig)) state.picks.delete(sig);
    else state.picks.add(sig);
    savePicks();
    document.querySelectorAll(`[data-pick="${CSS.escape(sig)}"]`).forEach((el) => {
      if (el.type === "checkbox") el.checked = state.picks.has(sig);
      else el.classList.toggle("on", state.picks.has(sig));
    });
  }

  // ---------- filtering / sorting ----------
  function matches(xp, q) {
    if (!q) return true;
    const hay = [xp.sig, xp.name, xp.label, xp.profile, xp.workspace, xp.dataset_name, xp.origin, ...(xp.experiments || [])]
      .filter(Boolean).join(" ").toLowerCase();
    return q.toLowerCase().split(/\s+/).every((t) => hay.includes(t));
  }

  // ---------- model picker: a searchable list under a text input, so nobody copies sigs by hand ----------
  // Fills the input with the sig (keeping any "@best"-style suffix typed after it) and fires input + change.
  function modelPicker(input, { models, extra = [], onPick } = {}) {
    const wrap = document.createElement("span");
    wrap.className = "mp";
    input.replaceWith(wrap);
    wrap.append(input);
    input.removeAttribute("list");
    input.setAttribute("autocomplete", "off");
    const list = document.createElement("div");
    list.className = "mp-list";
    list.hidden = true;
    wrap.append(list);
    let rows = [];
    let active = -1;

    const all = () => [
      ...(models || state.xps).slice().sort(SORTS.recent).map((x) => ({
        value: x.sig, xp: x, title: title(x).replace(/_/g, " "),
        meta: [x.sig, x.dataset_name || Cover.prettyWorkspace(x.workspace), x.epochs_done ? `${x.epochs_done} ep` : "", ago(x.last_activity)].filter(Boolean).join(" · "),
      })),
      ...extra.map((e) => ({ ...e, hay: `${e.value} ${e.title} ${e.meta || ""}`.toLowerCase() })),
    ];
    const draw = () => {
      list.innerHTML = rows.length
        ? rows.map((r, i) => `<div class="mp-row${i === active ? " on" : ""}" data-i="${i}"><b>${esc(r.title)}</b><span class="dim">${esc(r.meta || "")}</span></div>`).join("")
        : `<div class="mp-row dim">no model matches: Enter keeps what you typed</div>`;
      const on = list.querySelector(".mp-row.on");
      if (on) on.scrollIntoView({ block: "nearest" });
    };
    const open = () => {
      const q = input.value.split("@")[0].trim();
      rows = all().filter((r) => (r.xp ? matches(r.xp, q) : q.toLowerCase().split(/\s+/).every((t) => r.hay.includes(t)))).slice(0, 40);
      active = q && rows.length ? 0 : -1;
      list.hidden = false;
      draw();
    };
    const close = () => { list.hidden = true; active = -1; };
    const choose = (r) => {
      const at = input.value.indexOf("@");
      input.value = r.value + (r.xp && at >= 0 ? input.value.slice(at) : "");
      close();
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      if (onPick) onPick(input.value);
    };

    input.addEventListener("focus", open);
    input.addEventListener("input", (ev) => { if (ev.isTrusted) open(); });
    input.addEventListener("blur", close);
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !list.hidden) { ev.preventDefault(); close(); return; }
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        if (list.hidden) open();
        else if (rows.length) { active = (active + (ev.key === "ArrowDown" ? 1 : rows.length - 1)) % rows.length; draw(); }
        return;
      }
      if (ev.key === "Enter" && !list.hidden && active >= 0) { ev.preventDefault(); choose(rows[active]); }
    });
    // mousedown, not click: picking must happen before the input loses focus and closes the list
    list.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      const row = ev.target.closest("[data-i]");
      if (row) choose(rows[Number(row.dataset.i)]);
    });
  }

  // the Scène and Atelier forms restore from these keys, so a model page can preset them
  function useModel(where, sig) {
    const edit = (key, fn) => storage(() => localStorage.setItem(key, JSON.stringify(fn(JSON.parse(localStorage.getItem(key) || "{}")))));
    if (where === "stage") edit("nr.stage.launch", (s) => ({ ...s, models: [...(s.models || []).filter((m) => m !== sig), sig].slice(-9) }));
    if (where === "train") edit("nr.train.form", (s) => ({ ...s, init_mode: "xp", init_xp: sig }));
    location.hash = `#/${where}`;
  }

  const SORTS = {
    recent: (a, b) => (b.last_activity || 0) - (a.last_activity || 0),
    name: (a, b) => title(a).localeCompare(title(b)),
    best: (a, b) => (a.best_valid_ce ?? Infinity) - (b.best_valid_ce ?? Infinity),
    epochs: (a, b) => (b.epochs_done || 0) - (a.epochs_done || 0),
  };

  function groupKey(xp, mode) {
    if (mode === "profile") return xp.profile || "sans profil";
    if (mode === "workspace") return Cover.prettyWorkspace(xp.workspace) || "sans atelier";
    if (mode === "lineage") {
      const root = rootOf(xp);
      return root === xp && !(xp.parents || []).length && !state.xps.some((o) => (o.parents || []).includes(xp.sig))
        ? "orphelins"
        : `${title(root)} · ${root.sig}`;
    }
    return "tout";
  }

  function toolbar(extra) {
    const p = state.prefs;
    const seg = (key, label, options) => `<div class="seg"><span>${label}</span>${options
      .map(([v, l]) => `<button data-pref="${key}" data-val="${v}" class="${p[key] === v ? "on" : ""}">${l}</button>`).join("")}</div>`;
    return `<div class="toolbar">
      <input type="search" id="q" placeholder="filter — label, sig, profile, workspace…" value="${esc(p.q)}" autocomplete="off">
      ${extra.includes("group") ? seg("group", "shelves", [["profile", "profile"], ["workspace", "workspace"], ["lineage", "lineage"], ["none", "none"]]) : ""}
      ${extra.includes("sort") ? seg("sort", "order", [["recent", "recent"], ["best", "best ce"], ["epochs", "epochs"], ["name", "a–z"]]) : ""}
      <span class="count" id="count"></span>
    </div>`;
  }

  function bindToolbar(rerender) {
    const q = document.getElementById("q");
    q.addEventListener("input", () => { state.prefs.q = q.value; savePrefs(); rerender(true); });
    document.querySelectorAll("[data-pref]").forEach((b) => b.addEventListener("click", () => {
      state.prefs[b.dataset.pref] = b.dataset.val;
      savePrefs();
      route();
    }));
  }

  // ---------- catalogue ----------
  async function viewCatalogue() {
    setNav("catalogue");
    const xps = await loadXps();
    view.innerHTML = toolbar(["group", "sort"]) + `<div id="shelves"></div>`;
    const draw = () => {
      const list = xps.filter((x) => matches(x, state.prefs.q)).sort(SORTS[state.prefs.sort] || SORTS.recent);
      document.getElementById("count").textContent = `${list.length} / ${xps.length} XPs`;
      const groups = new Map();
      for (const xp of list) {
        const k = groupKey(xp, state.prefs.group);
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push(xp);
      }
      const ordered = [...groups.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
      document.getElementById("shelves").innerHTML = list.length
        ? ordered.map(([k, items]) => `<section class="shelf">
            ${state.prefs.group === "none" ? "" : `<div class="shelf-head"><h2>${esc(k)}</h2><span class="dim">${items.length}</span></div>`}
            <div class="grid">${items.map(bookHtml).join("")}</div>
          </section>`).join("")
        : `<div class="empty">nothing matches</div>`;
    };
    bindToolbar(draw);
    draw();
  }

  function bookHtml(xp) {
    const picked = state.picks.has(xp.sig);
    return `<a class="book" href="#/xp/${encodeURIComponent(xp.sig)}">
      ${coverHtml(xp)}
      <button class="pick ${picked ? "on" : ""}" data-pick="${esc(xp.sig)}" title="Select for comparison">${picked ? "✓" : "+"}</button>
      <div class="book-cap">
        <div class="t">${xp.active ? '<span class="live-dot" title="written to in the last 15 min"></span>' : ""}${esc(title(xp))}</div>
        <div class="m num">${esc(xp.sig)} · ${xp.epochs_done || 0} ép. · ${fmt(xp.best_valid_ce)}</div>
      </div>
    </a>`;
  }

  // ---------- index table ----------
  const COLUMNS = [
    ["", null],
    ["label", (x) => title(x)],
    ["sig", (x) => x.sig],
    ["profile", (x) => x.profile || ""],
    ["workspace", (x) => Cover.prettyWorkspace(x.workspace)],
    ["epochs", (x) => x.epochs_done || 0, "r"],
    ["best ce", (x) => x.best_valid_ce ?? Infinity, "r"],
    ["@ep", (x) => x.best_valid_epoch ?? 0, "r"],
    ["last ce", (x) => x.last_valid_ce ?? Infinity, "r"],
    ["origin", (x) => x.origin || ""],
    ["sessions", (x) => x.sessions || 0, "r"],
    ["ckpts", (x) => x.checkpoint_count, "r"],
    ["last activity", (x) => x.last_activity || 0, "r"],
  ];

  async function viewTable() {
    setNav("table");
    const xps = await loadXps();
    view.innerHTML = toolbar([]) + `<div class="table-wrap"><table><thead><tr>
        <th></th>${COLUMNS.map(([name, , cls], i) => `<th data-col="${i}" class="${cls || ""} ${state.prefs.tsort === name ? "sorted" : ""}">${name}${state.prefs.tsort === name ? (state.prefs.tdir > 0 ? " ↑" : " ↓") : ""}</th>`).join("")}
      </tr></thead><tbody id="rows"></tbody></table></div>`;
    const draw = () => {
      const col = COLUMNS.find(([n]) => n === state.prefs.tsort) || COLUMNS[12];
      const key = col[1] || (() => 0);
      const list = xps.filter((x) => matches(x, state.prefs.q)).sort((a, b) => {
        const va = key(a), vb = key(b);
        return (va < vb ? -1 : va > vb ? 1 : 0) * state.prefs.tdir;
      });
      document.getElementById("count").textContent = `${list.length} / ${xps.length} XPs`;
      document.getElementById("rows").innerHTML = list.map((x) => `<tr data-href="#/xp/${encodeURIComponent(x.sig)}">
        <td><input type="checkbox" data-pick="${esc(x.sig)}" ${state.picks.has(x.sig) ? "checked" : ""} aria-label="select ${esc(x.sig)}"></td>
        <td><span class="chip" style="background:${Cover.palette(x.sig).ground};--chip-ink:${Cover.palette(x.sig).ink}"></span></td>
        <td>${x.active ? '<span class="live-dot"></span>' : ""}${esc(title(x))}</td>
        <td class="dim">${esc(x.sig)}</td>
        <td>${esc(x.profile || "—")}</td>
        <td>${esc(Cover.prettyWorkspace(x.workspace) || "—")}</td>
        <td class="r num">${x.epochs_done || 0}${x.epochs_planned ? `<span class="dim">/${x.epochs_planned}</span>` : ""}</td>
        <td class="r num">${fmt(x.best_valid_ce)}</td>
        <td class="r num dim">${x.best_valid_epoch ?? "—"}</td>
        <td class="r num">${fmt(x.last_valid_ce)}</td>
        <td>${esc(x.origin || "")}${x.parents && x.parents.length ? ` <span class="dim">← ${x.parents.map(esc).join(" + ")}</span>` : ""}</td>
        <td class="r num">${x.sessions || ""}</td>
        <td class="r num">${x.checkpoint_count || ""}</td>
        <td class="r dim">${ago(x.last_activity)}</td>
      </tr>`).join("");
    };
    document.querySelectorAll("th[data-col]").forEach((th) => th.addEventListener("click", () => {
      const name = COLUMNS[+th.dataset.col][0];
      if (!name) return;
      if (state.prefs.tsort === name) state.prefs.tdir *= -1;
      else { state.prefs.tsort = name; state.prefs.tdir = ["label", "sig", "profile", "workspace", "origin"].includes(name) ? 1 : -1; }
      if (name === "best ce" || name === "last ce") state.prefs.tdir = state.prefs.tsort === name && th.classList.contains("sorted") ? state.prefs.tdir : 1;
      savePrefs();
      viewTable();
    }));
    bindToolbar(draw);
    draw();
  }

  // ---------- chart ----------
  function drawChart(el, series, opts) {
    opts = opts || {};
    const width = Math.max(el.clientWidth, 320);
    const height = opts.height || 260;
    const m = { l: 52, r: 16, t: 12, b: 28 };
    const all = series.flatMap((s) => s.values.filter((v) => v != null));
    if (!all.length) { el.innerHTML = `<div class="empty">${esc(opts.empty || "no history.json for this XP")}</div>`; return; }
    const label = (i) => (opts.labels ? opts.labels[i] : i + 1);
    const num = (v, d) => (opts.sci ? v.toExponential(2) : v.toFixed(d));
    const n = Math.max(...series.map((s) => s.values.length));
    let lo = Math.min(...all), hi = Math.max(...all);
    if (opts.clip) {
      const sorted = [...all].sort((a, b) => a - b);
      hi = Math.min(hi, sorted[Math.floor(sorted.length * 0.97)] + (hi - lo) * 0.02);
    }
    const pad = (hi - lo) * 0.06 || 0.5;
    lo -= pad; hi += pad;
    const x = (i) => m.l + (n <= 1 ? 0 : (i / (n - 1)) * (width - m.l - m.r));
    const y = (v) => m.t + (1 - (Math.min(v, hi) - lo) / (hi - lo)) * (height - m.t - m.b);
    const ticks = opts.ticks || 5;
    let g = "";
    for (let i = 0; i <= ticks; i++) {
      const v = lo + ((hi - lo) * i) / ticks;
      g += `<line class="grid-l" x1="${m.l}" x2="${width - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 3}" text-anchor="end">${num(v, Math.abs(hi - lo) < 1 ? 3 : 2)}</text>`;
    }
    const step = Math.max(1, Math.ceil(n / 10));
    for (let i = 0; i < n; i += step) g += `<text x="${x(i)}" y="${height - 8}" text-anchor="middle">${label(i)}</text>`;
    for (const t of opts.marks || []) {
      if (t.epoch >= 1 && t.epoch <= n) g += `<line x1="${x(t.epoch - 1)}" x2="${x(t.epoch - 1)}" y1="${t.full ? m.t : height - m.b}" y2="${height - m.b + 5}" stroke="${t.color || "currentColor"}" stroke-width="${t.full ? 1 : 1.5}"${t.full ? ' stroke-dasharray="2 3" opacity=".7"' : ""}><title>${esc(t.label)}</title></line>`;
    }
    const paths = series.map((s) => {
      let d = "", pen = false;
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
        pen = true;
      });
      const best = s.best != null ? `<circle cx="${x(s.best)}" cy="${y(s.values[s.best])}" r="3.5" fill="${s.color}"/>` : "";
      return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.6}" ${s.dash ? 'stroke-dasharray="4 4"' : ""} stroke-linejoin="round"/>${best}`;
    }).join("");
    el.innerHTML = `<svg class="chart" viewBox="0 0 ${width} ${height}" style="height:${height}px">
      <line class="axis" x1="${m.l}" x2="${width - m.r}" y1="${height - m.b}" y2="${height - m.b}"/>
      ${g}${paths}
      <line class="hover-l" id="hl" y1="${m.t}" y2="${height - m.b}" x1="0" x2="0" visibility="hidden"/>
      <text id="ht" x="0" y="${m.t + 10}"></text>
      <rect x="${m.l}" y="${m.t}" width="${width - m.l - m.r}" height="${height - m.t - m.b}" fill="transparent" id="hz"/>
    </svg>
    ${opts.legend === false ? "" : `<div class="legend">${series.map((s) => `<span>${s.href ? `<a href="${s.href}">` : ""}<i class="${s.dash ? "dash" : ""}" style="border-color:${s.color}"></i>${esc(s.name)}${s.href ? "</a>" : ""}</span>`).join("")}</div>`}`;
    const svg = el.querySelector("svg");
    const hz = svg.querySelector("#hz"), hl = svg.querySelector("#hl"), ht = svg.querySelector("#ht");
    hz.addEventListener("mousemove", (ev) => {
      const box = svg.getBoundingClientRect();
      const px = ((ev.clientX - box.left) / box.width) * width;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (width - m.l - m.r)) * (n - 1))));
      hl.setAttribute("x1", x(i)); hl.setAttribute("x2", x(i)); hl.setAttribute("visibility", "visible");
      const parts = series.map((s) => (s.values[i] != null ? `${s.short || s.name} ${num(s.values[i], 3)}` : null)).filter(Boolean);
      ht.textContent = `${opts.unit || "ep"} ${label(i)}  ${parts.join("  ")}${opts.notes && opts.notes[i] ? `  · ${opts.notes[i]}` : ""}`;
      const right = x(i) > width * 0.6;
      ht.setAttribute("x", right ? x(i) - 8 : x(i) + 8);
      ht.setAttribute("text-anchor", right ? "end" : "start");
    });
    hz.addEventListener("mouseleave", () => { hl.setAttribute("visibility", "hidden"); ht.textContent = ""; });
  }

  // ---------- tracks / player ----------
  function tracksHtml(files, label, limit) {
    if (!files.length) return `<div class="dim">no audio</div>`;
    const shown = limit ? files.slice(0, limit) : files;
    const rest = limit && files.length > limit;
    return `<div class="tracklist"${rest ? ` data-files="${esc(JSON.stringify(files))}" data-label="${esc(label)}"` : ""}>
      <div class="tracks">${shown.map((f) => `<button class="track" data-src="${esc(f.path)}" data-label="${esc(label + " — " + f.name)}">
        <span class="ico"></span><span class="nm">${esc(f.name.replace(/\.[^.]+$/, ""))}</span></button>`).join("")}</div>
      ${rest ? `<button class="more" data-more>${files.length - limit} more</button>` : ""}</div>`;
  }

  const audio = new Audio();
  audio.preload = "none";
  const player = document.getElementById("player");
  let currentSrc = null;

  function play(src, label) {
    if (currentSrc === src) {
      audio.paused ? audio.play() : audio.pause();
      return;
    }
    currentSrc = src;
    audio.src = fileUrl(src);
    audio.play().catch(() => {});
    document.getElementById("player-title").textContent = label;
    player.hidden = false;
  }

  function syncPlaying() {
    player.classList.toggle("on", !audio.paused);
    document.querySelectorAll(".track").forEach((t) => t.classList.toggle("playing", t.dataset.src === currentSrc && !audio.paused));
  }
  audio.addEventListener("play", syncPlaying);
  audio.addEventListener("pause", syncPlaying);
  audio.addEventListener("ended", () => {
    const tracks = [...document.querySelectorAll(".track")];
    const i = tracks.findIndex((t) => t.dataset.src === currentSrc);
    syncPlaying();
    if (i >= 0 && tracks[i + 1] && tracks[i + 1].closest(".tracks") === tracks[i].closest(".tracks")) tracks[i + 1].click();
  });
  audio.addEventListener("timeupdate", () => {
    const d = audio.duration || 0;
    document.getElementById("player-fill").style.width = d ? `${(audio.currentTime / d) * 100}%` : "0";
    const t = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    document.getElementById("player-time").textContent = `${t(audio.currentTime)} / ${d ? t(d) : "–"}`;
  });
  document.getElementById("player-toggle").addEventListener("click", () => (audio.paused ? audio.play() : audio.pause()));
  document.getElementById("player-bar").addEventListener("click", (ev) => {
    const box = ev.currentTarget.getBoundingClientRect();
    if (audio.duration) audio.currentTime = ((ev.clientX - box.left) / box.width) * audio.duration;
  });

  // ---------- XP detail ----------
  async function viewXp(sig) {
    setNav("catalogue");
    await loadXps();
    view.innerHTML = `<div class="loading">ouverture…</div>`;
    let d;
    try {
      d = await getJSON(`/api/xps/${encodeURIComponent(sig)}`);
    } catch (_) {
      view.innerHTML = `<div class="empty">no XP named ${esc(sig)}</div>`;
      return;
    }
    const xp = d.summary || state.bySig.get(sig) || { sig };
    const isBundle = xp.kind === "bundle";
    const color = chartColor(sig);
    const md = d.metadata || {};
    const parents = (xp.parents || []).map((p) => `<a href="#/xp/${encodeURIComponent(p)}">${esc(state.bySig.get(p) ? title(state.bySig.get(p)) : p)}</a>`).join(" + ");
    const children = (d.children || []).map((c) => `<a href="#/xp/${encodeURIComponent(c)}">${esc(state.bySig.get(c) ? title(state.bySig.get(c)) : c)}</a>`).join(", ");
    const numericEpochs = d.sample_epochs.filter((e) => /^\d+$/.test(e));
    const defaultEpoch = numericEpochs[numericEpochs.length - 1] || d.sample_epochs[0];
    const picked = state.picks.has(sig);

    const facts = [
      ["sig", esc(sig)],
      ["profile", esc(xp.profile || "—")],
      ["dataset", `${esc(xp.dataset_name || "—")}${xp.train_count ? ` <span class="dim">· ${xp.train_count} train clips</span>` : ""}`],
      ["epochs", `${xp.epochs_done || 0}${xp.epochs_planned ? ` / ${xp.epochs_planned} planned` : ""}`],
      ["best valid ce", `${fmt(xp.best_valid_ce)}${xp.best_valid_epoch ? ` <span class="dim">at epoch ${xp.best_valid_epoch}</span>` : ""}`],
      ["last valid ce", fmt(xp.last_valid_ce)],
      ["origin", xp.origin ? `${esc(xp.origin)}${parents ? ` from ${parents}` : ""}${xp.source_checkpoint ? ` <span class="dim">(${esc(xp.source_checkpoint)})</span>` : ""}` : "trained from scratch"],
      children ? ["descendants", children] : null,
      ["last activity", `${xp.active ? '<span class="live-dot"></span>' : ""}${ago(xp.last_activity)}`],
      ["written", esc(md.written_at ? md.written_at.slice(0, 19).replace("T", " ") : "—")],
      ["folder", `<span class="dim">${esc(xp.folder || ".dora/xps/" + sig)}</span>`],
    ].filter(Boolean);

    const notes = (d.experiments || []).map((e) => `<div class="note">
        <p><span class="k">${esc(e.status)}</span><a href="#/experiments">${esc(e.title)}</a></p>
        ${e.verdict ? `<p><span class="k">verdict</span>${esc(e.verdict)}</p>` : ""}
        ${e.next_action ? `<p><span class="k">next</span>${esc(e.next_action)}</p>` : ""}
        ${Object.entries(e.annotations || {}).map(([k, v]) => `<p><span class="k">${esc(k)}</span>${esc(typeof v === "string" ? v : JSON.stringify(v))}</p>`).join("")}
      </div>`).join("");

    view.innerHTML = `<article class="book-page">
      <div class="left">
        ${coverHtml(xp)}
        <div class="left-actions">
          <button class="ghost ${picked ? "on" : ""}" data-pick="${esc(sig)}">${picked ? "✓ selected" : "+ compare"}</button>
          ${(xp.parents || []).length ? `<a class="ghost" href="#/compare/${[...xp.parents, sig].map(encodeURIComponent).join(",")}" style="text-decoration:none">vs parent</a>` : ""}
        </div>
        ${xp.checkpoint_count ? `<div class="left-actions">
          <button class="ghost" data-use="stage" title="add it to the Scène's models">▶ play</button>
          <button class="ghost" data-use="train" title="warm start a new training from its latest checkpoint">train from</button>
        </div>` : ""}
      </div>
      <div class="right">
        <div class="crumb"><a href="#/">Catalogue</a> / ${esc(xp.profile || "—")}</div>
        <h1 class="title-xl">${esc(title(xp).replace(/_/g, " "))}</h1>
        <div class="byline">${esc(Cover.prettyWorkspace(xp.workspace) || "—")}</div>
        <dl class="facts">${facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
        <div class="section"><h3>Ledger</h3>${notes || `<div class="dim">not part of any experiment yet</div>`}<div id="ledger-controls"></div></div>
        <div class="section"><h3>Impressions <span class="dim">how playing it felt · sighted, kept apart from the blind verdicts</span></h3><div id="impressions"></div></div>
        ${familyHtml(xp, d.children || [])}
        <div class="section"><h3>Curves
            <span class="seg" id="metric-seg"></span>
          </h3><div id="chart"></div></div>
        ${d.sample_epochs.length ? `<div class="section"><h3>Samples <span class="dim">written by Dora during training</span></h3>
          <div class="epochs">${d.sample_epochs.map((e) => `<button data-epoch="${esc(e)}" class="${e === defaultEpoch ? "on" : ""}">${esc(e)}</button>`).join("")}</div>
          <div id="samples"><div class="dim">…</div></div></div>` : ""}
        <div class="section"><h3>Generations <span class="dim">${d.generations.length} run${d.generations.length === 1 ? "" : "s"}</span></h3>
          ${d.generations.length ? `<div id="gens"></div>` : `<div class="dim">no generation runs point at this XP</div>`}</div>
        ${(d.sessions || []).length ? `<div class="section"><h3>Sessions <span class="dim">${d.sessions.length} · every kind that played this model</span></h3>
          <table class="kv">${d.sessions.slice(0, 16).map((s) => `<tr><td>${esc(s.id)}</td><td>${esc(s.kind)} · ${s.take_count} take${s.take_count === 1 ? "" : "s"}${s.label ? ` · ${esc(s.label)}` : ""}</td><td class="r dim small">${esc((s.created_at || "").slice(0, 16).replace("T", " "))}</td></tr>`).join("")}</table>
          ${d.sessions.length > 16 ? `<div class="dim small"><code>session list --model ${esc(sig)}</code> lists all ${d.sessions.length}</div>` : ""}</div>` : ""}
        <div class="section"><h3>Checkpoints <span class="dim">${d.checkpoints.length}</span></h3>
          <table class="kv">${d.checkpoints.map((c) => `<tr><td>${esc(c.name)}</td><td class="num">${c.size_mb.toLocaleString()} MB · ${ago(c.modified)}</td>
            <td class="r nowrap small">${isBundle
              ? '<span class="dim">portable snapshot — not a .th checkpoint</span>'
              : `<a href="#/derive/perturb/.dora/xps/${encodeURIComponent(sig)}/${encodeURIComponent(c.name)}">perturb</a> · <a href="#/derive/merge/.dora/xps/${encodeURIComponent(sig)}/${encodeURIComponent(c.name)}">merge</a>`}</td></tr>`).join("") || `<tr><td class="dim">none left on disk</td></tr>`}</table></div>
        ${overridesHtml(d)}
      </div>
    </article>`;

    if (NR.hooks.xpLedger) NR.hooks.xpLedger(sig, d);

    // impressions: append-only, newest first
    const impEl = document.getElementById("impressions");
    let rating = null;
    const drawImpressions = () => {
      const stars = (n) => n == null ? "" : "★".repeat(n) + "☆".repeat(5 - n);
      impEl.innerHTML = `<form class="ledger-form" id="imp-form">
          <div class="row"><span>rating</span><span class="seg">${[1, 2, 3, 4, 5].map((n) => `<button type="button" data-rate="${n}" class="${n === rating ? "on" : ""}">${n}</button>`).join("")}</span></div>
          <label><span>note</span><textarea name="note" rows="2" placeholder="optional if you rate"></textarea></label>
          <label><span>session</span><select name="session"><option value="">—</option>${(d.sessions || []).map((s) => `<option value="${esc(s.id)}">${esc(s.id)} · ${esc(s.kind)}</option>`).join("")}</select></label>
          <div class="form-actions"><button type="submit" class="solid">Add</button><span class="form-msg"></span></div>
        </form>
        ${[...(d.impressions || [])].reverse().map((i) => `<div class="note">
          <p><span class="k">${esc(i.created_at.slice(0, 16).replace("T", " "))}</span>${stars(i.rating)}${i.session ? ` <span class="dim">· ${esc(i.session)}</span>` : ""}</p>
          ${i.note ? `<p>${esc(i.note)}</p>` : ""}</div>`).join("")}`;
      const form = document.getElementById("imp-form");
      form.querySelectorAll("[data-rate]").forEach((b) => b.addEventListener("click", () => {
        rating = Number(b.dataset.rate) === rating ? null : Number(b.dataset.rate);
        form.querySelectorAll("[data-rate]").forEach((x) => x.classList.toggle("on", Number(x.dataset.rate) === rating));
      }));
      form.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        const msg = form.querySelector(".form-msg");
        try {
          const saved = await postJSON(`/api/xps/${encodeURIComponent(sig)}/impressions`, { rating, note: form.elements.note.value, session: form.elements.session.value });
          d.impressions = [...(d.impressions || []), saved];
          rating = null;
          drawImpressions();
        } catch (err) {
          msg.textContent = err.message;
          msg.classList.add("err");
        }
      });
    };
    drawImpressions();

    // curves
    const trainKeys = Object.keys(d.history.train || {});
    const validKeys = Object.keys(d.history.valid || {});
    const metrics = [...new Set([...validKeys, ...trainKeys])].filter((k) => k !== "duration" && k !== "grad_scale");
    let metric = metrics.includes("ce") ? "ce" : metrics[0];
    const seg = document.getElementById("metric-seg");
    const drawMetric = () => {
      seg.innerHTML = metrics.map((k) => `<button class="${k === metric ? "on" : ""}" data-metric="${esc(k)}">${esc(k)}</button>`).join("");
      const series = [];
      const valid = d.history.valid && d.history.valid[metric];
      const train = d.history.train && d.history.train[metric];
      if (train) series.push({ name: `train ${metric}`, short: "train", values: train, color: "var(--ink-3)", dash: true, width: 1.2 });
      if (valid) {
        let best = null;
        valid.forEach((v, i) => { if (v != null && (best == null || v < valid[best])) best = i; });
        series.push({ name: `valid ${metric}`, short: "valid", values: valid, color, width: 2, best });
      }
      const marks = d.checkpoints.filter((c) => c.epoch).map((c) => ({ epoch: c.epoch, label: c.name, color: "var(--ink-2)" }));
      drawChart(document.getElementById("chart"), series, { marks, clip: /ppl|grad/.test(metric), sci: metric === "lr" });
    };
    seg.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-metric]");
      if (b) { metric = b.dataset.metric; drawMetric(); }
    });
    drawMetric();

    // samples
    const loadEpoch = async (epoch) => {
      const box = document.getElementById("samples");
      if (!box) return;
      document.querySelectorAll("[data-epoch]").forEach((b) => b.classList.toggle("on", b.dataset.epoch === epoch));
      box.innerHTML = `<div class="dim">…</div>`;
      const files = await getJSON(`/api/xps/${encodeURIComponent(sig)}/samples/${encodeURIComponent(epoch)}`);
      box.innerHTML = tracksHtml(files, `${title(xp)} · ${epoch}`, 18);
      syncPlaying();
    };
    document.querySelectorAll("[data-epoch]").forEach((b) => b.addEventListener("click", () => loadEpoch(b.dataset.epoch)));
    if (defaultEpoch) loadEpoch(defaultEpoch);

    // generations
    const gensEl = document.getElementById("gens");
    if (gensEl) {
      let shown = 8;
      const drawGens = () => {
        gensEl.innerHTML = d.generations.slice(0, shown).map((g) => `<div class="run">
            <div class="run-head"><b>${esc(g.run_id)}</b><span>${esc(g.mode)}</span>${g.checkpoint ? `<span>${esc(g.checkpoint)}</span>` : ""}
              <span>${g.duration_seconds}s</span>${g.temperature != null ? `<span>temp ${g.temperature}</span>` : ""}${g.top_k != null ? `<span>top-k ${g.top_k}</span>` : ""}${g.seed != null ? `<span>seed ${g.seed}</span>` : ""}
              ${g.listening_score != null ? `<span>listening ${fmt(g.listening_score, 2)}</span>` : ""}</div>
            ${g.listening_summary ? `<div class="summary">${esc(g.listening_summary)}</div>` : ""}
            ${tracksHtml(g.files, `${title(xp)} · ${g.run_id}`, 8)}
          </div>`).join("") + (d.generations.length > shown ? `<button class="more" id="more-gens">${d.generations.length - shown} more runs</button>` : "");
        const more = document.getElementById("more-gens");
        if (more) more.addEventListener("click", () => { shown += 12; drawGens(); });
        syncPlaying();
      };
      drawGens();
    }
  }

  function familyHtml(xp, children) {
    const chain = ancestry(xp);
    if (!chain.length && !children.length) return "";
    const mini = (x, self) => x.missing
      ? `<a href="#/xp/${encodeURIComponent(x.sig)}" title="not on disk"><div class="cover" style="background:var(--paper-2)"></div><div>${esc(x.sig)}</div></a>`
      : `<a href="#/xp/${encodeURIComponent(x.sig)}" class="${self ? "self" : ""}" title="${esc(title(x))}">${Cover.render ? coverHtml(x, { subtitle: false }) : ""}<div>${esc(x.sig)}</div></a>`;
    const parts = [];
    for (const gen of chain) parts.push(gen.map((p) => mini(p)).join(""), `<span class="arrow">→</span>`);
    parts.push(mini(xp, true));
    const kids = children.map((c) => state.bySig.get(c)).filter(Boolean);
    if (kids.length) parts.push(`<span class="arrow">→</span>`, kids.map((k) => mini(k)).join(""));
    return `<div class="section"><h3>Lineage <span class="dim">${chain.length} generation${chain.length === 1 ? "" : "s"} back · ${kids.length} descendant${kids.length === 1 ? "" : "s"}</span></h3>
      <div class="family">${parts.join("")}</div></div>`;
  }

  function overridesHtml(d) {
    const o = d.overrides || {};
    const diff = d.override_diff;
    const keys = Object.keys(o).filter((k) => !k.startsWith("/") && !k.includes("/bin/"));
    if (!keys.length) return "";
    let diffRows = "";
    if (diff) {
      diffRows = [
        ...Object.entries(diff.changed).map(([k, [was, now]]) => `<tr class="changed"><td>${esc(k)}</td><td><span class="was">${esc(was)}</span>${esc(now)}</td></tr>`),
        ...Object.entries(diff.added).map(([k, v]) => `<tr class="added"><td>${esc(k)}</td><td>${esc(v)}</td></tr>`),
        ...Object.entries(diff.removed).map(([k, v]) => `<tr><td>${esc(k)}</td><td><span class="was">${esc(v)}</span></td></tr>`),
      ].join("");
    }
    return `<div class="section"><h3>Overrides ${diff ? `<span class="dim">vs parent</span>` : ""}</h3>
      ${diff ? (diffRows ? `<table class="kv">${diffRows}</table>` : `<div class="dim">identical to the parent</div>`) : ""}
      <details ${diff ? "" : "open"}><summary>all ${keys.length} overrides</summary>
        <table class="kv">${keys.map((k) => `<tr><td>${esc(k)}</td><td>${esc(o[k])}</td></tr>`).join("")}</table>
      </details></div>`;
  }

  // ---------- compare ----------
  // Spot inks from the palette that stay distinct from each other on both paper colours.
  const COMPARE_INKS = ["#e3157e", "#6a4bff", "#1f9aa8", "#c29a00", "#1f9e33", "#ef5a40", "#9b3fb0", "#7a6a4f"];
  async function viewCompare(sigs) {
    setNav("");
    await loadXps();
    view.innerHTML = `<div class="loading">assemblage…</div>`;
    const details = await Promise.all(sigs.map((s) => getJSON(`/api/xps/${encodeURIComponent(s)}`).catch(() => null)));
    const items = details.map((d, i) => ({ sig: sigs[i], d })).filter((x) => x.d);
    if (!items.length) { view.innerHTML = `<div class="empty">nothing to compare</div>`; return; }
    const series = items.map(({ sig, d }, i) => {
      const xp = d.summary || { sig };
      const color = COMPARE_INKS[i % COMPARE_INKS.length];
      return { name: `${title(xp)} (${sig})`, short: sig, color, dash: i >= COMPARE_INKS.length, width: 1.8, href: `#/xp/${encodeURIComponent(sig)}`, values: (d.history.valid || {}).ce || [] };
    });
    const inkOf = new Map(items.map(({ sig }, i) => [sig, COMPARE_INKS[i % COMPARE_INKS.length]]));
    view.innerHTML = `<div class="compare-covers">${items.map(({ sig, d }) => `<a class="book" href="#/xp/${encodeURIComponent(sig)}">${coverHtml(d.summary || { sig })}
        <div class="book-cap"><div class="t">${esc(title(d.summary || { sig }))}</div><div class="m">${esc(sig)} · best ${fmt((d.summary || {}).best_valid_ce)}</div></div></a>`).join("")}</div>
      <div class="section"><h3>Valid ce <span class="dim">by epoch of each XP's own run</span></h3><div id="chart"></div></div>
      <div class="section"><h3>Latest samples</h3><div class="compare-cols" id="cmp-samples"></div></div>`;
    drawChart(document.getElementById("chart"), series, { height: 300 });
    const cols = document.getElementById("cmp-samples");
    for (const { sig, d } of items) {
      const col = document.createElement("div");
      const epochs = d.sample_epochs.filter((e) => /^\d+$/.test(e));
      const epoch = epochs[epochs.length - 1];
      col.innerHTML = `<div class="crumb" style="margin-bottom:8px"><i style="display:inline-block;width:14px;border-top:2px solid ${inkOf.get(sig)};vertical-align:middle;margin-right:8px"></i>${esc(title(d.summary || { sig }))}${epoch ? ` · epoch ${esc(epoch)}` : ""}</div><div class="dim">…</div>`;
      cols.appendChild(col);
      if (epoch) {
        getJSON(`/api/xps/${encodeURIComponent(sig)}/samples/${encodeURIComponent(epoch)}`).then((files) => {
          col.lastElementChild.outerHTML = tracksHtml(files, `${title(d.summary || { sig })} · ${epoch}`, 6);
          syncPlaying();
        });
      } else {
        const g = d.generations.find((r) => r.files.length);
        col.lastElementChild.outerHTML = g ? tracksHtml(g.files, `${sig} · ${g.run_id}`, 6) : `<div class="dim">no audio</div>`;
      }
    }
  }

  // ---------- router ----------
  async function route() {
    const hash = decodeURIComponent(location.hash.replace(/^#/, "")) || "/";
    const parts = hash.split("/").filter(Boolean);
    try {
      if (parts[0] === "xp" && parts[1]) await viewXp(parts[1]);
      else if (parts[0] === "table") await viewTable();
      else if (NR.views[parts[0]]) await NR.views[parts[0]](parts.slice(1));
      else if (parts[0] === "compare") await viewCompare((parts[1] || "").split(",").filter(Boolean));
      else await viewCatalogue();
    } catch (err) {
      view.innerHTML = `<div class="empty">could not load: ${esc(err.message)}</div>`;
    }
    syncPlaying();
  }

  document.addEventListener("click", (ev) => {
    const pick = ev.target.closest("[data-pick]");
    if (pick) {
      if (pick.tagName !== "INPUT") ev.preventDefault();
      ev.stopPropagation();
      togglePick(pick.dataset.pick);
      if (pick.classList.contains("ghost")) pick.textContent = state.picks.has(pick.dataset.pick) ? "✓ selected" : "+ compare";
      else if (pick.classList.contains("pick")) pick.textContent = state.picks.has(pick.dataset.pick) ? "✓" : "+";
      return;
    }
    const use = ev.target.closest("[data-use]");
    if (use) {
      useModel(use.dataset.use, decodeURIComponent(location.hash.split("/")[2] || ""));
      return;
    }
    if (ev.target.closest("[data-clear-picks]")) {
      state.picks.clear();
      savePicks();
      document.querySelectorAll("[data-pick]").forEach((el) => (el.type === "checkbox" ? (el.checked = false) : el.classList.remove("on")));
      document.querySelectorAll(".pick").forEach((el) => (el.textContent = "+"));
      return;
    }
    const track = ev.target.closest(".track");
    if (track) { play(track.dataset.src, track.dataset.label); syncPlaying(); return; }
    const more = ev.target.closest("[data-more]");
    if (more) {
      const list = more.closest(".tracklist");
      list.outerHTML = tracksHtml(JSON.parse(list.dataset.files), list.dataset.label);
      syncPlaying();
      return;
    }
    const row = ev.target.closest("tr[data-href]");
    if (row && ev.target.tagName !== "INPUT") location.hash = row.dataset.href;
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.code !== "Space" || ["INPUT", "TEXTAREA", "BUTTON"].includes(document.activeElement?.tagName) || player.hidden) return;
    ev.preventDefault();
    audio.paused ? audio.play() : audio.pause();
  });

  document.getElementById("refresh").addEventListener("click", async (ev) => {
    const btn = ev.currentTarget;
    btn.textContent = "Scanning…";
    try {
      await postJSON("/api/refresh");
      state.xps = null;
      state.experiments = null;
      await route();
    } finally {
      btn.textContent = "Rescan";
    }
  });

  const NR = (window.NR = {
    views: {},
    hooks: {},
    state, view, esc, fmt, ago, fileUrl, getJSON, postJSON, loadXps, setNav, title, coverHtml, route, drawChart, chartColor,
    storage, modelPicker, pauseMain: () => audio.pause(),
    invalidate() { state.xps = null; state.experiments = null; },
  });

  window.addEventListener("hashchange", () => { route(); window.scrollTo(0, 0); });
  renderTray();
  // ledger.js and listen.js register their views after this file runs; route once they have.
  document.addEventListener("DOMContentLoaded", route);
})();
