// Generative duotone covers, after the Éditions Sans Soleil catalogue: one ground, one spot ink,
// a smeared image, wide capitals top and bottom, a double hairline down the spine.
// The smear is the XP's own validation curve; perturbations glitch, merges fold.
(function () {
  "use strict";

  const W = 300;
  const H = 470;

  // [ground, ink], sampled from the printed covers
  const PALETTES = [
    ["#151b3d", "#e3157e"],
    ["#1a1a1a", "#7cc9d6"],
    ["#4e3d1e", "#e6cf22"],
    ["#e3d624", "#2f2a12"],
    ["#ef5a40", "#35199a"],
    ["#ee3fe0", "#4f155a"],
    ["#f2eec8", "#3d3a7a"],
    ["#2ae23e", "#0d3a15"],
    ["#e8e8e4", "#1c1c1c"],
    ["#74e4f0", "#1c2c55"],
    ["#2d1417", "#dd7390"],
    ["#101c17", "#c9f05a"],
  ];

  function hash(text) {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function palette(sig) {
    const [ground, ink] = PALETTES[hash(sig) % PALETTES.length];
    return { ground, ink };
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  }

  function prettyWorkspace(ws) {
    if (!ws) return "";
    return ws.replace(/^\.(workspace|macchina_ricordi)_/, "").replace(/^\./, "").replace(/_/g, " ");
  }

  function wrap(text, maxChars) {
    const words = text.split(/\s+/).filter(Boolean);
    const lines = [];
    let line = "";
    for (let word of words) {
      while (word.length > maxChars) {
        if (line) { lines.push(line); line = ""; }
        lines.push(word.slice(0, maxChars));
        word = word.slice(maxChars);
      }
      if (!line) line = word;
      else if ((line + " " + word).length <= maxChars) line += " " + word;
      else { lines.push(line); line = word; }
    }
    if (line) lines.push(line);
    return lines;
  }

  // Lexend Zetta is very wide: roughly one em per capital, plus tracking.
  function fitTitle(text, width, maxLines, start, min) {
    for (let size = start; size >= min; size -= 1) {
      const perLine = Math.floor(width / (size * 1.02));
      const lines = wrap(text, perLine);
      if (lines.length <= maxLines) return { size, lines };
    }
    const perLine = Math.floor(width / (min * 1.02));
    return { size: min, lines: wrap(text, perLine).slice(0, maxLines) };
  }

  function curvePoints(values, r, minTop) {
    let ys = (values || []).filter((v) => typeof v === "number" && isFinite(v));
    if (ys.length < 3) {
      ys = [];
      let v = 0;
      for (let i = 0; i < 24; i++) { v += (r() - 0.5) * 2; ys.push(v); }
    }
    const lo = Math.min(...ys);
    const hi = Math.max(...ys);
    const span = hi - lo || 1;
    const top = Math.max(minTop, 150 + r() * 30);
    const bottom = 320 + r() * 30;
    return ys.map((v, i) => [
      -30 + (i / (ys.length - 1)) * (W + 60),
      bottom - ((v - lo) / span) * (bottom - top),
    ]);
  }

  function smoothPath(points) {
    if (points.length < 2) return "";
    let d = `M${points[0][0].toFixed(1)},${points[0][1].toFixed(1)}`;
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i - 1] || points[i];
      const p1 = points[i];
      const p2 = points[i + 1];
      const p3 = points[i + 2] || p2;
      const c1x = p1[0] + (p2[0] - p0[0]) / 6;
      const c1y = p1[1] + (p2[1] - p0[1]) / 6;
      const c2x = p2[0] - (p3[0] - p1[0]) / 6;
      const c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += ` C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    return d;
  }

  let instance = 0;

  function render(xp, opts) {
    opts = opts || {};
    const id = `cv${++instance}`;
    const sig = xp.sig || "unknown";
    const seed = hash(sig);
    const r = rng(seed);
    const { ground, ink } = palette(sig);
    const n = (x) => Math.floor(x * 1000) / 1000;

    const title = (xp.name || xp.label || sig).replace(/[_\-]+/g, " ").toUpperCase();
    const author = (prettyWorkspace(xp.workspace) || (xp.has_metadata === false ? "sans métadonnées" : "")).toUpperCase();
    const fitted = fitTitle(title, 262, 3, 24, 12);
    const authorFit = fitTitle(author, 262, 2, 15, 9);
    const titleTop = 44;
    const lineH = fitted.size * 1.12;
    const titleBottom = titleTop + fitted.lines.length * lineH;
    const subtitleY = titleBottom + 4;
    const authorTop = H - 34 - (authorFit.lines.length - 1) * authorFit.size * 1.12;

    // image parameters
    const styles = [
      { fx: 0.004, fy: 0.055, w: 1.0 },   // horizontal streaks, water
      { fx: 0.012, fy: 0.014, w: 0.7 },   // blots
      { fx: 0.06, fy: 0.005, w: 0.35 },   // vertical tear
    ];
    let pick = r() * styles.reduce((a, s) => a + s.w, 0);
    let style = styles[0];
    for (const s of styles) { if ((pick -= s.w) <= 0) { style = s; break; } }
    const done = xp.epochs_done || 0;
    const planned = xp.epochs_planned || done || 1;
    const completion = Math.max(0.15, Math.min(1, done / planned));
    const coverage = 0.18 + completion * 0.2 + r() * 0.08;
    const thr = 0.5 + (0.3 - coverage) * 0.55;
    const A = 11;
    const fieldTop = Math.max(subtitleY + 30, 110 + r() * 60);
    const fieldHeight = 180 + r() * 120;
    const strokeW = 16 + r() * 26;
    const displace = 30 + r() * 50;
    const path = smoothPath(curvePoints(xp.valid_curve, r, subtitleY + 40));

    const defs = [];
    defs.push(`<filter id="${id}f" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">
      <feTurbulence type="fractalNoise" baseFrequency="${n(style.fx)} ${n(style.fy)}" numOctaves="4" seed="${seed % 997}"/>
      <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ${A} 0 0 0 ${n(-A * thr)}" result="m"/>
      <feComposite in="SourceGraphic" in2="m" operator="in"/>
    </filter>`);
    defs.push(`<filter id="${id}s" x="-20%" y="-40%" width="140%" height="180%" color-interpolation-filters="sRGB">
      <feTurbulence type="fractalNoise" baseFrequency="${n(style.fx * 1.4)} ${n(style.fy * 1.2)}" numOctaves="3" seed="${(seed >> 3) % 997}" result="t"/>
      <feDisplacementMap in="SourceGraphic" in2="t" scale="${n(displace)}" xChannelSelector="R" yChannelSelector="G" result="d"/>
      <feTurbulence type="fractalNoise" baseFrequency="0.035 0.45" numOctaves="2" seed="${(seed >> 5) % 997}" result="g"/>
      <feColorMatrix in="g" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -6 0 0 0 3.9" result="gm"/>
      <feComposite in="d" in2="gm" operator="in"/>
    </filter>`);
    defs.push(`<filter id="${id}g" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="${(seed >> 7) % 997}"/>
      <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -9 0 0 0 3.35" result="m"/>
      <feComposite in="SourceGraphic" in2="m" operator="in"/>
    </filter>`);
    defs.push(`<linearGradient id="${id}m" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fff" stop-opacity="0"/>
      <stop offset="0.25" stop-color="#fff" stop-opacity="1"/>
      <stop offset="0.75" stop-color="#fff" stop-opacity="1"/>
      <stop offset="1" stop-color="#fff" stop-opacity="0"/>
    </linearGradient>
    <mask id="${id}k" maskUnits="userSpaceOnUse" x="0" y="0" width="${W}" height="${H}">
      <rect x="0" y="${n(fieldTop)}" width="${W}" height="${n(fieldHeight)}" fill="url(#${id}m)"/>
    </mask>`);

    let art = `<g id="${id}a">
      <rect width="${W}" height="${H}" fill="${ink}" filter="url(#${id}f)" mask="url(#${id}k)"/>
      <path d="${path}" fill="none" stroke="${ink}" stroke-width="${n(strokeW)}" stroke-linecap="butt" filter="url(#${id}s)"/>
      <path d="${path}" fill="none" stroke="${ink}" stroke-width="${n(strokeW * 0.25)}" transform="translate(0 ${n(18 + r() * 20)})" filter="url(#${id}s)" opacity="0.9"/>
    </g>`;

    let overlay = "";
    if (xp.origin === "perturb") {
      const bands = 3 + Math.floor(r() * 5);
      for (let i = 0; i < bands; i++) {
        const y = 120 + r() * 260;
        const h = 3 + r() * 26;
        const dx = (r() - 0.5) * 90;
        defs.push(`<clipPath id="${id}c${i}"><rect x="0" y="${n(y)}" width="${W}" height="${n(h)}"/></clipPath>`);
        overlay += `<g clip-path="url(#${id}c${i})"><rect x="0" y="${n(y)}" width="${W}" height="${n(h)}" fill="${ground}"/><use href="#${id}a" x="${n(dx)}"/></g>`;
      }
    } else if (xp.origin === "merge") {
      defs.push(`<clipPath id="${id}h"><rect x="${W / 2}" y="0" width="${W / 2}" height="${H}"/></clipPath>`);
      overlay += `<g clip-path="url(#${id}h)"><rect x="${W / 2}" y="0" width="${W / 2}" height="${H}" fill="${ground}"/><use href="#${id}a" transform="translate(${W} ${n(r() * 40 - 20)}) scale(-1 1)"/></g>`;
    } else if (xp.origin === "warm-start") {
      overlay += `<use href="#${id}a" transform="translate(${n((r() - 0.5) * 12)} ${n(10 + r() * 14)})" opacity="0.35"/>`;
    }

    const ruleX = W / 2;
    const ruleGap = 4.2;
    const ruleTop = subtitleY + 14;
    const ruleBottom = authorTop - authorFit.size - 12;
    const titleSvg = fitted.lines
      .map((line, i) => `<text x="${W / 2}" y="${n(titleTop + i * lineH)}" font-size="${fitted.size}">${esc(line)}</text>`)
      .join("");
    const authorSvg = authorFit.lines
      .map((line, i) => `<text x="${W / 2}" y="${n(authorTop + i * authorFit.size * 1.12)}" font-size="${authorFit.size}">${esc(line)}</text>`)
      .join("");
    const subtitle = opts.subtitle !== false && xp.profile
      ? `<text x="${W / 2}" y="${n(subtitleY)}" font-size="9.5" font-family="Georgia, 'Times New Roman', serif" font-style="normal" letter-spacing="0.02em" text-transform="none">${esc(xp.profile)}${done ? ` · ${done} ép.` : ""}</text>`
      : "";

    return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(title)}">
      <defs>${defs.join("")}</defs>
      <rect width="${W}" height="${H}" fill="${ground}"/>
      ${art}
      ${overlay}
      <rect width="${W}" height="${H}" fill="${ground}" filter="url(#${id}g)"/>
      <g stroke="${ink}" stroke-width="0.8" opacity="0.95">
        <line x1="${n(ruleX - ruleGap)}" y1="${n(ruleTop)}" x2="${n(ruleX - ruleGap)}" y2="${n(ruleBottom)}"/>
        <line x1="${n(ruleX + ruleGap)}" y1="${n(ruleTop)}" x2="${n(ruleX + ruleGap)}" y2="${n(ruleBottom)}"/>
      </g>
      <g fill="${ink}" text-anchor="middle" font-family="'Lexend Zetta', sans-serif" font-weight="300" dominant-baseline="hanging">
        ${titleSvg}
        ${subtitle}
        ${authorSvg}
      </g>
    </svg>`;
  }

  window.Cover = { render, palette, prettyWorkspace };
})();
