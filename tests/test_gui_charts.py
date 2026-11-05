from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "macchina_ricordi/gui/static"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to exercise chart rendering")
def test_chart_scientific_axes_and_hover_leave_default_formatting_unchanged():
    source = (STATIC / "app.js").read_text()
    function = source.split("  function drawChart(", 1)[1].split("  // ---------- tracks / player", 1)[0]
    script = "const assert = require('node:assert/strict');\nfunction drawChart(" + function
    script += r"""
function render(opts) {
  const listeners = {};
  const ht = {textContent: '', setAttribute() {}};
  const hl = {setAttribute() {}};
  const hz = {addEventListener(name, callback) {listeners[name] = callback;}};
  const svg = {
    querySelector(selector) {return {'#ht': ht, '#hl': hl, '#hz': hz}[selector];},
    getBoundingClientRect() {return {left: 0, width: 400};},
  };
  const el = {clientWidth: 400, innerHTML: '', querySelector() {return svg;}};
  const esc = value => String(value);
  // No legend text is needed, keeping this harness independent of the page helpers.
  drawChart(el, [{name: 'lr', values: [0.0001, null, 0.000005], color: 'black'}],
            {...opts, legend: false});
  listeners.mousemove({clientX: 52});
  const first = ht.textContent;
  listeners.mousemove({clientX: 218});
  assert.equal(ht.textContent, 'ep 2  '); // missing values must not be formatted
  listeners.mousemove({clientX: 384});
  const last = ht.textContent;
  listeners.mouseleave();
  assert.equal(ht.textContent, '');
  return {html: el.innerHTML, first, last};
}
const scientific = render({sci: true});
assert.match(scientific.html, />[0-9.]+e-\d+<\/text>/);
assert.equal(scientific.first, 'ep 1  lr 1.00e-4');
assert.equal(scientific.last, 'ep 3  lr 5.00e-6');
const standard = render({});
assert.equal(standard.first, 'ep 1  lr 0.000');
assert.equal(standard.last, 'ep 3  lr 0.000');
assert.doesNotMatch(standard.html, />[0-9.]+e-\d+<\/text>/);
"""
    result = subprocess.run([shutil.which("node"), "-e", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename", ["app.js", "train.js"])
@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to check chart options")
def test_model_and_training_charts_only_enable_scientific_notation_for_lr(filename):
    source = (STATIC / filename).read_text()
    options = source.split("clip: /ppl|grad/.test(metric), sci:", 1)[1].split("}", 1)[0]
    script = "const assert = require('node:assert/strict');\n"
    script += "const sci = metric => (" + options.strip() + ");\n"
    script += "assert.equal(sci('lr'), true);\n"
    for metric in ("ce", "ppl", "grad_norm"):
        script += f"assert.equal(sci({json.dumps(metric)}), false);\n"
    result = subprocess.run([shutil.which("node"), "-e", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
