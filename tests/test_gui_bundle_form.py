from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to exercise the browser form functions")
def test_bundle_warm_start_form_never_silently_falls_back_to_scratch() -> None:
    source = (Path(__file__).parents[1] / "macchina_ricordi/gui/static/train.js").read_text()
    request = source.split("    const request = () => {", 1)[1].split("    const remember", 1)[0]
    checkpoints = source.split("    const loadCheckpoints = async () => {", 1)[1].split("    const guide", 1)[0]
    script = """
const assert = require('node:assert/strict');
const f = new Proxy({}, {get(target, key) {return target[key] ??= {value: '', checked: false};}});
const form = {querySelector() {return {value: 'file'};}};
const NR = {state: {bySig: new Map([['bundle-test', true]])}, async getJSON() {
  return {summary: {kind: 'bundle', folder: 'models/name"quoted'}, checkpoints: [{name: 'model.safetensors'}]};
}};
const saved = {};
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('"', '&quot;');
"""
    script += "const request = () => {" + request
    script += "const loadCheckpoints = async () => {" + checkpoints
    script += """
(async () => {
  f.init_mode.value = '';
  assert.equal(request().init_from_checkpoint, '');
  for (const mode of ['xp', 'file']) {
    f.init_mode.value = mode;
    assert.throws(request, /Select a valid warm-start/);
  }
  f.init_mode.value = 'xp';
  f.init_xp.value = 'bundle-test';
  await loadCheckpoints();
  assert.ok(f.init_ckpt.innerHTML.includes('value="models/name&quot;quoted"'));
  assert.ok(!f.init_ckpt.innerHTML.includes('.dora/xps'));
  f.init_ckpt.value = 'models/published';
  assert.equal(request().init_from_checkpoint, 'models/published');
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    result = subprocess.run([shutil.which("node"), "-e", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
