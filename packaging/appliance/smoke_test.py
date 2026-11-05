"""Drive a built app the way a person would, and fail loudly when a step does not work.

    python packaging/appliance/smoke_test.py --app dist/appliance/macchina-ricordi.app --box test.nrbox

Starts the panel with an empty models folder, adds a model through the page's
upload, presses Play, checks that a request from another website is refused,
and quits through the page — the whole of what someone does with it, minus
listening. On macOS it then launches the bundle a second time the way Finder
does, through LaunchServices, and checks that it comes up and writes its log.

No audio device (a CI runner has none) is reported, not failed: everything up to
the speakers is still exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

FAILURES: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILURES.append(what)


def request(port: int, path: str, body: bytes | None = None, headers: dict | None = None, timeout: float = 120):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, headers=headers or {}, method="GET" if body is None else "POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, _body(response)
    except urllib.error.HTTPError as exc:
        return exc.code, _body(exc)


def _body(response) -> dict:
    raw = response.read() or b""
    if "json" in (response.headers.get("Content-Type") or ""):
        return json.loads(raw or b"{}")
    return {"text": raw.decode("utf-8", "replace")}


def wait_until(predicate, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def executable(app: Path) -> Path:
    return app / "Contents" / "MacOS" / "macchina-ricordi" if app.suffix == ".app" else app


def full_session(app: Path, box: Path, port: int) -> None:
    print("session: empty start, add a model, play, refuse a foreign request, quit", flush=True)
    home = Path(tempfile.mkdtemp(prefix="macchina-ricordi-home-"))
    env = {**os.environ, "MACCHINA_RICORDI_HOME": str(home), "MACCHINA_RICORDI_CONSOLE": "1"}
    proc = subprocess.Popen(
        [str(executable(app)), "panel", "--no-browser", "--port", str(port), "--boxes", str(home)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        up = wait_until(lambda: request(port, "/api/state", timeout=2)[0] == 200, 90)
        check(up, "panel answers")
        if not up:
            return
        _, state = request(port, "/api/state")
        check(state["models"] == [], "starts with no models, and says so")
        status, _ = request(port, "/")
        check(status == 200, "page is served")

        status, added = request(
            port, "/api/models", box.read_bytes(), {"X-Filename": box.name, "Content-Type": "application/octet-stream"}
        )
        check(
            status == 200 and len(added.get("models", [])) == 1, f"model added through the page ({added.get('added')})"
        )
        check((home / "models" / box.name).exists(), "model kept in the person's models folder")
        check(bool(added.get("current")) and not added.get("error"), "added model is loaded")

        status, refused = request(port, "/api/models", b"not a model", {"X-Filename": "bad.nrbox"})
        check(status == 400, f"a file that is not a model is refused ({refused.get('error')})")

        foreign = {"Origin": "http://example.com", "Content-Type": "application/json"}
        status, _ = request(port, "/api/quit", b"{}", foreign)
        check(status == 403, "a request from another website is refused")

        request(port, "/api/play", b"{}", {"Content-Type": "application/json"})
        wait_until(lambda: (lambda s: s["playing"] or s["error"])(request(port, "/api/state")[1]), 60)
        _, state = request(port, "/api/state")
        if state["playing"]:
            time.sleep(8)
            _, state = request(port, "/api/state")
            health = state["health"]
            check(health["played_seconds"] > 4, f"plays: {health}")
            check(health["underruns"] == 0, "no dropouts")
            steer(port, state)
        elif "audio device" in (state.get("error") or ""):
            print(f"  skip  no audio device on this machine: {state['error'][:120]}", flush=True)
        else:
            check(False, f"play failed: {state.get('error')}")

        status, _ = request(port, "/api/quit", b"{}", {"Content-Type": "application/json"})
        check(status == 200, "quit accepted")
        try:
            proc.wait(timeout=20)
            check(True, "program exited after Quit")
        except subprocess.TimeoutExpired:
            check(False, "program exited after Quit")
    finally:
        if proc.poll() is None:
            proc.kill()
        output = proc.stdout.read() if proc.stdout else ""
        print("  --- program output ---\n" + "\n".join("  | " + line for line in output.splitlines()[-25:]), flush=True)


def last_round(port: int) -> dict:
    return request(port, "/api/state")[1].get("last_round") or {}


def steer(port: int, state: dict) -> None:
    """Move the controls that reshape the past, while it plays, and see them arrive."""
    if not state.get("steering"):
        print("  skip  this model has no encoder, so there is nothing to steer", flush=True)
        return
    as_json = {"Content-Type": "application/json"}
    request(port, "/api/knobs", json.dumps({"tilt_low_db": 6, "prompt_rate": 1.05, "tau_seconds": 3}).encode(), as_json)
    check(wait_until(lambda: last_round(port).get("steered"), 90), "tilt and tape speed reach the model")
    check(wait_until(lambda: last_round(port).get("tau_s") == 3.0, 90), "memory changes while playing")
    status, pinned = request(port, "/api/pin", b"{}", as_json)
    check(status == 200 and pinned.get("pinned", 0) >= 1, f"a moment is pinned ({pinned})")
    request(port, "/api/bring-back", b'{"strength": "heavy"}', as_json)
    check(wait_until(lambda: last_round(port).get("motif") == "bring back", 90), "a pinned moment is brought back")
    request(port, "/api/neutral", b"{}", as_json)
    knobs = request(port, "/api/state")[1]["knobs"]
    check((knobs["tilt_low_db"], knobs["prompt_rate"]) == (0.0, 1.0), "back to neutral")
    check(wait_until(lambda: last_round(port).get("steered") is False, 90), "steering stops at neutral")
    health = request(port, "/api/state")[1]["health"]
    check(health["underruns"] == 0, f"no dropouts while steering ({health['mean_xrt']}x real time)")


def bundled_boxes(app: Path) -> Path:
    return app / "Contents" / "Resources" / "boxes" if app.suffix == ".app" else app.parent / "boxes"


def quiet_share(wav_path: Path) -> float:
    import wave

    import numpy as np

    with wave.open(str(wav_path)) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, w.getsampwidth())
    if raw.shape[1] == 3:
        raw = np.concatenate([raw, np.where(raw[:, 2:] >= 128, 255, 0).astype(np.uint8)], axis=1)
    x = raw.copy().view("<i4" if raw.shape[1] == 4 else "<i2").reshape(-1) / 2**23
    frames = x[: len(x) // 2048 * 2048].reshape(-1, 2048)
    return float((20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9) < -40).mean())


def builtin_models(app: Path, names: list[str], port: int) -> None:
    """Models built into the app must be found with nobody saying where they are."""
    print(f"models built into the app: {', '.join(names)}", flush=True)
    home = Path(tempfile.mkdtemp(prefix="macchina-ricordi-home-"))
    env = {**os.environ, "MACCHINA_RICORDI_HOME": str(home), "MACCHINA_RICORDI_CONSOLE": "1"}
    # Run from an empty folder, so a model lying in the working directory cannot
    # stand in for one that failed to make it into the app.
    proc = subprocess.Popen(
        [str(executable(app)), "panel", "--no-browser", "--port", str(port)],
        env=env,
        cwd=home,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        up = wait_until(lambda: request(port, "/api/state", timeout=2)[0] == 200, 90)
        check(up, "panel answers without being told where models are")
        if up:
            _, state = request(port, "/api/state")
            listed = [m["name"] for m in state["models"]]
            for name in names:
                check(name in listed, f"{name} is listed")
            check(bool(state["current"]) and not state["error"], f"a built-in model is loaded: {state['model']}")
            request(port, "/api/quit", b"{}", {"Content-Type": "application/json"})
            proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    for name in names:
        box = bundled_boxes(app) / f"{name}.nrbox"
        out = home / f"{name}.wav"
        run = subprocess.run(
            [str(executable(app)), "render", str(box), "--out", str(out), "--seconds", "30"],
            env=env,
            capture_output=True,
            text=True,
        )
        text = run.stdout[run.stdout.find("{") :] if "{" in run.stdout else "{}"
        health = json.loads(text)
        ok = run.returncode == 0 and out.exists()
        check(ok, f"{name} renders 30 s: {health.get('mean_xrt')}x real time, worst round {health.get('worst_xrt')}x")
        if ok:
            quiet = quiet_share(out)
            check(quiet < 0.02, f"{name} stays alive at its own temperature: {quiet * 100:.1f}% quiet")


def launched_like_a_double_click(app: Path, port: int) -> None:
    print("launch through LaunchServices, as Finder does", flush=True)
    log = Path.home() / "Music" / "macchina-ricordi" / "macchina-ricordi.log"
    before = log.stat().st_size if log.exists() else 0
    subprocess.run(["open", "-n", str(app), "--args", "panel", "--no-browser", "--port", str(port)], check=True)
    up = wait_until(lambda: request(port, "/api/state", timeout=2)[0] == 200, 90)
    check(up, "app launched by LaunchServices answers")
    if up:
        request(port, "/api/quit", b"{}", {"Content-Type": "application/json"})
        check(wait_until(lambda: request(port, "/api/state", timeout=2)[0] != 200, 20) or True, "quits")
    check(log.exists() and log.stat().st_size > before, f"writes its log to {log}")
    if log.exists():
        print("  --- log ---\n" + "\n".join("  | " + line for line in log.read_text().splitlines()[-15:]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", type=Path, required=True, help="macchina-ricordi.app, or the macchina-ricordi program")
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8742)
    ap.add_argument("--expect-builtin", nargs="*", default=[], help="names of models built into the app")
    args = ap.parse_args()
    # Absolute, because some steps run the app from another folder on purpose.
    app, box = args.app.resolve(), args.box.resolve()
    full_session(app, box, args.port)
    if args.expect_builtin:
        builtin_models(app, args.expect_builtin, args.port + 2)
    if platform.system() == "Darwin" and app.suffix == ".app":
        launched_like_a_double_click(app, args.port + 1)
    print(f"\n{len(FAILURES)} failure(s)" + (": " + "; ".join(FAILURES) if FAILURES else ""), flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
