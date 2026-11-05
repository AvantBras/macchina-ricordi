"""A small local HTTP server for the GUI. Stdlib only, so it runs from ``.venv`` as-is.

Every request must name the loopback address the server is bound to in its Host header, which stops a web page from
reaching it through DNS rebinding. Writes (ledger edits, A/B verdicts, model impressions, training launches,
housekeeping) are POSTs that also need a custom header, which a cross-origin page cannot send without a CORS
preflight this server never approves.
"""

from __future__ import annotations

import json
import mimetypes
import re
from collections.abc import Collection
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from macchina_ricordi.judging.store import DEFAULT_IMPRESSION_PATH, ImpressionRecord, append_impression
from macchina_ricordi.models.ledger import LedgerEditError, save_experiment

from .ab import AbManager, AbSetup, AbSetupError
from .corpus import CorpusManager, CorpusRequest, CorpusRequestError
from .data import GuiIndex, to_jsonable
from .derive import DeriveManager, DeriveRequest, DeriveRequestError
from .housekeeping import HousekeepingManager, HouseRequest
from .stage import StageManager, StageNotFound
from .training import LaunchRequest, TrainingManager, TrainingRequestError

STATIC_DIR = Path(__file__).resolve().parent / "static"
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
_CHUNK = 256 * 1024
_MAX_BODY = 1024 * 1024
WRITE_HEADER = "X-Macchina-Ricordi"
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")

mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("audio/flac", ".flac")


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        index: GuiIndex,
        ab: AbManager | None = None,
        training: TrainingManager | None = None,
        house: HousekeepingManager | None = None,
        corpus: CorpusManager | None = None,
        derive: DeriveManager | None = None,
    ) -> None:
        super().__init__(address, GuiRequestHandler)
        self.index = index
        self.ab = ab or AbManager(index.repo_root)
        self.stage = StageManager(index.repo_root, active_trainings=self._training_labels)
        self.training = training or TrainingManager(index.repo_root)
        self.house = house or HousekeepingManager(
            index.repo_root, dora_dir=index.dora_xps.parent, active_sigs=self._training_sigs
        )

        self.corpus = corpus or CorpusManager(
            index.repo_root,
            xps_by_workspace=self._xps_by_workspace,
            generation_dirs=self._generation_dirs,
            active_workspaces=self._training_workspaces,
            musicgen_check=_musicgen_check,
        )
        self.derive = derive or DeriveManager(index.repo_root, musicgen_check=_merge_check, zoo=lambda: index.zoo)

    def _xps_by_workspace(self) -> dict[str, list[str]]:
        by_workspace: dict[str, list[str]] = {}
        for summary in self.index.summaries():
            if summary.workspace:
                by_workspace.setdefault(summary.workspace, []).append(summary.sig)
        return by_workspace

    def _generation_dirs(self, sigs: Collection[str]) -> list[dict[str, Any]]:
        """Folders of audio these XPs generated: generation runs and sweeps, newest first."""
        wanted = set(sigs)
        catalog = self.index.catalog
        found: dict[Path, dict[str, Any]] = {}
        for run in catalog.generations:
            if run.model_sig in wanted:
                folder = Path(run.metadata_path).parent
                found.setdefault(folder, {"kind": "generation", "sig": run.model_sig, "when": run.written_at})
        for sweep in catalog.sweeps:
            if sweep.model_sig in wanted:
                found.setdefault(Path(sweep.path), {"kind": "sweep", "sig": sweep.model_sig, "when": sweep.created_at})
        rows = []
        for folder, info in found.items():
            wavs = sum(1 for _ in folder.rglob("*.wav")) if folder.is_dir() else 0
            if wavs:
                rows.append({"path": _repo_relative(folder, self.index.repo_root), "wavs": wavs, **info})
        rows.sort(key=lambda row: row.get("when") or "", reverse=True)
        return rows

    def _training_workspaces(self) -> set[str]:
        return {run["workspace"] for run in self.training.runs() if run["active"] and run.get("workspace")}

    def _training_labels(self) -> list[str]:
        runs = self.training.runs()
        return [run.get("effective_label") or run.get("xp_sig") or run["id"] for run in runs if run["active"]]

    def _training_sigs(self) -> set[str]:
        return {run["xp_sig"] for run in self.training.runs() if run["active"] and run.get("xp_sig")}


class GuiRequestHandler(BaseHTTPRequestHandler):
    server: GuiServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        self._dispatch(head_only=False)

    def do_HEAD(self) -> None:
        self._dispatch(head_only=True)

    def do_POST(self) -> None:
        if not self._write_allowed():
            self._send_error(HTTPStatus.FORBIDDEN, "writes are only accepted from the GUI page itself")
            return
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        path = unquote(urlsplit(self.path).path)
        index, ab = self.server.index, self.server.ab
        try:
            if path == "/api/refresh":
                index.refresh()
                self._send_json({"ok": True, "xps": len(index.summaries())})
            elif path == "/api/ledger":
                fields = body.get("fields")
                if not isinstance(fields, dict):
                    raise LedgerEditError("fields must be an object")
                result = save_experiment(
                    index.repo_root, str(body.get("id") or ""), fields, create=bool(body.get("create"))
                )
                index.refresh()
                self._send_json(to_jsonable(result))
            elif path == "/api/train/preview":
                self._send_json(self.server.training.preview(LaunchRequest.from_payload(body.get("request") or {})))
            elif path == "/api/train/launch":
                detail = self.server.training.launch(
                    LaunchRequest.from_payload(body.get("request") or {}),
                    preview_key=str(body.get("preview_key") or ""),
                    start_now=bool(body.get("start_now")),
                )
                self._send_json(detail)
            elif path == "/api/corpus/preview":
                self._send_json(
                    self.server.corpus.corpus_preview(CorpusRequest.from_payload(body.get("request") or {}))
                )
            elif path == "/api/corpus/selections":
                self._send_json(self.server.corpus.save_selection(body))
            elif path == "/api/corpus/selections/delete":
                self._send_json(self.server.corpus.delete_selection(str(body.get("name") or "")))
            elif path == "/api/corpus/launch":
                self._send_json(
                    self.server.corpus.corpus_launch(
                        CorpusRequest.from_payload(body.get("request") or {}), str(body.get("key") or "")
                    )
                )
            elif (match := re.fullmatch(r"/api/corpus/workspaces/([^/]+)/weights/delete", path)) is not None:
                self._send_json(self.server.corpus.weights_delete(match.group(1), str(body.get("name") or "")))
            elif (match := re.fullmatch(r"/api/corpus/workspaces/([^/]+)/mix", path)) is not None:
                self._send_json(self.server.corpus.mix_save(match.group(1), body))
            elif (match := re.fullmatch(r"/api/corpus/workspaces/([^/]+)/rebalance(/save|/reset)?", path)) is not None:
                corpus, workspace = self.server.corpus, match.group(1)
                if match.group(2) == "/save":
                    self._send_json(
                        corpus.rebalance_save(workspace, str(body.get("digest") or ""), str(body.get("name") or ""))
                    )
                elif match.group(2) == "/reset":
                    self._send_json(corpus.rebalance_reset(workspace, str(body.get("dataset") or "")))
                else:
                    self._send_json(corpus.rebalance_start(workspace, body))
            elif path == "/api/derive/preview":
                self._send_json(self.server.derive.preview(DeriveRequest.from_payload(body.get("request") or {})))
            elif path == "/api/derive/launch":
                self._send_json(
                    self.server.derive.launch(
                        DeriveRequest.from_payload(body.get("request") or {}), str(body.get("key") or "")
                    )
                )
            elif path == "/api/house/plan":
                self._send_json(self.server.house.plan(HouseRequest.from_payload(body.get("request") or {})))
            elif path == "/api/house/commit":
                result = self.server.house.commit(
                    HouseRequest.from_payload(body.get("request") or {}), str(body.get("key") or "")
                )
                if result["action"] != "archive":
                    index.refresh()
                self._send_json(result)
            elif (match := re.fullmatch(r"/api/train/jobs/([^/]+)/stop", path)) is not None:
                self._send_json(self.server.training.stop(match.group(1)))
            elif path == "/api/stage/preview":
                self._send_json(self.server.stage.preview(body.get("request") or {}))
            elif path == "/api/stage/launch":
                self._send_json(self.server.stage.launch(body.get("request") or {}))
            elif (match := re.fullmatch(r"/api/stage/([^/]+)/(keys|set)", path)) is not None:
                stage, session_id = self.server.stage, match.group(1)
                if match.group(2) == "keys":
                    self._send_json(stage.press(session_id, body.get("keys")))
                else:
                    self._send_json(stage.set(session_id, str(body.get("field") or ""), body.get("value")))
            elif (match := re.fullmatch(r"/api/xps/([^/]+)/impressions", path)) is not None:
                if index.xp_folder(match.group(1)) is None:
                    self._send_error(HTTPStatus.NOT_FOUND)
                    return
                record = ImpressionRecord.from_dict({**body, "model": match.group(1), "created_at": ""})
                record = append_impression(record, path=index.repo_root / DEFAULT_IMPRESSION_PATH)
                self._send_json(to_jsonable(record))
            elif path == "/api/ab/sessions":
                session = ab.create(AbSetup.from_payload(body))
                self._send_json(session.state())
            elif (match := re.fullmatch(r"/api/ab/sessions/([^/]+)/(answer|undo|finish)", path)) is not None:
                session = ab.get(match.group(1))
                if session is None:
                    self._send_error(HTTPStatus.NOT_FOUND)
                    return
                action = match.group(2)
                if action == "answer":
                    session.answer(int(body.get("index", -1)), str(body.get("verdict") or ""))
                    self._send_json(session.state())
                elif action == "undo":
                    session.undo()
                    self._send_json(session.state())
                else:
                    self._send_json(session.finish())
            else:
                self._send_error(HTTPStatus.NOT_FOUND)
        except (LedgerEditError, AbSetupError, TrainingRequestError, CorpusRequestError, ValueError) as exc:
            self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except StageNotFound:
            self._send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            # an unexpected failure still gets an answer; otherwise the page only sees a dropped socket
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _write_allowed(self) -> bool:
        return self.headers.get(WRITE_HEADER) == "1" and self._host_allowed()

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        port = self.server.server_address[1]
        allowed = {f"{name}:{port}" for name in _LOOPBACK_HOSTS}
        if port == 80:  # browsers leave the default port out of Host
            allowed.update(_LOOPBACK_HOSTS)
        return host in allowed

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"body is not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload

    def _dispatch(self, *, head_only: bool) -> None:
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query)
        index = self.server.index
        if not self._host_allowed():
            self._send_error(HTTPStatus.FORBIDDEN, "this page is only served to this machine")
            return
        try:
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", head_only=head_only, cache=False)
            elif path.startswith("/static/"):
                self._send_static(path.removeprefix("/static/"), head_only=head_only)
            elif path.startswith("/files/"):
                resolved = index.resolve_file(path.removeprefix("/files/"))
                if resolved is None:
                    self._send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_file(resolved, head_only=head_only)
            elif path == "/api/xps":
                self._send_json(to_jsonable(index.summaries()))
            elif path == "/api/experiments":
                self._send_json(index.experiments())
            elif path == "/api/corpus":
                self._send_json(self.server.corpus.workspaces())
            elif path == "/api/corpus/library":
                self._send_json(self.server.corpus.library())
            elif path == "/api/corpus/selections":
                self._send_json(self.server.corpus.selections())
            elif path == "/api/corpus/library/folder":
                try:
                    self._send_json(self.server.corpus.library_folder(query.get("path", [""])[0]))
                except CorpusRequestError as exc:
                    self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            elif (match := re.fullmatch(r"/api/corpus/workspaces/([^/]+)/rebalance", path)) is not None:
                self._send_json({"job": self.server.corpus.rebalance_state(match.group(1))})
            elif (match := re.fullmatch(r"/api/corpus/(workspaces|jobs)/([^/]+)", path)) is not None:
                try:
                    if match.group(1) == "jobs":
                        self._send_json(self.server.corpus.job(match.group(2)))
                    else:
                        self._send_json(self.server.corpus.workspace(match.group(2)))
                except CorpusRequestError as exc:
                    self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            elif path == "/api/derive":
                self._send_json(self.server.derive.overview())
            elif path == "/api/derive/checkpoint":
                try:
                    self._send_json(self.server.derive.describe(query.get("path", [""])[0]))
                except DeriveRequestError as exc:
                    self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            elif (match := re.fullmatch(r"/api/derive/jobs/([^/]+)", path)) is not None:
                try:
                    self._send_json(self.server.derive.job(match.group(1)))
                except DeriveRequestError as exc:
                    self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            elif path == "/api/house":
                self._send_json(self.server.house.inventory())
            elif path == "/api/house/job":
                self._send_json({"job": self.server.house.job()})
            elif path == "/api/stage":
                self._send_json(self.server.stage.overview())
            elif (match := re.fullmatch(r"/api/stage/([^/]+)", path)) is not None:
                try:
                    self._send_json(self.server.stage.frame(match.group(1), follow=query.get("follow", [""])[0]))
                except StageNotFound:
                    self._send_error(HTTPStatus.NOT_FOUND)
            elif path == "/api/train/options":
                self._send_json(self.server.training.options())
            elif path == "/api/train/status":
                self._send_json(self.server.training.status())
            elif (match := re.fullmatch(r"/api/train/jobs/([^/]+)", path)) is not None:
                try:
                    self._send_json(self.server.training.run_detail(match.group(1)))
                except TrainingRequestError as exc:
                    self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            elif path == "/api/ab/setup":
                self._send_json({"questions": self.server.ab.questions(), **self.server.ab.sources(index.catalog)})
            elif path == "/api/ab/summary":
                try:
                    self._send_json(self.server.ab.summary(query.get("metrics_csv", [None])[0]))
                except AbSetupError as exc:
                    self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            elif (match := re.fullmatch(r"/api/ab/sessions/([^/]+)/audio/(\d+)/(a|b|context)", path)) is not None:
                session = self.server.ab.get(match.group(1))
                audio = session.audio_path(int(match.group(2)), match.group(3)) if session else None
                if audio is None:
                    self._send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_file(audio, head_only=head_only, cache=False)
            elif (match := re.fullmatch(r"/api/ab/sessions/([^/]+)", path)) is not None:
                session = self.server.ab.get(match.group(1))
                if session is None:
                    self._send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_json(session.reveal() if session.finished else session.state())
            elif path == "/api/generations":
                sig = query.get("sig", [None])[0]
                self._send_json(to_jsonable(index.generations_for(sig)))
            elif (match := re.fullmatch(r"/api/xps/([^/]+)/samples/([^/]+)", path)) is not None:
                samples = index.samples(match.group(1), match.group(2))
                if samples is None:
                    self._send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_json(to_jsonable(samples))
            elif (match := re.fullmatch(r"/api/xps/([^/]+)", path)) is not None:
                detail = index.xp_detail(match.group(1))
                if detail is None:
                    self._send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_json(detail)
            else:
                self._send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            # an unexpected failure still gets an answer; otherwise the page only sees a dropped socket
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _send_static(self, relative: str, *, head_only: bool) -> None:
        candidate = (STATIC_DIR / relative).resolve()
        if not candidate.is_relative_to(STATIC_DIR) or not candidate.is_file():
            self._send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(candidate, head_only=head_only, cache=candidate.suffix == ".woff2")

    def _send_json(self, payload: Any) -> None:
        body = json.dumps(payload, default=str, allow_nan=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str | None = None) -> None:
        body = json.dumps({"error": message or status.phrase}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, head_only: bool, cache: bool = True) -> None:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/json":
            content_type += "; charset=utf-8"
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and size > 0 and (match := _RANGE_RE.match(range_header.strip())):
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:
                start = max(size - int(last), 0)
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(end - start + 1, 0)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(
    index: GuiIndex,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    ab: AbManager | None = None,
    training: TrainingManager | None = None,
    house: HousekeepingManager | None = None,
    corpus: CorpusManager | None = None,
    derive: DeriveManager | None = None,
) -> GuiServer:
    return GuiServer((host, port), index, ab, training, house, corpus, derive)


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _musicgen_check() -> str | None:
    from macchina_ricordi.musicgen.runtime import require_musicgen_python

    try:
        require_musicgen_python(None)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return f"data prepare needs the AudioCraft runtime: {exc}"
    return None


def _merge_check() -> str | None:
    from macchina_ricordi.musicgen.runtime import require_musicgen_python

    try:
        require_musicgen_python(None)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return f"merge and perturb need the AudioCraft runtime (torch): {exc}"
    return None
