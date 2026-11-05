"""Blind A/B sessions driven from the browser.

The same pair selection, preference store, undo and session sidecar as ``judge ab``;
only the loop is different. The server holds each session, so the page only ever sees
"A", "B" and opaque audio URLs until the session is finished and revealed.

Candidate paths are repo-relative, exactly as ``judge ab`` writes them when run from the
repository root, so both front ends share one store and one set of judged pairs.
"""

from __future__ import annotations

import json
import threading
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from macchina_ricordi.judging.ab import (
    GROUP_BY_BUCKET_PREFIX,
    GROUP_BY_MODEL,
    GROUP_BY_SESSION,
    AbPair,
    _presented_row,
    load_candidates,
    matchups,
    pair_context_metadata,
    parse_testing,
    select_pairs,
    testing_group_by,
    varied_knob,
)
from macchina_ricordi.judging.ab_context import ContextAudioError, ContextAudioPreparer
from macchina_ricordi.judging.preference import (
    DEFAULT_PREFERENCE_PATH,
    DEFAULT_SESSIONS_DIRNAME,
    PreferenceRecord,
    append_preference,
    filter_by_question,
    judged_pair_keys,
    load_preferences,
    metric_agreement,
    new_session_id,
    recorded_questions,
    remove_last_preference,
    session_agreement,
    write_session_sidecar,
)
from macchina_ricordi.judging.questions import PAIRWISE, PAIRWISE_CONTEXT, QUESTIONS, get_question
from macchina_ricordi.models.catalog import Catalog
from macchina_ricordi.sessions.takes import knob_text

from .jobs import relative

VERDICTS = ("a", "b", "tie", "skip")
SLOTS = ("a", "b", "context")


class AbSetupError(ValueError):
    """A session request that cannot be turned into pairs; the message says why, as the CLI does."""


@dataclass(frozen=True)
class AbSetup:
    sources: tuple[str, ...]
    question: str = "better"
    testing: str = "metric"
    pairs: int = 20
    seed: int = 0
    cross_group: bool = True
    group_by: str = GROUP_BY_MODEL
    group_column: str = "sweep_id"
    target_metric: str | None = None
    disagree: tuple[str, ...] | None = None
    target_quantile: float = 0.5
    context_window: int = 3
    seconds: float | None = None
    listener: str = "me"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AbSetup:
        try:
            sources = tuple(str(item) for item in payload.get("sources") or ())
            disagree_raw = payload.get("disagree") or ()
            if isinstance(disagree_raw, str):
                disagree_raw = [part for part in disagree_raw.split(",")]
            disagree = tuple(part.strip() for part in disagree_raw if str(part).strip())
            seconds = payload.get("seconds")
            return cls(
                sources=sources,
                question=str(payload.get("question") or "better"),
                testing=str(payload.get("testing") or "metric"),
                pairs=int(payload.get("pairs") or 20),
                seed=int(payload.get("seed") or 0),
                cross_group=bool(payload.get("cross_group", True)),
                group_by=str(payload.get("group_by") or GROUP_BY_MODEL),
                group_column=str(payload.get("group_column") or "sweep_id"),
                target_metric=str(payload.get("target_metric") or "").strip() or None,
                disagree=disagree or None,
                target_quantile=float(payload.get("target_quantile") or 0.5),
                context_window=int(payload.get("context_window") or 3),
                seconds=float(seconds) if seconds not in (None, "", 0, "0") else None,
                listener=str(payload.get("listener") or "me").strip() or "me",
            )
        except (TypeError, ValueError) as exc:
            raise AbSetupError(f"invalid session request: {exc}") from exc


@dataclass
class _Action:
    index: int
    outcome: str
    record: PreferenceRecord | None


@dataclass
class BlindSession:
    session_id: str
    setup: AbSetup
    pairs: list[AbPair]
    store_path: Path
    repo_root: Path
    sidecar: dict[str, Any]
    metric_lookup: dict[str, dict[str, float]] = field(default_factory=dict)
    preparer: AbstractContextManager[Any] = field(default_factory=nullcontext)
    history: list[_Action] = field(default_factory=list)
    finished: bool = False
    sidecar_path: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def question(self) -> Any:
        return get_question(self.setup.question)

    @property
    def position(self) -> int:
        return len(self.history)

    def state(self) -> dict[str, Any]:
        """What the page may know: no paths, groups or labels until the reveal."""
        judged = sum(1 for action in self.history if action.record is not None)
        return {
            "session_id": self.session_id,
            "question": self.question.id,
            "prompt": self.question.prompt,
            "contextual": self.question.type == PAIRWISE_CONTEXT,
            "seconds": self.setup.seconds,
            "total": len(self.pairs),
            "position": self.position,
            "judgements": judged,
            "can_undo": bool(self.history) and not self.finished,
            "finished": self.finished,
        }

    def audio_path(self, index: int, slot: str) -> Path | None:
        if slot not in SLOTS or not 0 <= index < len(self.pairs):
            return None
        if not self.finished and index != self.position:
            return None
        pair = self.pairs[index]
        if slot == "context":
            raw = pair.context_reel_path
        else:
            raw = (pair.first if slot == "a" else pair.second).playable_path
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.repo_root / path
        return path if path.is_file() else None

    def answer(self, index: int, verdict: str) -> None:
        with self.lock:
            if self.finished:
                raise AbSetupError("this session is finished")
            if index != self.position:
                raise AbSetupError(f"pair {index + 1} is not the current pair ({self.position + 1})")
            if verdict not in VERDICTS or verdict not in (*self.question.keys, "skip"):
                raise AbSetupError(f"answer must be one of {', '.join((*self.question.keys, 'skip'))}")
            pair = self.pairs[index]
            record = None
            if verdict != "skip":
                record = PreferenceRecord(
                    pair_id=pair.pair_id,
                    path_a=pair.first.path,
                    path_b=pair.second.path,
                    winner=verdict,
                    listener=self.setup.listener,
                    question=self.question.id,
                    session_id=self.session_id,
                    seconds_played=self.setup.seconds,
                    context={
                        "group_a": pair.first.group,
                        "group_b": pair.second.group,
                        "label_a": pair.first.label,
                        "label_b": pair.second.label,
                        "source": "gui",
                        "testing": self.setup.testing,
                        **pair_context_metadata(pair),
                    },
                    context_paths=list(pair.context_paths),
                )
                append_preference(record, path=self.store_path)
            self.history.append(_Action(index, "skipped" if verdict == "skip" else verdict, record))

    def undo(self) -> None:
        with self.lock:
            if self.finished or not self.history:
                raise AbSetupError("nothing to undo")
            action = self.history[-1]
            if action.record is not None:
                try:
                    remove_last_preference(action.record, path=self.store_path)
                except ValueError as exc:
                    raise AbSetupError(f"cannot undo: {exc}") from exc
            self.history.pop()

    def finish(self) -> dict[str, Any]:
        with self.lock:
            if not self.finished:
                outcomes = {action.index: action.outcome for action in self.history}
                presented = [_presented_row(self.pairs[action.index], action.outcome) for action in self.history]
                presented += [
                    _presented_row(pair, "unreached") for index, pair in enumerate(self.pairs) if index not in outcomes
                ]
                judged = [action.record for action in self.history if action.record is not None]
                sidecar_path = write_session_sidecar(
                    {
                        "question": self.question.id,
                        "prompt": self.question.prompt,
                        "listener": self.setup.listener,
                        "seconds_played": self.setup.seconds,
                        "player": "browser",
                        "judgements": len(judged),
                        "pairs": presented,
                        **self.sidecar,
                    },
                    session_id=self.session_id,
                    store_path=self.store_path,
                )
                self.sidecar_path = relative(sidecar_path, self.repo_root)
                self.finished = True
                self.preparer.__exit__(None, None, None)
            return self.reveal()

    def reveal(self) -> dict[str, Any]:
        if not self.finished:
            raise AbSetupError("the reveal comes after the session is finished")
        outcomes = {action.index: action.outcome for action in self.history}
        knob = varied_knob(self.setup.testing)
        rows = []
        wins: dict[str, list[int]] = {}
        for index, pair in enumerate(self.pairs):
            outcome = outcomes.get(index, "unreached")
            rows.append(
                {
                    "index": index,
                    "outcome": outcome,
                    "a": _side(pair.first, self._focus(), knob),
                    "b": _side(pair.second, self._focus(), knob),
                }
            )
            if outcome in ("a", "b", "tie"):
                for side, candidate in (("a", pair.first), ("b", pair.second)):
                    tally = wins.setdefault(candidate.group, [0, 0, 0])
                    tally[2] += 1
                    if outcome == side:
                        tally[0] += 1
                    elif outcome == "tie":
                        tally[1] += 1
        return {
            **self.state(),
            "testing": self.setup.testing,
            "knob": knob,
            "matchups": matchups(((pair, outcomes.get(index, "")) for index, pair in enumerate(self.pairs)), knob)
            if knob
            else [],
            "focus": self._focus(),
            "agreement": self._agreement(),
            "store": relative(self.store_path, self.repo_root),
            "sidecar": self.sidecar_path,
            "pairs": rows,
            "groups": [
                {"group": group, "wins": w, "ties": t, "appearances": n}
                for group, (w, t, n) in sorted(wins.items(), key=lambda item: -item[1][0] / max(item[1][2], 1))
            ],
        }

    def _focus(self) -> list[str]:
        return [metric for metric in (self.setup.target_metric, *(self.setup.disagree or ())) if metric]

    def _agreement(self) -> list[dict[str, Any]]:
        """As the terminal reveal: this session's agreement per metric, beside the store's over the same pool."""
        judged = [action.record for action in self.history if action.record is not None]
        if varied_knob(self.setup.testing):
            return []
        metrics = self._focus() or sorted({metric for values in self.metric_lookup.values() for metric in values})
        if not judged or not metrics:
            return []
        rows = session_agreement(
            judged,
            load_preferences(self.store_path),
            self.metric_lookup,
            question=self.question.id,
            metrics=metrics,
            testing=self.setup.testing,
        )
        return [
            {"session": item.to_dict(), "store": store.to_dict() if store is not None else None} for item, store in rows
        ]


def _side(candidate: Any, focus: list[str], knob: str | None) -> dict[str, Any]:
    return {
        "path": candidate.path,
        # a model session's group is the shared knobs; the side is told apart by its model, which the page names
        "group": candidate.controls.get("model", "") if knob == "model" else candidate.group,
        "label": candidate.label,
        "metrics": {metric: candidate.metrics[metric] for metric in focus if metric in candidate.metrics},
        "knob": f"{knob}={knob_text(candidate.controls.get(knob))}" if knob and knob != "model" else "",
    }


WITNESS_SHOWN = 40


class AbManager:
    """Holds the live sessions for one server; sessions are finished (sidecar written) on close."""

    def __init__(self, repo_root: Path, *, store_path: Path | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.store_path = (store_path or self.repo_root / DEFAULT_PREFERENCE_PATH).resolve()
        self._sessions: dict[str, BlindSession] = {}
        self._lock = threading.Lock()

    def questions(self) -> list[dict[str, Any]]:
        return [
            {"id": q.id, "prompt": q.prompt, "type": q.type, "metrics": list(q.metrics)}
            for q in QUESTIONS
            if q.type in (PAIRWISE, PAIRWISE_CONTEXT)
        ]

    def sources(self, catalog: Catalog) -> dict[str, list[dict[str, Any]]]:
        """Every session with something to judge, newest first. A stream offers only the rounds it scored live."""
        from ..models.record import model_display_names
        from ..sessions.store import list_sessions

        scored = {Path(stream.path).resolve(): stream.scored_rounds for stream in catalog.streams}
        names = model_display_names(self.repo_root)
        rows = []
        for session in list_sessions(self.repo_root):
            takes = scored.get(session.folder.resolve(), 0) if session.kind == "stream" else session.take_count
            if takes < 1:
                continue
            models = [names.get(sig, sig) for sig in session.sigs] or [Path(m).name for m in session.models]
            rows.append(
                {
                    "path": relative(session.folder, self.repo_root),
                    "id": session.id,
                    "kind": session.kind,
                    "takes": takes,
                    "models": models or ["?"],
                    "started_at": session.created_at,
                }
            )
        return {"sessions": rows}

    def create(self, setup: AbSetup) -> BlindSession:
        try:
            question = get_question(setup.question)
        except KeyError as exc:
            raise AbSetupError(str(exc.args[0])) from exc
        if question.type not in (PAIRWISE, PAIRWISE_CONTEXT):
            raise AbSetupError(f"'{question.id}' is a {question.type} question and cannot be asked pairwise")
        if not setup.sources:
            raise AbSetupError("pick at least one session")
        if not 1 <= setup.pairs <= 200:
            raise AbSetupError("pairs must be between 1 and 200")
        if setup.disagree and len(setup.disagree) != 2:
            raise AbSetupError(f"disagreement targeting takes exactly two metrics, got {len(setup.disagree)}")
        if setup.disagree and setup.target_metric:
            raise AbSetupError("metric targeting and disagreement targeting are two different targetings; pick one")
        try:
            parse_testing(setup.testing)
        except ValueError as exc:
            raise AbSetupError(str(exc)) from None
        within = setup.testing != "groups"
        knob = varied_knob(setup.testing)
        if (
            not within
            and setup.group_by not in (GROUP_BY_MODEL, GROUP_BY_SESSION)
            and not setup.group_by.startswith(GROUP_BY_BUCKET_PREFIX)
        ):
            raise AbSetupError("group by must be model_xp, session or bucket:METRIC")
        group_by = testing_group_by(setup.testing, setup.group_by)
        sources = [self._source(raw) for raw in setup.sources]
        contextual = question.type == PAIRWISE_CONTEXT
        if contextual:
            non_streams = [source for source in sources if not (self.repo_root / source / "performance.jsonl").exists()]
            if non_streams:
                raise AbSetupError(f"'{question.id}' needs stream sessions; not streams: {', '.join(non_streams)}")

        candidates = load_candidates(sources, group_column=setup.group_column, group_by=group_by)
        if len(candidates) < 2:
            raise AbSetupError("need at least two playable, scored clips across the chosen sources")
        already = judged_pair_keys(load_preferences(self.store_path), question=question.id)
        pairs = select_pairs(
            candidates,
            count=setup.pairs,
            seed=setup.seed,
            cross_group=setup.cross_group and not within,
            within_group=within,
            vary=knob,
            exclude_keys=already,
            target_metric=setup.target_metric,
            disagree=list(setup.disagree) if setup.disagree else None,
            target_quantile=setup.target_quantile,
            context_window=setup.context_window if contextual else None,
            allow_same_group_fallback=not contextual,
        )
        if not pairs:
            if contextual:
                raise AbSetupError(
                    "no eligible shared-context pairs remain: pairs must come from one session, 1–3 playable "
                    "positions apart, and share at least five of the previous eight rounds"
                )
            if setup.target_metric or setup.disagree:
                raise AbSetupError(
                    "no unjudged pairs separate on that targeting; widen the quantile, add sources, or drop it"
                )
            if knob == "model":
                raise AbSetupError("no two unjudged clips from different models share every knob")
            if knob:
                raise AbSetupError(f"no two unjudged clips share a model and every knob but {knob}, which must differ")
            if within:
                raise AbSetupError("no two unjudged clips share a model and knobs; add sources with longer runs")
            raise AbSetupError("no new pairs to judge; every eligible pair is already in the store")

        preparer: AbstractContextManager[Any] = nullcontext()
        if contextual:
            context_preparer = ContextAudioPreparer()
            context_preparer.__enter__()
            try:
                pairs = context_preparer.prepare(pairs)
            except ContextAudioError as exc:
                context_preparer.__exit__(None, None, None)
                raise AbSetupError(f"shared-context audio preparation failed: {exc}") from exc
            preparer = context_preparer

        groups = sorted({candidate.group for candidate in candidates})
        session = BlindSession(
            session_id=self._unique_session_id(),
            setup=setup,
            pairs=list(pairs),
            store_path=self.store_path,
            repo_root=self.repo_root,
            metric_lookup={candidate.path: candidate.metrics for candidate in candidates},
            preparer=preparer,
            sidecar={
                "source": "gui",
                "sources": sources,
                "seed": setup.seed,
                "testing": setup.testing,
                "group_by": group_by,
                "group_column": setup.group_column,
                "cross_group": setup.cross_group and not within,
                "target_metric": setup.target_metric,
                "disagree": list(setup.disagree) if setup.disagree else None,
                "target_quantile": setup.target_quantile,
                "candidates": len(candidates),
                "groups": groups,
                "context_window": setup.context_window if contextual else None,
            },
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def _unique_session_id(self) -> str:
        base = new_session_id()
        sessions_dir = self.store_path.parent / DEFAULT_SESSIONS_DIRNAME
        candidate, suffix = base, 1
        with self._lock:
            while candidate in self._sessions or (sessions_dir / f"{candidate}_session.json").exists():
                suffix += 1
                candidate = f"{base}-{suffix}"
        return candidate

    def _source(self, raw: str) -> str:
        path = (self.repo_root / raw).resolve()
        if not path.is_relative_to(self.repo_root) or not path.is_dir():
            raise AbSetupError(f"not a directory in this repo: {raw}")
        return path.relative_to(self.repo_root).as_posix()

    def get(self, session_id: str) -> BlindSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            if not session.finished:
                session.finish()

    def summary(self, metrics_csv: str | None = None) -> dict[str, Any]:
        records = load_preferences(self.store_path)
        questions = []
        for question_id in recorded_questions(records):
            subset = filter_by_question(records, question_id)
            decided = [record for record in subset if record.winner in ("a", "b")]
            questions.append({"question": question_id, "judgements": len(subset), "decided": len(decided)})
        sessions_dir = self.store_path.parent / DEFAULT_SESSIONS_DIRNAME
        sessions = []
        if sessions_dir.is_dir():
            for path in sorted(sessions_dir.glob("*_session.json"), reverse=True)[:30]:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                sessions.append(
                    {
                        "session_id": payload.get("session_id"),
                        "question": payload.get("question"),
                        "judgements": payload.get("judgements"),
                        "offered": len(payload.get("pairs") or []),
                        "source": payload.get("source") or "score ab",
                        "sources": payload.get("sources") or [],
                    }
                )
        scores_dir = self.repo_root / "artifacts" / "scores"
        csvs = sorted(
            (relative(path, self.repo_root) for path in scores_dir.glob("*.csv")) if scores_dir.is_dir() else (),
            reverse=True,
        )
        agreement: list[dict[str, Any]] = []
        if metrics_csv and records:
            agreement = self._agreement(records, metrics_csv)
        verdicts = self._other_verdicts()
        return {
            "store": relative(self.store_path, self.repo_root),
            "judgements": len(records),
            "questions": questions,
            "sessions": sessions,
            "metrics_csvs": csvs,
            "metrics_csv": metrics_csv,
            "agreement": agreement,
            **verdicts,
        }

    def _other_verdicts(self) -> dict[str, Any]:
        """The verdicts that are not A/B pairs, from the logs beside the store (``judging.store``)."""
        from macchina_ricordi.judging.store import load_verdicts, verdict_counts

        folder = self.store_path.parent
        verdicts = load_verdicts(
            preferences_path=self.store_path,
            labels_path=folder / "labels.jsonl",
            witness_path=folder / "witness.jsonl",
            kinds=("label", "witness"),
        )
        witness = [verdict for verdict in verdicts if verdict.kind == "witness"][-WITNESS_SHOWN:]
        return {
            "verdict_counts": verdict_counts(verdicts),
            "witness": [
                {
                    "path": relative(verdict.paths[0], self.repo_root),
                    "score": float(verdict.answer),
                    "ratings": verdict.detail.get("ratings") or {},
                    "by": verdict.by,
                    "created_at": verdict.created_at,
                    "error": verdict.detail.get("error"),
                }
                for verdict in reversed(witness)
            ],
        }

    def _agreement(self, records: list[PreferenceRecord], metrics_csv: str) -> list[dict[str, Any]]:
        from macchina_ricordi.scoring.batch import read_metric_lookup
        from macchina_ricordi.scoring.cli import PREFERENCE_METRICS

        path = (self.repo_root / metrics_csv).resolve()
        if not path.is_relative_to(self.repo_root) or not path.is_file():
            raise AbSetupError(f"no metrics CSV at {metrics_csv}")
        lookup = read_metric_lookup(path)
        metrics = list(PREFERENCE_METRICS)
        rows = []
        for question_id in recorded_questions(records):
            family = [metric for metric in get_question(question_id).metrics if metric not in metrics]
            for item in metric_agreement(records, lookup, metrics=metrics + family, question=question_id):
                rows.append(item.to_dict())
        return rows
