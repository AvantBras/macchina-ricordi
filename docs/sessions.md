# Sessions

Everything `play` makes is a **session**: one occasion of making audio from one or more models, as a list of
**takes** (single audio files), each with the controls that made it. Generate, continue, sweep, stream, duet and
variations are kinds of session, read and listed the same way. See the [glossary](glossary.md#session).

| Kind | Made by | Its takes |
|---|---|---|
| `generate` | `play generate` | one batch of samples |
| `continue` | `play continue` | one batch of continuations of a prompt |
| `sweep` | `play sweep` | one take per grid point (and per sample) |
| `stream` | `play stream` | one take per round, written live |
| `duet` | `play duet` | the stereo mix, then each voice's stem |
| `variations` | `play variations` | N re-performances of a parent take, then its codec round trip |

## Where they go

```
artifacts/sessions/2026-10-06/
├── generate_20261006T090503Z/
│   ├── session.json      # kind, models, starting controls, engine, created/finished, take count
│   ├── takes.jsonl       # one line per take: audio, label, role, model, seed, controls, scores, parent
│   ├── takes/000.wav …
│   └── run.json          # the generate request, which `play replay` reads
├── sweep_20261006T090521Z_smoke/     # --output-prefix smoke becomes the _smoke label
│   ├── session.json  takes.jsonl
│   └── sweep_plan.json  index.csv  index.html  combo_000/ …
├── stream_20261006T090634Z/
│   ├── session.json      # written when the stream stops
│   └── control.json  live_status.json  performance.jsonl  events.jsonl  manifest.json  rounds/ …
├── duet_20261006T090656Z/
└── variations_20261006T090649Z/
```

A session's id is its folder name: `<kind>_<run id>[_<label>]`. Every kind writes `session.json` when it finishes.
Every kind but the stream also writes `takes.jsonl`. A stream's takes are the rows of its `performance.jsonl`,
which it appends to while it plays. The producer's own files (a sweep's `index.csv`, a stream's `events.jsonl`)
stay beside the record, so nothing that read them before needs to change.

The folders in `artifacts/sessions/` that are not dates (`listening_odyssey/`, `chain50_exploration/`, the
Markdown notes) are the hand-kept listening notebook, not sessions.

**Older sessions** stay where they were made: `artifacts/runs/`, `artifacts/sweeps/`, `artifacts/streams/`,
`artifacts/duets/`, `artifacts/variations/`. They have no `session.json`; `macchina_ricordi/sessions/read.py` reads them
from their own files, so they list, show and judge like new ones. Pass `--output-dir` to put a new session
somewhere else; it is still read the same way.

## Naming a session or a take

A **session reference** is its folder, its id, or any part of the id that matches one session:
`stream_20261005T141908Z`, `20261005T141908Z` (an old stream's id is its run id), `20261005T1419`.

A **take reference** is `<session>#<n>`, counting from 0: `stream_20261006T090634Z#2`. In a stream, take *n* is
round *n* unless the stream rerolled (a reroll is another take of the same round, listed as `round 3 · take 1`).
A take's audio file works too.

## Commands

```bash
python -m macchina_ricordi.cli session list                         # newest first; --kind, --model deep, --all, --json
python -m macchina_ricordi.cli session show 20261005T141908Z --takes
python -m macchina_ricordi.cli session show generate_20261006T090503Z --json   # header plus every take

python -m macchina_ricordi.cli judge ab stream_20261005T141908Z sweep_20261006T090521Z_smoke
python -m macchina_ricordi.cli score rescore generate_20261006T090503Z variations_20261006T090649Z   # artifacts/scores/rescore.csv
python -m macchina_ricordi.cli play replay generate_20261006T090503Z
python -m macchina_ricordi.cli play variations --xp-path deep --input stream_20261006T090634Z#2
python -m macchina_ricordi.cli play monitor --session 20261005T141908Z
```

- **`judge ab`**, **`judge label`** and **`score rescore`** take sessions of any kind, by folder or id. They read the same listing of playable takes, `sessions/takes.py`.
  A session found by id is written to the preference store with the same relative clip paths as its folder would
  be, so verdicts never count twice.
  A duet offers only its mix (the voices are parts of it), and variations leave out the reference round trip.
  Streams offer the rounds they scored live; `score rescore` scores every round.
- **`score rescore`** takes any session and scores every take.
- **`play replay`** remakes a generate or continue session and compares bytes. A stream round depends on what
  played before it, so it cannot be replayed. A sweep's takes come from the sweep helper (fp32, no CUDA graphs)
  while replay runs the generate engine (fp16, CUDA graphs), so they would never match byte for byte.
- **`play variations --input`** takes a take reference, and the new session records it as its parent.
- **`play monitor`** with no `--session` watches the newest stream in either home.
- **The GUI's Écoute page** lists every session with something to judge, of every kind, in one table.

## In code

```python
from macchina_ricordi.sessions.store import list_sessions, find_session, find_take, seal_session, new_session_dir

session = find_session("20261005T141908Z")  # Session: id, kind, path, models, sigs, controls, take_count
for take in session.takes():  # Take: index, audio, label, role, model, seed, controls, scores
    ...
```

A producer that writes a new kind of output asks `new_session_dir(kind)` for its folder and calls
`seal_session(folder)` when it finishes; `read.py` then needs one adapter that turns its files into a `Session`
and its `Take`s.

`Session.facts` holds what a producer recorded that is particular to its kind: a run id, a generation's prefix
and model label, a sweep's recipe and grid size, a stream's stop reason and rounds written. `session.json` keeps
them; for a session sealed before they existed, `read.facts_of(session)` reads them again from the producer's
files. The catalog (`model catalog`) builds its generation, sweep and stream rows from sessions and their takes,
and lists duets and variations too, so it sees exactly what `session list` sees. A take's `listening_score` is
read from its listening report by `macchina_ricordi/listening/reports.py`, the same reader the catalog uses.
