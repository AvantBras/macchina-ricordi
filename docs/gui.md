# The GUI

A local web page for browsing the zoo (every XP in `.dora/xps/`, its curves, the samples Dora wrote during training, the generation runs that point at it, its lineage), for keeping the experiment ledger, for playing a running stream, for blind A/B listening, for building and rebalancing datasets, for launching and watching training, and for housekeeping `.dora/xps`.

```bash
python -m macchina_ricordi.cli gui            # http://127.0.0.1:8765/
python -m macchina_ricordi.cli gui --open     # and open it in the browser
python -m macchina_ricordi.cli gui --port 9000
python -m macchina_ricordi.cli gui --store /tmp/practice/preferences.jsonl   # A/B verdicts somewhere else
```

It needs nothing beyond `.venv` (standard library server, fonts bundled) and binds to localhost only. Browsing reads the same files `model catalog` reads, so it is safe to leave open while training runs. It writes only when you ask it to:

- the experiment ledger, at `research/training/experiment_ledger.toml` — a runtime file the GUI creates on your first save when you create or edit an experiment (nothing ships with the checkout). Track the ledger in Git if you want version history.
- blind A/B verdicts, appended to `artifacts/preferences/preferences.jsonl` (or `--store`), with a session sidecar next to it, exactly as `judge ab` does.
- stream launches: the `play stream` command a terminal would run, detached, with its log in `artifacts/stage_jobs/` and a `launch.json` in its session; then the stream's `control.json`, when you move one of its controls: the edit a TUI key makes, under the same lock, and only while the stream plays. See [playing a stream](#playing-a-stream).
- training launches: the `train run` CLI does the writing, as it would from a terminal, and the GUI adds one record per launch. See [training](#training).
- datasets: new corpora (`training_datasets/<name>/`) and new workspaces, by running `data corpus` and `data prepare`; saved weights under `artifacts/weights/` (never inside a workspace). See [datasets](#datasets).
- housekeeping: deleting epoch checkpoints or whole XP folders, and copying checkpoints to the archive disk, through the admin commands' own code, with their reports. See [housekeeping](#housekeeping).

Writes require a loopback `Host` and a custom header, which ordinary cross-site browser requests
cannot supply without a CORS preflight the server does not allow. **This is not authentication:**
other local processes can supply those headers and invoke the same launch, stop and deletion actions.
Use the GUI only on a machine whose local users and processes you trust; do not expose it through a
reverse proxy, tunnel, or shared/public server. See [security and trust boundaries](../SECURITY.md).

## Views

- **Catalogue**: one cover per XP, shelved by recipe, workspace or lineage family. Filter by label, sig, recipe, workspace or ledger id; order by recency, best valid ce, epochs or name. A green dot means the XP's `history.json` or logs changed in the last 15 minutes.
- **Index**: the same XPs as a sortable table: epochs done/planned, best valid ce and its epoch, last valid ce, origin and parents, sessions that played it, checkpoints left on disk.
- **XP page**: facts, ledger entries that name the XP (verdict, next action, annotations), the lineage strip (ancestors → this → descendants), train/valid curves for every metric in `history.json` with checkpoint epochs ticked on the axis, Dora's per-epoch samples, generation runs with their listening summaries, every session that played it (any kind), your impressions of it (a 1–5 gut rating and/or a note, optionally tied to one of those sessions, appended to `artifacts/preferences/impressions.jsonl`; sighted, so kept apart from the blind verdicts), checkpoints, and the Dora overrides, with the diff against the parent shown first. **▶ play** adds the model to the Scène's start form and **train from** opens the Atelier with a warm start from it.
- **Compare**: pick XPs with `+` on a cover (or the index checkboxes, or **vs parent** on an XP page) to overlay their valid ce and play their latest samples side by side. Related XPs usually share reference prompts, so the same file names line up across columns.
- **Expériences**: `research/training/experiment_ledger.toml`, one entry per row with its member XPs as covers and a link to compare them all. See [the ledger](#the-ledger).
- **Scène**: the stream sessions, the ones playing first, and a form that starts one. Open one to play it with every control and reading of `play stream --tui`. See [playing a stream](#playing-a-stream).
- **Écoute**: blind A/B sessions. See [blind listening](#blind-listening).
- **Corpus**: the prepared workspaces, what each holds (by source and by length) and which XPs trained on it, new corpora built from the library, and rebalancing. See [datasets](#datasets).
- **Atelier**: the GPU, what is training now, a launch form, and past launches, each with a live page. See [training](#training).
- **Merge & perturb** (linked from the Atelier, and from each checkpoint on an XP page): make a new starting point by blending two checkpoints or adding noise to one, and see every merged or perturbed checkpoint on disk. See [merge and perturb](#merge-and-perturb).
- **Réserve**: what `.dora/xps` holds on disk, folder by folder, and the actions that trim it. See [housekeeping](#housekeeping).

After updating the code, restart a running `gui`: the page's scripts are served fresh on each load, but the server's Python is the one it started with.

Space toggles playback. When a track ends, the next one in the same list starts.

## The ledger

**+ New experiment** on the Expériences page, or **+ new experiment from this XP** on an XP page (the sig is filled in), writes a new `[[experiments]]` entry. **edit** on an entry changes its title, status, hypothesis, verdict, next action, tags and XP list. From an XP page, the Ledger section attaches the XP to an existing experiment or removes it from one.

Every save sets `updated_at` to today and rewrites the file in the layout a hand-written entry has: the header comments stay, arrays that fit on a line stay inline, `[experiments.annotations]` tables and any keys the GUI does not edit are kept as they were. A save is refused, and nothing is written, if the new file would not parse back to exactly what was intended. Check `git diff` and commit it like any other file.

## Playing a stream

The Scène page is `play stream --tui` in the browser: the same readings, the same controls, the same keys. It never plays sound itself. The stream's own process does, so nothing the browser does (a slow tab, a reload, closing it) can make the music drop out. Start a stream from the page or in a terminal, with or without `--tui`; the TUI and the page can steer one stream at once.

**Starting one** (*Entrée en scène*): add models in number-key order, the first one starts and the others are the `--playlist` the number keys swap between, then set the knobs to start at (each stays live afterwards). Before anything starts, a preview checks the form:

- each model resolved: its checkpoint (the first one's `@best`, `@final` or `@<epoch>` is kept; a swap always loads the latest, so a selector on the others is ignored, and the preview says so) and its trained segment length;
- the guidance for those knobs on the first model, the same lines the session page shows;
- what else holds the GPU: a training running, or a stream already playing (it would also share the speakers). The button then reads **Play anyway**;
- the command it will run.

**Play** runs that command as a detached job, like a training: it keeps playing if you close the page or stop the GUI. The page waits until the stream's loop runs (until its `live_status.json` exists) and opens the session; the model loads there, under *generating round 0*. A stream that dies before its loop runs (a broken install, a bad argument) leaves the end of its log on the form instead; one whose model fails to load reads *ended without a manifest* on its page, with the error in its log. A stream that found no sound device plays on silently, as from a terminal: its page says playback is off. The log, everything the stream printed, its worker's diagnostics included, is `artifacts/stage_jobs/<id>.log`, shown under *Log* on the session's page; the session holds a `launch.json` naming it.

**Stopping** a stream is the `Q` command, from any front end: it finishes the round in progress, plays out the buffer and writes its manifest. For a stream started from this page, a stop the loop has not acted on within a minute gets a Ctrl-C (the stream's own clean stop), then SIGTERM, then SIGKILL, 20 seconds apart. A GUI restarted since the launch only sends the command.

**The list** shows the streams playing now, then every past session, newest first. A stream counts as playing while its process keeps `live_status.json` fresh (see [the live status](musicgen-stream.md#the-live-status)). *Stopping* means it was told to stop and is playing out its buffer; *ended without a manifest* means its process went away without finishing (killed, or crashed). A stream started before `live_status.json` existed counts as playing while it still appends rounds, and its page says the readings are estimated.

**A session's page**:

- The strip at the top: what the loop is doing (*generating round 214 · 4.2 s*, *throttled — buffer full*), the round you are hearing and how far behind the generator it is, the round an edit made now lands at, the buffer against its cap (the tick is where it resumes), and the last round's generation time against the audio it adds. Between two polls the buffer, the phase timer and the playhead keep moving at real-time speed. Under it, each knob edit follows its way to the speakers: *waiting · lands in round N*, *generated in round N · audible in ~X s*, *audible now* (the round it landed in is read off the logged rounds, so an edit pressed just as a round started moves on to the next one). A one-shot command still pending is named.
- The round ribbon: one tile per round, oldest first, tinted by model. The playing tile is solid, with a playhead running through it; dashed tiles are the round being generated (filling as it goes) and the round an edit made now lands in (pulsing while an edit waits for it). Glyphs over a tile mark what its row records (pin, reroll, injection, motif recap, stuck kick, tail rescue, model swap, a slow round), and an accent underline marks a round whose knobs differ from the one before; hover a tile for its round, model, marks and edits.
- The controls. Each knob has a slider, `−` / `+` buttons and its keys; holding a button sweeps, as holding a key does. Steps, limits and messages are the TUI's own: the page sends keys to the server, which applies the TUI's keymap, and a slider sets a value within the knob's limits. The tilt and rate sliders mark where they are flat, and their cap turns to the accent colour away from it, because they compound while held.
- The buffer cap and its floor, the throttle, the rest before each round (*Pacing*), the gestures (reroll, pin, inject light or heavy, stop), and the `--playlist` models.
- The readings of the round just generated, or of the round playing (`f`), with the candidate table; `b` hides them (blind), as in the TUI. Below them: arc, novelty, silence and the buffer over the last 60 rounds, each self-scaled, with a dashed line wherever a knob moved; hover for a round's value and the edits it ran with.
- The guidance and the last events.

The keys are the TUI's ([key table](musicgen-stream.md#playing-from-one-terminal---tui)), except Tab, which the browser keeps for moving the focus. `?` lists them. They act while no text field has the focus.

**What it writes**, once a stream plays: only its `control.json`, by the read-modify-write a TUI key does, under `control.json.lock`, and only while it plays; an ended session opens read-only on its last frame. `p` pins the round the speakers are on, read from `live_status.json` when you press it. The number keys need the playlist, which `play stream --playlist` records in the session as `playlist.json`.

## Blind listening

The Écoute page is `judge ab` in the browser, on the same store: the same pair selection, the same questions, the same targeting, the same sidecar, and pairs already judged for a question are never offered again, whichever front end judged them. Pick what you are testing (**a metric**, the default: both clips from one model at the same knobs; **a generation parameter**: one model, every other knob equal, the named one different; **models**: the same knobs, different models or checkpoints; or **groups**: across models, evenings or metric buckets, see [what a session tests](scoring-structure.md#blind-ab--the-part-that-actually-settles-arguments)), the question, the [sessions](sessions.md) to draw clips from (one table of every kind: a stream offers the rounds it scored live, a duet its mix), and optionally a targeting (see [Spending judgements where they inform](scoring-structure.md#spending-judgements-where-they-inform)), then **Begin**.

During the session the page knows nothing about the clips: the server holds the pairs and serves the audio at opaque `A`/`B` URLs, so neither the page nor its developer tools can tell which model is playing. The verdict buttons unlock once both clips (and the context reel, for `newer`) have been heard.

| key | action |
|---|---|
| `1` / `2` | play A / B |
| `c` | play the context reel (`newer`) |
| `r` | replay the whole sequence |
| `space` | pause / resume |
| `a` / `b` / `t` | record A, B or tie and advance |
| `s` | skip the pair |
| `u` | undo the last verdict or skip |
| `q` | end the session and reveal |

The reveal lists every pair with what is behind each side (playable again, with the targeted metric's value), then each metric's agreement with this session's verdicts beside the store's running total over the chosen sources (only same-condition verdicts, in a metric session). A parameter session shows instead, for each pair of values, how often the lower value won, tied and lost; a models session, for each pair of models, how often the first won, tied and lost. A groups session also shows wins per group. The Écoute page also shows what the store holds per question, the recent sessions, and — once you pick a scored CSV from `artifacts/scores/` (`score rescore` output) — each metric's agreement with your verdicts, as `judge prefs --metrics-csv` prints it. Under **Other verdicts** it lists the alarm labels per question and the captioner witness's latest ratings (`judge witness` records them in `artifacts/preferences/witness.jsonl`), the same logs `judge log` reads.

For `newer`, the page fills in `novelty_whitened` as the targeted metric (or, testing groups, `bucket:novelty_whitened` as the grouping), which is how that question is meant to be asked: high against low novelty inside one session. Stopping the server finishes any open session, so its sidecar is always written.

The store records clips by path, and "already judged" is decided by comparing paths. `judge ab` is run from the repo root with relative source directories (a session given by id is turned into its relative folder), so its paths look like `artifacts/streams/<id>/rounds/<n>.wav` or `artifacts/sessions/<date>/<session>/takes/000.wav`. The GUI switches to the repo root when it starts and stores paths the same way; that is what lets the two front ends share one store. To try the page without adding to the real store (a practice run, a demo), start it with `--store` pointing somewhere else.

## Training

The Atelier page queues trainings and lets you watch, stop and cancel them. It never talks to Dora directly: a training is a job of the same [queue](musicgen.md#queue-trainings) as `train queue add`, and it runs the same `train run` command you would type.

**Launching.** Pick a recipe, a prepared workspace (any recipe trains on any workspace; with none, run `data prepare` first), optionally weights saved for that workspace (see [datasets](#datasets)), where to start from (scratch, a checkpoint of an XP, a [portable bundle](models.md#the-portable-bundle) in `models/` or picked from the file list, or any checkpoint path), and optionally a run label, epochs, segment duration and more overrides. Epochs and segment duration become `--override optim.epochs=…` and `--override dataset.segment_duration=…`, after the recipe's own. The **?** next to *more overrides* opens a guide to the keys worth knowing (schedule, data, model shape, checkpoints), each with the selected recipe's value or AudioCraft's default and what it does; clicking one adds it to the list. Then **Preview**:

- The preview is `train run --dry-run`, in-process, which records nothing. Like the CLI's dry run it renders the AudioCraft dataset config and may build `logs/dataset_fingerprint.json`. Resolving the Dora signature takes about five seconds.
- It shows the signature, the label, the epochs, the segment duration against the clip length, warm-start warnings, what is training or queued ahead of it, the terminal command that queues the same job, and the full Dora command.
- If that signature is already on disk, it says so: Dora will **continue that XP**, not start a new one. Change the label or an override for a fresh run.

**Sampling** decides how training crops are drawn from the workspace's files:

- *each file equally often* (the default, and what every recipe does): AudioCraft picks a file uniformly, then a random crop in it. Short files are over-represented per second of audio.
- *in proportion to length*: every second of audio is equally likely, so a 10-minute track is drawn five times as often as a 2-minute one. This adds `--override +dataset.train.sample_on_duration=true`.

The page shows the chosen workspace's shortest and longest file, so you can see whether the choice matters: for clips cut to one length it barely does, for whole tracks it can be a factor of 50. It changes the training split only; validation keeps drawing every file equally, so valid ce stays comparable across runs. The other kind of weighting, per-clip cluster weights, is the *weights* choice above: they multiply into either sampling choice. (A workspace weighted in place before saved weights existed says so here; `data rebalance --reset` clears it.)

**Crops** chooses between fresh crops (the default) and a token pool: *train from a token pool* adds `--pool`, which encodes the crops once before training and reuses them, about 8× faster for a nano run ([details](musicgen.md#train-from-a-token-pool---pool-opt-in)). The preview names the pool, its crops and repeats, and whether it is already built or how long the build takes; the job shows **building** meanwhile, and can be stopped like a training. *pool seed* draws another set of crops, which a warm start from a model trained on the same pool needs.

**Add** is only accepted for exactly the form that was previewed, within 30 minutes; change anything and you preview again. It queues the training as a job in `artifacts/train_jobs/<id>/` and makes sure a queue worker runs: with the GPU free it starts at once (the button says **Add and start**), otherwise it waits under *File d'attente* until the jobs ahead of it are done. With something training, **start now, beside it** skips the queue (two runs on one 12 GB GPU usually means both crawl or one dies). The GPU watchdog is on by default here, unlike on the command line.

**Where things go.** A job runs detached, in its own session, so it keeps going if you close the page or stop the GUI. Its folder holds what any training records (see [MusicGen](musicgen.md#every-training-is-a-job)): `job.json`, `job_status.json`, `train_plan.json`, `train_run_status.json`, and `train.log` (everything it printed, which the job page shows). Nothing is written into the dataset.

**Watching.** The page lists what trains (*En cours*), what waits (*File d'attente*, in order, each with **Cancel**), and every past training in *Historique*, whoever started it: this page, a terminal, an agent. A job's page polls every few seconds: state, epochs done out of planned, time per epoch (from the stage durations Dora records in `history.json`), time remaining, the curves, the log tail, and the GPU. States are *queued*, *starting* (the job's process is up, Dora has not reported yet), *running*, *stopping*, then *completed*, *failed*, *stopped* or *cancelled*. A job whose process is gone without recording how it ended reads as *failed*.

**Stopping.** **Stop training** sends SIGTERM to the Dora process tree (its process group, with the watchdog on), not to the `train run` above it: that one sees its child exit and records the outcome itself (a signal to it would kill it before it could). Whatever is still running 30 seconds later gets SIGKILL. Checkpoints Dora has written stay; progress since the last one is lost. On a queued job the button is **Cancel**.

**Trainings started in a terminal or by an agent** are jobs too (`train run` records one), so they show and stop here exactly like the page's own. Trainings from before jobs, recorded in a workspace's `logs/`, are not listed.

## Merge and perturb

The page runs `model merge` and `model perturb` (see [Model Lifecycle](musicgen-admin.md#merge-checkpoints)) as detached jobs, the way the Atelier runs training. The result keeps only the weights, without the optimizer state, so it is a starting point for a warm start, not a run to resume. It lands in `artifacts/checkpoints/merged/<name>/checkpoint.th` or `artifacts/checkpoints/perturbed/<name>/checkpoint.th`, next to its `merged_from.json` / `perturbed_from.json` sidecar. **warm start a run from it**, on the finished job or on any row of the table, opens the Atelier with that file as the warm start.

The page is **Greffe** in the menu. **How far it moved.** Each sidecar also records `movement`: per group (embeddings, each transformer layer, the heads) the relative change |after − before| / |before|, the cosine, and the weights' RMS. For a perturbation that is source → output, so you see what a sigma did (σ 0.05 on a micro moved its layers ~75%: the RMS of its weights is ~0.07). For a merge it is A → B, how far apart the parents are; the result sits alpha of the way along. The table draws it as a bar per group, the finished job's page larger, with a reading in words whose scale is rough, not ear-validated. Checkpoints made before this have no `movement`.

**Picking checkpoints.** Type part of a model's name, label, dataset or sig and pick it from the list (most recent first), then pick one of its checkpoints; or pick a merged or perturbed file from the same list, to perturb a merge or merge a perturbation. Each pick shows the model's shape and size. The *perturb* and *merge* links next to a checkpoint on an XP page open the form with it as the source (A, for a merge). The name is suggested from the sources and settings (`e16bf504-150_0444687c-last_slerp_a0.3`, the convention of the older folders) until you type your own.

- *merge*: *linear* averages each weight; *slerp* follows the arc between the two weight vectors, which keeps their norm. *alpha* 0 is all A, 1 is all B.
- *perturb*: *sigma* is the noise scale (0.005 to 0.02 is gentle, 0.05 and up is damage), on all weights, the attention layers, their projections only, the output heads, or the keys containing a fragment you type. The same seed gives the same noise.

**Preview.** It checks, before anything is loaded:

- **Shape.** A merge needs both models to have the same width, depth, feed-forward size, codebooks and conditioning: otherwise the helper fails after loading gigabytes. The shape is read from each XP's `.hydra/config.yaml` (the `transformer_lm` block); for a merged or perturbed file, from the XP its sidecar names as source, followed back through earlier merges. When neither works (a loose file whose sidecar was overwritten, or a source that was moved), the preview says the shape is unknown and lets it through. Different numbers of attention heads only warn: the weights have the same shape, but each head of the result mixes unrelated parts of the two models.
- **Name.** The output folder must not exist yet; the page never overwrites a checkpoint.
- **Memory.** It loads whole checkpoints on the CPU, optimizer state included: an XP's last checkpoint can be three times the weights alone (2.4 GB for `e16bf504`). The preview warns when that is more than the free memory.
- **Runtime.** It needs torch, so the AudioCraft runtime (`.venv-musicgen`).

**The button** is accepted only for the form that was previewed, within 30 minutes, and while no other merge or perturb runs. The job writes `artifacts/derive_jobs/<id>.log` and `<id>_status.json`; its page shows the log and the output. Merging two micro checkpoints took about two seconds; bigger models take longer, mostly loading and hashing the files for the sidecar.

**The table** lists every merged and perturbed checkpoint found under `artifacts/checkpoints`, `artifacts/sessions`, `merged_checkpoints` and a workspace's `merged_checkpoints/`, newest first: its sources (an XP and epoch, or a path), the settings, shape, size, and the XPs warm-started from it. A child is found by the file's path, or, for a file moved since, by its hash in the child's `macchina_ricordi_train_metadata.json`. Older outputs were sometimes written side by side into one folder (`artifacts/checkpoints/*.th`), where each write replaced the previous sidecar: only the last one is described, and the others are counted as *unrecorded*. The Atelier's *path or URI* list offers all of these files, and portable bundles under
`models/` appear in the model pickers alongside the XPs.

## Datasets

The Corpus page lists every dataset, the built ones in `datasets/` and the frozen legacy workspaces (`.workspace*`, `.macchina_ricordi*` and older dot-folders): train and valid clips, hours of audio, clip lengths, the folder it was prepared from, and how many XPs trained on it. A source marked **gone** no longer exists (many were prepared from `/tmp`): that workspace's `data/` is now the only copy of its clips.

**A dataset's page** shows:

- a histogram of the train clip lengths, and for each segment length a recipe trains at, the share of clips shorter than it. Those clips get padded with silence, and the model learns that silence. Clips within 0.05 s of the segment count as full: decoded slices land a few milliseconds short.
- the sources the clips come from, with their share of the train split. A clip's source is the part of its name before `__` (the naming `data corpus` and the older build scripts use), else its subfolder. Click a source to list only its clips.
- the XPs trained on it, and every clip, playable.
- its saved weights, and computing new ones (below).

**A new corpus** (**+ New corpus**) takes source folders from the library, meaning every folder under `library/`, `examples/sample_data` and `training_datasets`, and builds a new dataset with one command you could type yourself, `data build <name> <sources>…`, into `datasets/<name>/`:

1. It gathers their audio into `datasets/<name>/clips`. With *whole files*, files are hard-linked (no extra disk) and keep their format; training crops random segments from them. With *slices*, each file is cut into FLAC slices of the chosen length, 32 kHz mono; a tail shorter than the *shortest clip kept* (by default the slice length) is dropped. Clips are named `<source>__<file>[__<n>]`. `corpus.json` next to `clips/` records the recipe and where every clip came from. The same file reached twice (a hard link, or the same name and size in two folders) is used once: several library folders (`mega*`, `five_way`…) are assembled from the others, and the library hides them unless you ask.
2. It splits the clips into train and valid (hard links: no extra disk) and writes the manifests and `dataset.json`. Any recipe can train on the result, at its own segment length. Untick *split it into a dataset* to only gather the clips, with `data corpus`, into `training_datasets/<name>/clips`.

**Finding folders in the library.** Each folder shows the artists, albums, years and genres read from its files' tags (album artist over track artist, so a compilation reads as one), its last change (the newest of the folder and its audio files, which is when it was added for a downloaded album), and the workspaces already prepared from it. A `*` after the artist means some files carry no artist tag.

- Click a column to sort by it (folder, artist, year, files, audio, modified, prepared into); the sort is remembered.
- The search box matches folder names, artists, albums and genres. The *artists* list, or an artist name in the table, shows only that artist's folders. Tags spell one artist several ways (`DJ Lostboi` and `dj lostboi`, `Kali Malone featuring …`), so an artist matches every tag that holds its name as whole words, whatever the case.
- *never prepared* hides the folders a workspace was already prepared from.
- **tick the N shown** ticks everything the filter shows, for instance an artist's whole discography; **untick all** clears the selection; the ✕ on a picked source unticks just that one. Ticks are kept when you leave the page.
- **▸** before a folder lists its files, with their tags, and plays them. m4a files play in most browsers; formats the browser cannot play (aiff, wma) are listed but greyed. Untick a file there to leave it out of the corpus (`--exclude`); the folder's file count then reads `8/9`, and *keep them all* undoes it. Left-out files are remembered even while their folder is unticked, and only sent for ticked folders.
- **Duplicates.** *same recordings as …* under a folder names the folders that hold the same recordings, typically one album downloaded twice (mp3 and FLAC); click it to show that folder. In a listed folder, **≈** marks each such file (hover for where the other copy is). The check compares titles, lengths within 2 s, and the audio's loudness envelope, so two unrelated clips that merely share a name and length (`set_01.wav` in two folders) are not taken for one; assembled folders are left out of it. Details in [MusicGen](musicgen.md#2-build-a-dataset).
- **Selections** name the ticked folders and the files left out: type a name and **save**, pick one and **load** (it replaces the ticks and turns on *ticked only*). The list also holds the sources of every corpus built by `data corpus`, read from its `corpus.json`; loading one also sets its slicing, to rebuild it differently. Saved selections live in `artifacts/corpus_selections.json`; folders that no longer exist are counted as gone and skipped on load.

The preview gives the per-source table (files, audio, clips, each source's share, files left out by hand), what is skipped and why, the probable duplicates among the ticked folders, and the exact commands. *Leave probable duplicates out* (on by default) keeps one copy of each (`--skip-duplicates`: lossless over lossy, then the larger file); with it off, the preview warns that they would count twice and lists them. For each segment length a recipe trains at, it shows the share of clips shorter than it, which would be padded with silence (slices of 10 s suit a 10 s recipe, but every one of them would be padded at 15 s). It warns when clips would be padded at every one of those lengths, and when one source is more than half the clips: training draws each clip equally often, so a source's share of the clips is its share of training.

A dataset is never rebuilt: a name that exists is refused, so a different cut gets a new name. The workspaces with a leading dot were prepared before datasets existed; they are frozen and listed beside the datasets.

**Jobs** run detached, as one shell command, so they finish even if you close the page or stop the GUI. Each leaves `artifacts/data_jobs/<id>.log` and `<id>_status.json`; the log ends with the exit code, which is how the page knows the outcome. One data job runs at a time. Reading the library the first time takes several seconds (durations and tags are read from every file's header, and m4a files need ffprobe). The results are kept in `artifacts/gui_cache/audio_probe.json`, keyed on each file's path, size and modification time, so later scans, including after a restart, only read new or changed files. It also keeps the loudness envelopes the duplicate check decodes (with ffmpeg, only for files whose title and length already match), which is most of a cold scan's time. Deleting that file is safe; the next scan rebuilds it.

**Weights** give every train clip a weight; a run that uses them draws a clip in proportion to its weight. Validation is never weighted. Weights are saved under a name, as an overlay: a weighted copy of the train manifest in `artifacts/weights/<workspace>/<name>/`. The workspace itself is never written, so it can carry several weightings, and a run uses one only when you pick it in the Atelier (or `train run --weights <name>`).

- **Compute** runs `data rebalance --plan-out` as a job. It clusters the clips by timbre (MFCC and spectral statistics, k clusters). Without generations the weights flatten the clusters toward equal shares (*inverse-share*). If you tick generation folders of models trained on the workspace (their runs and sweeps are listed), they favour the clusters those models under-produce (*coverage*). Weights stay within [1/cap, cap] and average 1.
- The result lists each cluster with its share, its weight (and its share of the generations), and four example clips, the ones nearest the cluster's centre, so you can hear what a cluster is before weighting it.
- **Save these weights** keeps them under the name you type (`flat_a05`, `cov_r1`, ...). It is refused if the manifest changed since the weights were computed, or if the name is taken. Saving is fine while the workspace trains, since nothing in the workspace changes.
- Each saved weighting is listed with its settings, its clusters, and **Delete**. A run trained with it gets `_rw<digest>` in its auto label, so a new XP. Saved weights go *stale* if the workspace's manifest changes; the Atelier then greys them out.
- **Mix by source** gives each source folder a share of the train draws instead. Type relative numbers (3 and 1 mean three quarters and a quarter); the shares update as you type. **Save mix** writes the overlay at once, with nothing computed, and it is listed with its shares like the others. Every source with train clips needs a number, and a clip's source is the part of its name before `__`.
- Judge a rebalanced model by what it generates, not by valid ce. alpha 0.5 is the value that worked; 1.0 overshot (see [MusicGen](musicgen.md#rebalance-the-training-distribution-anti-wells)).

## Housekeeping

The Réserve page starts with the disk: how much `.dora/xps` holds, split into final `checkpoint.th` files, epoch checkpoints (`checkpoint_N.th`), other `.th` files (merges, snapshots) and everything else (samples, logs, history), against the free space left. Below it is every folder in `.dora/xps`, sortable, with its size, its checkpoints, and how many files outside it mention it.

In the epoch column, **★** marks the checkpoint whose epoch has the best valid ce among those still on disk, and **↗** one that something outside the XP refers to (hover for what). Folders that are not XPs show as *empty folder* (a leftover holding only a cleanup report) or *not an XP* (for instance a folder of morph checkpoints).

Tick folders (click a row, or use the quick selections: *unreferenced*, *no checkpoint*, *empty folders*), pick an action, and **Preview**. The preview is a table of exactly what would go; the commit button under it applies that plan and nothing else. If anything on disk changed in between, the commit is refused and you preview again.

| action | what it runs | deletes | report |
|---|---|---|---|
| **Epoch checkpoints** | `tidy clean` | `checkpoint_N.th` beyond the last N per XP (default 1), except the protected ones | `checkpoint_cleanup_report.json` in each XP that lost a file |
| **Whole XPs** | `tidy prune`, limited to the ticked folders | the ticked folders, entirely | `.dora/prune_report_<ts>.json` |
| **Archive** | `tidy archive` | nothing: copies to another disk | `.dora/checkpoint_archive_report_<ts>.json` |

What is protected:

- **Epoch checkpoints something refers to are always kept.** A reference is any mention of `xps/<sig>/checkpoint_N.th` in the project's records: a child XP's metadata (its warm start), a merge or perturbation, a generation run, a sweep, a stream or listening session, a workspace log, a note under `research/` or `docs/`. Mentions inside the XP's own folder, and cleanup reports, do not count. The same protection is the default of `tidy clean` in the terminal (`--no-protect-referenced` turns it off).
- **Each XP's best epoch checkpoint** (★), unless you untick *also keep each XP's best one*. `tidy clean --keep-best` does the same.
- **Whole XPs that something refers to are kept** unless you tick *also delete XPs that something refers to*. Most XPs are referenced (every generation run names its model), so the plan says what refers to each one: *parent of* a child XP, *mentioned in* a run or a note. Deleting a parent does not break its children, which keep their own weights, but their warm start can no longer be reproduced, and runs made with a deleted XP can no longer be replayed.
- **XPs with a keep marker** (`.keep` or `logs/keep.json` in the folder) are never deleted, as with `prune`.
- **XPs being trained** are left out of every action: an active run in the Atelier, or any folder whose history, logs or checkpoints were written in the last 15 minutes. Their rows are greyed out and cannot be ticked.

`checkpoint.th` is never deleted on its own: it goes only with its whole XP. Most of the space is usually there, so the epoch cleanup frees little once it has been run; deleting whole XPs, after archiving them if they might be wanted again, is what frees real space.

**Archiving** copies each checkpoint, with its metadata, history and Hydra config, into `<destination>/<sig>/versions/<sha256>/`, verified by SHA-256, and records it in `<destination>/<sig>/archive_index.json`, so a second run skips what is already there. The destination defaults to `MACCHINA_RICORDI_ARCHIVE` from `~/.config/macchina_ricordi/checkpoint-archive.env` (the [systemd timer's](musicgen-admin.md#run-automatically-with-systemd) config) when that file exists. The page refuses a destination on the same disk as `.dora`, so an unmounted drive cannot quietly fill the system disk; mount the archive disk and preview again. The copy runs inside the GUI process, with progress on the page. Keep the GUI running until it finishes: stopped halfway, the checkpoint being copied is abandoned (its `.staging-…` folder stays at the destination), the ones already copied are kept, and the next archive run copies the rest.

## The covers

Each cover is generated from the XP, after the [Éditions Sans Soleil](https://sans-soleil.com/catalogue/) book covers: two inks (a ground and one spot colour, picked from the sig), wide capitals for the label at the top and the workspace at the bottom, and a double hairline down the spine. The smeared stroke is the XP's own validation curve. Perturbed XPs get glitch-shifted bands, merges are folded in half, and warm-starts carry a faint echo of themselves. The same sig always draws the same cover.

## How parents are found

The same way as everywhere else: the GUI reads the zoo ([models.md](models.md#the-zoo-one-graph)), so its parents,
children and families are the ones `model show`, `model lineage` and the catalog print. When a checkpoint's record
carries both a `merged_from` and a `perturbed_from` (a perturbation of a merge-derived checkpoint inherits the merge
record), the more recently created operation is the direct parent; `model lineage` used to prefer the merge, and no
longer does. The model page lists every session that played the model, of any kind (generations, sweeps, streams,
duets, variations), and the index counts them.


