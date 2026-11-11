# Security and trust boundaries

macchina-ricordi is a local music and training toolbox for a trusted machine. It is not a
sandbox for arbitrary models, nor an authenticated multi-user or internet-facing service.

## Local web interface

The GUI binds to loopback only. Host checks and a custom request header protect write actions
from ordinary cross-site browser requests; they do **not** authenticate local users or processes.
A local process can invoke the same generation, training, stop and housekeeping/deletion actions
as the page. Responses and experiment metadata can include local filesystem paths.

Do not expose the GUI through a public reverse proxy or tunnel. Use it only where local callers
are trusted. Closing the browser does not stop detached training or stream jobs; stop them explicitly.

## Models and checkpoints

Prefer the published Safetensors bundle and the pinned/checksummed starter download helper.
Checksums prove that files match a chosen release, not that every model or its publisher is safe.
Bundle validation rejects unsupported configuration and malformed tensor layouts, but is not a
resource sandbox: loading large models still requires suitable memory and compute.

Legacy AudioCraft `.th` imports, exports, merges and warm starts can use Python pickle loading.
**Only load `.th` checkpoints from sources you trust:** opening a malicious pickle can execute code.
A filename, model card or file extension does not make such a checkpoint safe.

## Optional neural captioners

Captioners are opt-in and require the `captioner-witness` dependencies. Their current default is
`trust_remote_code=True`: running one can execute Python from its Hugging Face model repository,
including when you choose a different `--captioner-model-id`. Use only publishers and code you trust.

For captioner commands, pass `--no-captioner-trust-remote-code` to disallow that remote Python;
models that require custom code may then refuse to load. This does not sandbox model inference or
its dependencies. Plain DSP listening and the starter-model downloader do not enable captioners.

## Runtime dependency maintenance

The Pillow findings affecting the former MusicGen runtime lock were resolved by removing the unused
Gradio demo dependency and locking **Pillow 13.0.0**. The runtime does not install AudioCraft's demo
dependencies. Reintroducing them requires checking their constraints against the patched runtime.

A lockfile check verifies dependency resolution, not GPU compatibility. Runtime dependency updates
also need a runtime health check, generation with a trained model, and a small real training run on
the target GPU. Recheck dependency advisories whenever the runtime lock changes; a clean audit is
not a guarantee that the runtime is a sandbox or that untrusted input is safe.

## Sharing reports and reporting problems

Review logs, sidecars and metadata before sharing: local dataset paths and experiment details may
be recorded intentionally. Never attach credentials, `.env` files, private datasets or raw training
checkpoints to a public issue. For a security problem, use the repository's private vulnerability
reporting feature if enabled; otherwise arrange a private channel with the maintainer before posting
sensitive details. There is no automatic guarantee of a private reporting channel or response time.
