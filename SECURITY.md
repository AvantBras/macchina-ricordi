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

## Known runtime dependency limitation

The current `musicgen/uv.lock` still contains **Pillow 12.3.0**, for which the dependency audit
reports image-decoder/encoder vulnerabilities fixed in Pillow 13.0.0. The pinned Gradio 6.30.0
requires `pillow<13.0`, so the patched version cannot currently be selected within the declared
runtime constraints. The latest checked Gradio 6.31.0 has the same upper bound.

This is an unresolved dependency finding, not evidence that an exploit is reachable through the
starter-model workflow. Do not process untrusted images in that runtime or treat it as a hardened
service. Resolving it requires a compatible Gradio update or a separately tested change to the demo
dependencies; overriding the constraint without testing is not a fix. Recheck this note whenever the
runtime dependencies change.

## Sharing reports and reporting problems

Review logs, sidecars and metadata before sharing: local dataset paths and experiment details may
be recorded intentionally. Never attach credentials, `.env` files, private datasets or raw training
checkpoints to a public issue. For a security problem, use the repository's private vulnerability
reporting feature if enabled; otherwise arrange a private channel with the maintainer before posting
sensitive details. There is no automatic guarantee of a private reporting channel or response time.
