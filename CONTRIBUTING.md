# Contributing

This is a personal, experimental music project. Small, focused pull requests are welcome;
please discuss substantial changes before investing time in them. There is no promise of a
review or support deadline.

## Propose changes, do not push to main

Work on a branch or a fork and open a pull request. Publishing the repository does not give
visitors write access. The maintainer decides what lands; please do not request write access
just to submit a contribution.

Run the ordinary CPU checks:

```bash
uv sync --locked --extra dev
make test
make lint
```

`make lint` checks for likely mistakes, not formatting preferences. `make format` is optional.
Normal contributions do not require a GPU or model downloads. Changes to the GPU runtime need
separate local validation; see [runtime maintenance](docs/musicgen.md#maintaining-the-runtime).

The prepared main-branch policy requires a pull request and passing CPU CI, forbids force-pushes
and branch deletion, and applies to the maintainer too. It deliberately does not require a
second person's approval. Its activation status and publication checklist are documented in
[repository protection](docs/repository-protection.md); committing a policy file does not enable
GitHub protection by itself.

Report vulnerabilities using the guidance in [SECURITY.md](SECURITY.md), not a public issue
containing sensitive details.
