"""Download the published starter bundle; no HF account or extra Python packages needed."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "a9220dd9-ep200"
REVISION = "afd6bcc2f25439bb6b0caee58dbbd603423eb4a5"
BASE_URL = f"https://huggingface.co/AvantBras/{MODEL}/resolve/{REVISION}"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / MODEL
SHA256 = {
    "model.safetensors": "b33a81ea8fd931b8c82112248c230bd26f7d2f75bffbbc5323e1cf103664a1fc",
    "config.json": "c5493817b462c02bf3a684dcd50b5b0fcaf35d8a0ab19b2dd17e7ed20358eaa3",
    "README.md": "243729bfeeebeb7c89a58a5bbf7ae05fb6526d19cee1cf1c6f47e964b489f2f2",
    "LICENSE": "ed82332e21569fefcaaec8a1d3e3395afd458f54722227c1ac27d1e62b00bf8e",
}


def verify(folder: Path) -> None:
    for name, expected in SHA256.items():
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe bundle file: {name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}; existing files will not be overwritten")


def download(output: Path = DEFAULT_OUTPUT) -> Path:
    output = output.expanduser().absolute()
    if output.is_symlink():
        raise ValueError("Refusing a symlink destination")
    if output.exists():
        verify(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{MODEL}-", dir=output.parent) as temporary:
        stage = Path(temporary) / MODEL
        stage.mkdir()
        for name in SHA256:
            print(f"Downloading {name}…", flush=True)
            request = urllib.request.Request(
                f"{BASE_URL}/{name}?download=true", headers={"User-Agent": "macchina-ricordi"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, (stage / name).open("wb") as stream:
                shutil.copyfileobj(response, stream)
        verify(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Destination appeared during download; refusing to replace it")
        stage.rename(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Bundle destination (default: checkout/models/a9220dd9-ep200)",
    )
    args = parser.parse_args()
    try:
        output = download(args.output_dir)
    except urllib.error.HTTPError as exc:
        parser.exit(1, f"Download failed: HTTP {exc.code}. Try again later.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(f"Ready: {output}\nLicence: CC BY-SA 4.0. Next: follow the starter-model section in README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
