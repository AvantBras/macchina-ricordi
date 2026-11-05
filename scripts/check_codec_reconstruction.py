"""A/B reconstruction check for a trained compression (codec) XP.

Encodes and decodes clips through the codec checkpoint and writes
``<stem>_orig.wav`` / ``<stem>_recon.wav`` pairs plus a small metrics summary,
so the codec can be judged by ear before any LM is trained on its tokens.

Run inside the musicgen venv, from the .audiocraft directory:

    .venv-musicgen/bin/python scripts/check_codec_reconstruction.py \
        --checkpoint .dora/xps/<sig>/checkpoint.th \
        --audio <clip.mp3> [<clip2.wav> ...] \
        --output-dir artifacts/codec_checks/<tag>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".audiocraft"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the compression XP checkpoint.th")
    parser.add_argument("--audio", nargs="+", required=True, help="Input clips to round-trip")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seconds", type=float, default=10.0, help="Max seconds per clip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from audiocraft.data.audio import audio_read, audio_write
    from audiocraft.solvers.compression import CompressionSolver

    model = CompressionSolver.model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    print(
        f"codec: frame_rate={model.frame_rate} Hz, n_q={model.num_codebooks}, "
        f"cardinality={model.cardinality}, sr={model.sample_rate}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in args.audio:
        wav, sr = audio_read(source, 0, args.seconds, pad=False)
        from audiocraft.data.audio_utils import convert_audio

        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        with torch.no_grad():
            codes, scale = model.encode(wav[None].to(args.device))
            recon = model.decode(codes, scale)[0].cpu()

        n = min(wav.shape[-1], recon.shape[-1])
        wav_t, recon_t = wav[..., :n], recon[..., :n]
        # Waveform-domain SNR is meaningless for a spectral+adversarial codec
        # (phase is unconstrained), so judge with a log-mel L1 distance — the
        # same family of signal the training losses optimize. Lower is better;
        # well-trained EnCodec-class codecs land around 0.3-0.5 here.
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=model.sample_rate, n_fft=1024, hop_length=256, n_mels=64
        ).to(wav_t.device)
        log_orig = torch.log10(mel(wav_t) + 1e-5)
        log_recon = torch.log10(mel(recon_t) + 1e-5)
        mel_l1 = torch.mean(torch.abs(log_orig - log_recon)).item()
        gain_db = (
            10
            * torch.log10(torch.sum(recon_t.double() ** 2) / max(torch.sum(wav_t.double() ** 2).item(), 1e-12)).item()
        )

        stem = Path(source).stem[:60]
        audio_write(out_dir / f"{stem}_orig", wav_t, model.sample_rate, strategy="clip")
        audio_write(out_dir / f"{stem}_recon", recon_t, model.sample_rate, strategy="clip")
        print(
            f"{stem}: log-mel L1 {mel_l1:5.3f}  gain {gain_db:+5.1f} dB  "
            f"tokens/s {model.frame_rate * model.num_codebooks:.0f}"
        )

    print(f"wrote pairs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
