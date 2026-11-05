"""Play one trained model on a CPU, with nothing else installed.

Two files and two environments:

- ``appliance.export``   an experiment folder -> ``.nrsrc`` source (AudioCraft env)
- ``appliance.pack``     ``.nrsrc`` -> ``.nrbox``, verified and probed (torch + onnx)
- ``python -m appliance`` plays a ``.nrbox`` with onnxruntime, numpy, sounddevice

``player`` holds the loop against a three-method backend; ``runtime_onnx`` is the
backend that ships and ``lm.TorchBackend`` the reference it is checked against.
See ``docs/appliance.md``.
"""

from __future__ import annotations

__all__ = ["box", "engine", "loader", "player", "spec"]
