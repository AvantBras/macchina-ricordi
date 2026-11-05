"""The shipped backend: two ONNX Runtime sessions and nothing else.

ONNX Runtime is a 62 MB install where torch is 711 MB, and it is the faster of
the two here — 0.52 s against 1.05 s for the language-model half of a round on
one core. The graph takes the KV cache as ordinary input and output tensors, so
the cache is concatenated each step rather than written in place; that copying
is measured and already inside those numbers.
"""

from __future__ import annotations

import numpy as np

from .box import Box
from .spec import ModelSpec


class OnnxBackend:
    def __init__(self, box: Box, *, threads: int | None = None, memory_arena: bool = False) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        # ORT's workers busy-wait between runs by default. With four threads that
        # kept four cores at 100% for a 16% faster round (0.41 s vs 0.49 s, using
        # 1.7 cores when off) -- the wrong trade for a laptop running all night.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # The arena keeps every allocation at its high-water mark. The full-window
        # decode is a large transient, so with the arena on a running player sat at
        # 1345 MB resident; off, it settles at 447 MB and peaks at 853 MB mid-decode.
        options.enable_cpu_mem_arena = memory_arena
        options.enable_mem_pattern = memory_arena
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.spec: ModelSpec = box.spec
        self._lm = ort.InferenceSession(box.lm, options, providers=providers)
        self._decoder = ort.InferenceSession(box.decoder, options, providers=providers)
        self._encoder = ort.InferenceSession(box.encoder, options, providers=providers) if box.encoder else None

    @property
    def can_encode(self) -> bool:
        return self._encoder is not None

    def start(self, batch: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.spec
        empty = np.zeros((s.num_layers, batch, s.num_heads, 0, s.head_dim), dtype=np.float32)
        return empty, empty

    def step(self, tokens: np.ndarray, offset: int, state) -> tuple[np.ndarray, tuple]:
        past_k, past_v = state
        logits, past_k, past_v = self._lm.run(
            None,
            {
                "tokens": np.ascontiguousarray(tokens, dtype=np.int64),
                "offset": np.asarray(offset, dtype=np.int64),
                "past_k": past_k,
                "past_v": past_v,
            },
        )
        return logits, (past_k, past_v)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        audio = self._decoder.run(None, {"codes": np.ascontiguousarray(codes, dtype=np.int64)})[0]
        return audio[0].astype(np.float32, copy=False)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Mono float32 samples -> ``[1, n_q, frames]`` codes."""
        if self._encoder is None:
            raise RuntimeError("this box was packed without an encoder")
        codes = self._encoder.run(None, {"audio": np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1)})[0]
        return codes.astype(np.int64, copy=False)
