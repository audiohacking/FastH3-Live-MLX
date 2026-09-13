# SPDX-License-Identifier: Apache-2.0
"""Minimal seek-based safetensors reader.

``safe_open`` mmaps the whole file, which Windows refuses for the ~21 GB
ComfyUI checkpoints while ComfyUI itself is resident (OSError 1455, "the paging
file is too small"). Reading single tensors by offset needs no mapping at all.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch

_NP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}
# dtypes numpy has no equivalent for: read as bytes, then reinterpret
_TORCH_VIEW = {
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class SafeTensorsFile:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.metadata = self.header.pop("__metadata__", {})
        self.data_start = 8 + n

    def keys(self):
        return self.header.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.header

    def shape(self, key: str) -> list[int]:
        return self.header[key]["shape"]

    def get_tensor(self, key: str) -> torch.Tensor:
        info = self.header[key]
        start, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + start)
            raw = f.read(end - start)
        dtype = info["dtype"]
        if dtype in _TORCH_VIEW:
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(_TORCH_VIEW[dtype])
        else:
            t = torch.from_numpy(np.frombuffer(bytearray(raw), dtype=_NP[dtype]).copy())
        return t.reshape(info["shape"])
