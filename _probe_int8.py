#!/usr/bin/env python3
"""Probe the Comfy-Org INT8 Ref2VA safetensors header via a raw HTTP range read."""
import json
import struct
import urllib.request

URL = (
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
)

req = urllib.request.Request(URL, headers={"Range": "bytes=0-4194303"})
with urllib.request.urlopen(req) as resp:
    raw = resp.read()

n = struct.unpack("<Q", raw[:8])[0]
print("header length:", n)
h = json.loads(raw[8 : 8 + n].decode())
print("metadata:", h.get("__metadata__"))
names = [k for k in h if k != "__metadata__"]
print("num tensors:", len(names))
for x in names[:14]:
    print("  ", x, h[x]["dtype"], h[x]["shape"])
weight = [x for x in names if x.endswith(".weight") and "scale" not in x]
scale = [x for x in names if x.endswith("scale")]
print("total", len(names), "| weight tensors", len(weight), "| scale tensors", len(scale))
print("--- sample late blocks ---")
for x in names:
    if x.startswith("blocks.49"):
        print(" ", x, h[x]["dtype"], h[x]["shape"])
print("--- non-block / non-adaln tensors ---")
for x in names:
    if not x.startswith(("blocks.", "adaln")):
        print(" ", x, h[x]["dtype"], h[x]["shape"])
