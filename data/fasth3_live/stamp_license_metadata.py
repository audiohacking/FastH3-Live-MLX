# SPDX-License-Identifier: Apache-2.0
"""Stamp MiniMax H3 licence and modification notices into a safetensors header.

Section III.2 of the MiniMax H3 Community License requires that "any modified
files carry prominent notices stating that you have modified such files". A
README says so about the repository; this says so inside the weight file
itself, so the notice survives being copied out of that repository.

Rewrites the file: the safetensors header sits at the front, so growing it
means writing a new copy. Tensor data is streamed through unchanged, and the
result is verified byte-for-byte against the source before the original is
touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time

NOTICE = ("MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
          "Copyright (c) 2026 MiniMax. All Rights Reserved.")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modification", required=True,
                   help="what was changed, in one sentence")
    p.add_argument("--base-model", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--derived-from", required=True,
                   help="the immediate upstream checkpoint this was built from")
    p.add_argument("--modified-by", default="convert_fasth3_to_comfyui.py",
                   help="the script that actually produced this file. It goes into the "
                        "provenance the licence notice carries, so name the real one")
    p.add_argument("--verify", action="store_true", help="hash both payloads afterwards")
    args = p.parse_args()

    header, data_start = read_header(args.src)
    meta = dict(header.pop("__metadata__", {}))
    meta.update({
        "license": "MiniMax H3 Community License Agreement",
        "license_notice": NOTICE,
        "base_model": args.base_model,
        "derived_from": args.derived_from,
        "modified": "true",
        "modification_notice": args.modification,
        "modified_by": f"FastH3 Live ({args.modified_by})",
    })
    header["__metadata__"] = meta

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)

    total = os.path.getsize(args.src) - data_start
    log(f"payload {total / 1e9:.2f} GB; header {len(blob)} bytes")

    src_hash, out_hash = hashlib.sha256(), hashlib.sha256()
    with open(args.src, "rb") as fin, open(args.out, "wb") as fout:
        fin.seek(data_start)
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        done = 0
        while chunk := fin.read(1 << 24):
            fout.write(chunk)
            src_hash.update(chunk)
            done += len(chunk)
            if done % (1 << 30) < (1 << 24):
                log(f"  {done / 1e9:5.1f} / {total / 1e9:.1f} GB")

    if args.verify:
        _, new_start = read_header(args.out)
        with open(args.out, "rb") as f:
            f.seek(new_start)
            while chunk := f.read(1 << 24):
                out_hash.update(chunk)
        ok = src_hash.hexdigest() == out_hash.hexdigest()
        log(f"payload sha256 {'MATCHES' if ok else 'DIFFERS'}: {out_hash.hexdigest()[:32]}...")
        if not ok:
            raise SystemExit("payload changed during rewrite -- refusing to continue")

    log(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")
    for k, v in meta.items():
        print(f"    {k}: {str(v)[:96]}")


if __name__ == "__main__":
    main()
