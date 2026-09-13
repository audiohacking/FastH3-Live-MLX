# Overnight handoff — FastH3 Live noise (2026-09-13)

## GPU status

**Idle** after smokes (no `liveserver` / long jobs). INT8 re-quant may be
running on CPU/disk (`/tmp/fasth3-smoke/requant-int8.log`).

## Root cause (fixed)

Diffusers→native convert used **chunked** `cat([Q,K,V])`. Official MiniMax-H3 /
h3.c stores DiT QKV **interleaved per head**
`[Q_h0,K_h0,V_h0, Q_h1,…]`. Wrong layout → muddy frames (std≈6). Same class of
bug as h3.c’s historical “identity interpretation” noise.

Fix: `scripts/convert_fasth3_diffusers_to_native.py` `_fuse_qkv` + in-place
patch of `models/MiniMax-H3-FastH3/FL2VA/transformer` (52 QKV tensors).

## Evidence

| Clip | std (mid) | Picture |
|------|-----------|---------|
| Broken student (chunked QKV) | ~5.5–6 | Mud / tiles |
| Base FL2VA `--steps 4` control | ~50 | OK structure |
| **Fixed student** interleaved QKV | **~50** | **Red ball on wood** |

Not INT8, not retime, not Metal4 auto-INT8 (true BF16 was also muddy until fuse fix).

## Morning Live test (ready)

INT8 re-quant from interleaved QKV done; 22f INT8 smoke std≈50 (picture OK).

```bash
export H3_FORCE_TENSOROPS=1 H3_AV=$PWD/scripts/h3-av
/Users/moysa/.pyenv/versions/3.11.9/bin/python3 liveserver.py \
  --model-dir "$(pwd)/models/MiniMax-H3-FastH3-INT8" \
  --host 0.0.0.0 --port 9000 --frames 124 --fps 1.7 --prefill 1 -v
# open VLC → http://127.0.0.1:9000/  (gen starts only when a viewer connects)
```

## Still open

- Live sustain / play-fps tuning with real picture
- Homebrew ffmpeg dyld (`libx265`) — keep `H3_AV=scripts/h3-av`
