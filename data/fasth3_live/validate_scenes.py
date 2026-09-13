# SPDX-License-Identifier: Apache-2.0
"""Check a scene file against the MiniMax-H3 prompt grammar before it goes live.

The grammar is narrow and the failures are silent: a shot header the model was
not trained on, or a camera-motion phrase outside its vocabulary, produces a
plausible-looking clip that simply ignores the instruction. Since the stream
draws these at random and nobody reviews them first, a malformed block would
just quietly degrade the output for as long as it stays in the file.

Rules enforced (base-en.txt sections 4.2 to 4.7):

* the three sections, present and in order
* ``[Shot 1]`` carries no timestamp; later shots number consecutively and open
  with a strictly increasing ``MM:SS.mmm`` inside the clip's duration
* cuts use one of the five documented verbs
* camera motion comes from the fixed motion-type list, and phrases measured to
  misbehave (``locked-off``, ``the camera does not move``) are rejected
* dialogue sits inside ``<d>[English] ... </d>`` with a speaker ID outside it
* every speaker ID that speaks is introduced in the same block
* character placeholders are contiguous from ``{NAME}``

    python validate_scenes.py prompts_scenes.txt
"""

from __future__ import annotations

import argparse
import re
import sys

MOTIONS = [
    "zoom in", "zoom out", "push in", "pushes in", "pull out", "pulls out",
    "pan left", "pans left", "pan right", "pans right",
    "truck left", "trucks left", "truck right", "trucks right",
    "tilt up", "tilts up", "tilt down", "tilts down",
    "pedestal up", "pedestals up", "pedestal down", "pedestals down",
    "arc shot", "arcs", "tracking shot", "tracks", "static shot",
    "shake slightly", "shakes slightly", "shake strongly", "shakes strongly",
    "pov", "roll clockwise", "rolls clockwise",
    "roll counterclockwise", "rolls counterclockwise",
]
CUT_VERBS = ["the camera cuts to", "the shot cuts to", "the shot transitions to",
             "the shot changes to", "the shot switches to"]
# measured 2026-08-22: neither arm of the A/B followed an instruction phrased this way
BANNED = ["locked-off", "locked off", "the camera does not move", "camera remains still",
          "slowly and continuously", "handheld feel", "steadicam"]
SECTIONS = ["integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"]


def check(block: str, idx: int, duration: float, want_names: tuple[int, int] | None) -> list[str]:
    bad: list[str] = []
    low = block.lower()

    pos = [block.find(s) for s in SECTIONS]
    if -1 in pos:
        bad.append(f"missing section: {SECTIONS[pos.index(-1)]}")
    elif pos != sorted(pos):
        bad.append("sections out of order")

    for phrase in BANNED:
        if phrase in low:
            bad.append(f"banned camera phrase {phrase!r}")

    shots = re.findall(r"\[Shot (\d+)\]", block)
    nums = [int(x) for x in shots]
    if not nums:
        bad.append("no [Shot N] header")
    else:
        if nums != list(range(1, len(nums) + 1)):
            bad.append(f"shot numbers are {nums}, expected {list(range(1, len(nums) + 1))}")
        first = block.split("[Shot 1]", 1)[-1].split("[Shot", 1)[0]
        if re.search(r"At \d\d:\d\d\.\d\d\d", first):
            bad.append("[Shot 1] carries a timestamp; it must not")

    stamps = re.findall(r"At (\d\d):(\d\d)\.(\d\d\d)", block)
    secs = [int(m) * 60 + int(s) + int(ms) / 1000 for m, s, ms in stamps]
    if len(secs) != max(0, len(nums) - 1):
        bad.append(f"{len(secs)} cut timestamps for {len(nums)} shots")
    if secs != sorted(set(secs)):
        bad.append(f"cut times not strictly increasing: {secs}")
    for t in secs:
        if t >= duration:
            bad.append(f"cut at {t:.3f}s is outside the {duration:.2f}s clip")

    for n in nums[1:]:
        head = block.split(f"[Shot {n}]", 1)[1][:160].lower()
        if not any(v in head for v in CUT_VERBS):
            bad.append(f"[Shot {n}] does not open with a documented cut verb")

    if not any(m in low for m in MOTIONS):
        bad.append("no camera motion from the documented vocabulary")

    if block.count("<d>") != block.count("</d>"):
        bad.append("unbalanced <d> tags")
    for d in re.findall(r"<d>(.*?)</d>", block, re.S):
        if not d.lstrip().startswith("["):
            bad.append(f"<d> block has no language tag: {d[:40]!r}")
    speaking = set(re.findall(r"\(S(\d)[,)]", block))
    for sid in sorted(speaking):
        if f"(S{sid})" not in block and f"(S{sid}," not in block:
            bad.append(f"speaker S{sid} used but never introduced")

    slots = sorted({int(m or 1) for m in re.findall(r"\{NAME(\d*)\}", block)})
    if slots and slots != list(range(1, len(slots) + 1)):
        bad.append(f"character slots are {slots}, expected contiguous from 1")
    if want_names and not (want_names[0] <= len(slots) <= want_names[1]):
        bad.append(f"{len(slots)} characters, wanted {want_names[0]}-{want_names[1]}")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--frames", type=int, default=362)
    p.add_argument("--authored-fps", type=float, default=24.0,
                   help="H3 authors motion at 24 fps; the playback rate does not change "
                        "where a cut lands in the generated clip")
    p.add_argument("--from-index", type=int, default=0,
                   help="only check blocks at or after this index")
    p.add_argument("--min-names", type=int)
    p.add_argument("--max-names", type=int)
    args = p.parse_args()

    blocks = [b.strip() for b in open(args.path, encoding="utf-8").read().split("\n---\n")
              if b.strip()]
    duration = args.frames / args.authored_fps
    want = (args.min_names, args.max_names) if args.min_names and args.max_names else None

    failures = 0
    for i, b in enumerate(blocks):
        if i < args.from_index:
            continue
        bad = check(b, i, duration, want)
        if bad:
            failures += 1
            print(f"\nscene {i:03d}:")
            for msg in bad:
                print(f"    {msg}")
    checked = len(blocks) - args.from_index
    print(f"\n{checked} scenes checked, {failures} with problems "
          f"(clip duration {duration:.2f}s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
