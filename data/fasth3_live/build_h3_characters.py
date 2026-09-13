# SPDX-License-Identifier: Apache-2.0
"""Parse the community H3 known-characters index into the two name pools the
stream draws from.

Only the ``good`` (verified usable) section is taken -- the ``onthefence`` and
``bad`` rows exist precisely because the model does not reproduce them
reliably, which is the opposite of what an unattended stream wants.

Source: https://huggingface.co/datasets/malcolmrey/various
        h3-center/known-characters/INDEX.md
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request

INDEX_URL = ("https://huggingface.co/datasets/malcolmrey/various/raw/main/"
             "h3-center/known-characters/INDEX.md")

# Reported by the community as the strongest text-only recognitions.
CURATED = [
    # real people, listed as themselves
    "Tom Hanks", "Samuel L. Jackson",
    # Chris Evans is indexed by character, not by his own name
    "Steve Rogers", "Ransom Drysdale",
    # Star Trek -- reported as the strongest franchise recognition
    "Spock", "Jean-Luc Picard", "Data", "William Riker", "Worf", "Geordi La Forge",
    # Seinfeld
    "Cosmo Kramer", "Jerry Seinfeld", "George Costanza", "Elaine Benes",
]


def parse(text: str) -> list[str]:
    names: list[str] = []
    in_good = False
    for line in text.splitlines():
        if line.startswith("## Folder:"):
            in_good = "`good`" in line
            continue
        if not in_good or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 2 or cells[0].startswith(":---"):
            continue
        m = re.match(r"\*\*(.+?)\*\*", cells[0])
        if m:
            names.append(m.group(1).strip())
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", help="local INDEX.md instead of downloading")
    p.add_argument("--out", default="h3_characters.json")
    args = p.parse_args()

    if args.index:
        text = open(args.index, encoding="utf-8").read()
    else:
        with urllib.request.urlopen(INDEX_URL, timeout=60) as r:
            text = r.read().decode("utf-8")

    full = parse(text)
    known = set(full)
    missing = [c for c in CURATED if c not in known]
    curated = [c for c in CURATED if c in known]

    json.dump({"curated": curated, "full": full}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"full pool     : {len(full)} characters")
    print(f"curated pool  : {len(curated)} kept")
    if missing:
        print(f"curated but not in the verified index (dropped): {missing}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
