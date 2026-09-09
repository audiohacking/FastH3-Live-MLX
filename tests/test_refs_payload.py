import unittest
from pathlib import Path

from h3_backend import parse_refs_payload


class RefPayloadTests(unittest.TestCase):
    def test_ref_size_defaults_to_max(self) -> None:
        items = parse_refs_payload(
            [{"kind": "image", "path": "/tmp/a.png", "name": "a"}]
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].ref_size, "max")

    def test_ref_size_match(self) -> None:
        items = parse_refs_payload(
            [{"kind": "image", "path": "/tmp/a.png", "name": "a", "ref_size": "match"}]
        )
        self.assertEqual(items[0].ref_size, "match")
