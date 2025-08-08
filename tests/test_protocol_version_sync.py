#!/usr/bin/env python3
"""
Protocol version synchronization tests for RIPER-Ω
Validates that protocol text, refresh function, and metadata are consistent.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from protocol import (
    get_complete_protocol,
    refresh_protocol,
    PROTOCOL_VERSION,
    get_protocol_metadata,
)

EXPECTED_VERSION = "2.6.1.1"
EXPECTED_DATE = "July 25, 2025"


class TestProtocolVersionSync(unittest.TestCase):
    def test_protocol_text_contains_correct_version_and_dates(self):
        full_text = get_complete_protocol()
        self.assertIn(EXPECTED_VERSION, full_text, "Protocol text must include the current version")
        self.assertIn(EXPECTED_DATE, full_text, "Protocol text must reflect the correct last updated/sync date")

    def test_refresh_protocol_reports_correct_version(self):
        info = refresh_protocol()
        self.assertEqual(info.get("protocol_version"), EXPECTED_VERSION)
        self.assertIn("last_refresh", info)
        self.assertIn("next_refresh_due", info)

    def test_protocol_metadata_consistency(self):
        self.assertEqual(PROTOCOL_VERSION, EXPECTED_VERSION)
        meta = get_protocol_metadata()
        self.assertEqual(meta.get("version"), EXPECTED_VERSION)
        self.assertEqual(meta.get("last_updated"), EXPECTED_DATE)
        self.assertEqual(meta.get("sync_date"), EXPECTED_DATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)

