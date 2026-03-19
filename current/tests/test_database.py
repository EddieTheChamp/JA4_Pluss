"""Tests for the local database matcher."""

from __future__ import annotations

import unittest

from current.services.local_matcher import LocalDatabaseMatcher
from current.core.models import DatabaseRecord, Observation


class LocalDatabaseMatcherTests(unittest.TestCase):
    def test_database_exact_match(self) -> None:
        matcher = LocalDatabaseMatcher.from_records(
            [
                DatabaseRecord(
                    ja4="t13d1516h2_aaa_bbb",
                    ja4s="t130200_1301_ccc",
                    application="chrome.exe",
                    category="browser",
                    count=8,
                )
            ]
        )

        result = matcher.match_observation(
            Observation(
                observation_id="obs-1",
                ja4="t13d1516h2_aaa_bbb",
                ja4s="t130200_1301_ccc",
            )
        )

        self.assertEqual(result.status, "unique")
        self.assertEqual(result.match_mode, "ja4_ja4s")
        self.assertEqual(result.candidates[0].application, "chrome.exe")
        self.assertEqual(result.candidates[0].category, "browser")

    def test_database_ambiguous_match(self) -> None:
        matcher = LocalDatabaseMatcher.from_records(
            [
                DatabaseRecord(
                    ja4="t13d1516h2_aaa_bbb",
                    application="chrome.exe",
                    category="browser",
                    count=10,
                ),
                DatabaseRecord(
                    ja4="t13d1516h2_aaa_bbb",
                    application="msedge.exe",
                    category="browser",
                    count=7,
                ),
            ]
        )

        result = matcher.match_observation(
            Observation(
                observation_id="obs-2",
                ja4="t13d1516h2_aaa_bbb",
            )
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.match_mode, "ja4_only")
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].application, "chrome.exe")


if __name__ == "__main__":
    unittest.main()
