import unittest

from splitter import SplitOptions, build_split_prompt, split_response_text


class SplitterTests(unittest.TestCase):
    def test_marker_segments_are_extracted_and_capped_without_losing_text(self):
        result = split_response_text(
            "[start]alpha[end][start]beta[end][start]gamma[end]",
            SplitOptions(max_segments=2),
        )

        self.assertTrue(result.changed)
        self.assertTrue(result.used_markers)
        self.assertEqual(result.segments, ["alpha", "beta\n\ngamma"])

    def test_single_marked_segment_is_cleaned_without_forcing_split(self):
        result = split_response_text(
            "[start]\nshort answer\n[end]",
            SplitOptions(max_segments=5),
        )

        self.assertTrue(result.changed)
        self.assertTrue(result.used_markers)
        self.assertEqual(result.segments, ["short answer"])

    def test_fallback_splits_long_unmarked_text_when_enabled(self):
        result = split_response_text(
            "abcdefghij",
            SplitOptions(max_segments=2, fallback_max_chars=4),
        )

        self.assertTrue(result.changed)
        self.assertFalse(result.used_markers)
        self.assertEqual(result.segments, ["abcd", "efghij"])

    def test_unmarked_text_is_unchanged_when_fallback_disabled(self):
        result = split_response_text(
            "plain response",
            SplitOptions(max_segments=3, fallback_max_chars=0),
        )

        self.assertFalse(result.changed)
        self.assertEqual(result.segments, ["plain response"])

    def test_prompt_includes_marker_contract_and_segment_limit(self):
        prompt = build_split_prompt(max_segments=4)

        self.assertIn("[start]", prompt)
        self.assertIn("[end]", prompt)
        self.assertIn("4", prompt)


if __name__ == "__main__":
    unittest.main()
