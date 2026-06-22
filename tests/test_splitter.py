import unittest

from splitter import (
    SplitOptions,
    build_segmentation_prompt,
    parse_llm_segments,
    split_response_text,
)


class SplitterTests(unittest.TestCase):
    def test_llm_json_array_segments_are_extracted_and_capped(self):
        result = parse_llm_segments(
            '["alpha", "beta", "gamma"]',
            SplitOptions(max_segments=2),
        )

        self.assertTrue(result.changed)
        self.assertTrue(result.used_llm)
        self.assertEqual(result.segments, ["alpha", "beta\n\ngamma"])

    def test_llm_segments_object_is_supported(self):
        result = parse_llm_segments(
            '{"segments": ["第一段", "第二段"]}',
            SplitOptions(max_segments=5),
        )

        self.assertTrue(result.changed)
        self.assertTrue(result.used_llm)
        self.assertEqual(result.segments, ["第一段", "第二段"])

    def test_json_inside_markdown_fence_is_supported(self):
        result = parse_llm_segments(
            '```json\n["one", "two"]\n```',
            SplitOptions(max_segments=5),
        )

        self.assertEqual(result.segments, ["one", "two"])

    def test_invalid_llm_segments_do_not_change_text(self):
        result = parse_llm_segments("not json", SplitOptions(max_segments=3))

        self.assertFalse(result.changed)
        self.assertFalse(result.used_llm)
        self.assertEqual(result.segments, [])

    def test_fallback_splits_long_unmarked_text_when_enabled(self):
        result = split_response_text(
            "abcdefghij",
            SplitOptions(max_segments=2, fallback_max_chars=4),
        )

        self.assertTrue(result.changed)
        self.assertFalse(result.used_llm)
        self.assertEqual(result.segments, ["abcd", "efghij"])

    def test_unmarked_text_is_unchanged_when_fallback_disabled(self):
        result = split_response_text(
            "plain response",
            SplitOptions(max_segments=3, fallback_max_chars=0),
        )

        self.assertFalse(result.changed)
        self.assertEqual(result.segments, ["plain response"])

    def test_prompt_includes_original_text_and_segment_limit(self):
        prompt = build_segmentation_prompt("原始回复", max_segments=4)

        self.assertIn("原始回复", prompt)
        self.assertIn("4", prompt)
        self.assertIn("JSON", prompt)


if __name__ == "__main__":
    unittest.main()
