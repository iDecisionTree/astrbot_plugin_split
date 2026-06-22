import unittest

from splitter import (
    SplitOptions,
    build_segmentation_prompt,
    describe_secondary_llm_failure,
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

    def test_llm_split_after_offsets_slice_original_text(self):
        result = parse_llm_segments(
            '{"split_after": [5, 10]}',
            SplitOptions(max_segments=4),
            original_text="abcdefghijk",
        )

        self.assertTrue(result.changed)
        self.assertTrue(result.used_llm)
        self.assertEqual(result.segments, ["abcde", "fghij", "k"])

    def test_llm_split_after_offsets_are_sanitized_and_capped(self):
        result = parse_llm_segments(
            '{"split_after": [0, 4, 4, "8", 99, 12]}',
            SplitOptions(max_segments=3),
            original_text="abcdefghijklmnop",
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.segments, ["abcd", "efgh", "ijklmnop"])

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
        self.assertIn('"split_after"', prompt)
        self.assertIn("Do not copy", prompt)

    def test_prompt_includes_segmentation_style(self):
        prompt = build_segmentation_prompt("hello", max_segments=4, style="active")

        self.assertIn("shorter", prompt)
        self.assertIn("active", prompt)

    def test_timeout_failure_is_reported_without_traceback(self):
        message, include_trace = describe_secondary_llm_failure(
            TimeoutError(),
            timeout_seconds=12,
        )

        self.assertIn("超时", message)
        self.assertIn("12", message)
        self.assertFalse(include_trace)


if __name__ == "__main__":
    unittest.main()
