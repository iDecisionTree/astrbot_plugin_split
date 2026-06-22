import re
from dataclasses import dataclass


_MARKER_RE = re.compile(r"\[start\](.*?)\[end\]", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SplitOptions:
    max_segments: int = 5
    fallback_max_chars: int = 0
    strip_segments: bool = True


@dataclass(frozen=True)
class SplitResult:
    segments: list[str]
    changed: bool
    used_markers: bool


def build_split_prompt(max_segments: int = 5) -> str:
    segment_limit = max(1, int(max_segments or 1))
    return (
        "Output segmentation rule: wrap every text response segment exactly as "
        "[start]segment text[end]. "
        f"Use at most {segment_limit} segments. "
        "For short answers, use one [start]...[end] block. "
        "Do not put any response text outside these markers. "
        "Keep code blocks, lists, and markdown inside segment content."
    )


def split_response_text(text: str, options: SplitOptions | None = None) -> SplitResult:
    opts = options or SplitOptions()
    original = text or ""

    marked_segments = _extract_marked_segments(original, opts.strip_segments)
    if marked_segments:
        return SplitResult(
            segments=_cap_segments(marked_segments, opts.max_segments, "\n\n"),
            changed=True,
            used_markers=True,
        )

    if opts.fallback_max_chars > 0 and len(original) > opts.fallback_max_chars:
        return SplitResult(
            segments=_cap_segments(
                _split_by_length(original, opts.fallback_max_chars, opts.strip_segments),
                opts.max_segments,
                "",
            ),
            changed=True,
            used_markers=False,
        )

    return SplitResult(segments=[original], changed=False, used_markers=False)


def _extract_marked_segments(text: str, strip_segments: bool) -> list[str]:
    segments = []
    for match in _MARKER_RE.finditer(text):
        segment = match.group(1)
        if strip_segments:
            segment = segment.strip()
        if segment:
            segments.append(segment)
    return segments


def _cap_segments(segments: list[str], max_segments: int, separator: str) -> list[str]:
    segment_limit = max(1, int(max_segments or 1))
    cleaned = [segment for segment in segments if segment]

    if len(cleaned) <= segment_limit:
        return cleaned

    if segment_limit == 1:
        return [separator.join(cleaned)]

    return [*cleaned[: segment_limit - 1], separator.join(cleaned[segment_limit - 1 :])]


def _split_by_length(text: str, max_chars: int, strip_segments: bool) -> list[str]:
    max_chars = max(1, int(max_chars))
    remaining = text.strip() if strip_segments else text
    segments: list[str] = []

    while len(remaining) > max_chars:
        split_at = _find_split_boundary(remaining, max_chars)
        segment = remaining[:split_at]
        remaining = remaining[split_at:]

        if strip_segments:
            segment = segment.strip()
            remaining = remaining.lstrip()

        if segment:
            segments.append(segment)

    tail = remaining.strip() if strip_segments else remaining
    if tail:
        segments.append(tail)

    return segments


def _find_split_boundary(text: str, max_chars: int) -> int:
    window = text[: max_chars + 1]
    min_boundary = max(1, int(max_chars * 0.5))
    delimiters = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")

    for delimiter in delimiters:
        index = window.rfind(delimiter)
        if index >= min_boundary:
            return index + len(delimiter)

    return max_chars
