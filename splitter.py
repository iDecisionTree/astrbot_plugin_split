import asyncio
import json
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SplitOptions:
    max_segments: int = 5
    fallback_max_chars: int = 0
    strip_segments: bool = True


@dataclass(frozen=True)
class SplitResult:
    segments: list[str]
    changed: bool
    used_llm: bool


def build_segmentation_prompt(
    text: str,
    max_segments: int = 5,
    style: str = "natural",
) -> str:
    segment_limit = max(1, int(max_segments or 1))
    split_limit = max(0, segment_limit - 1)
    style_name, style_instruction = _style_instruction(style)
    return (
        "Decide where to split the assistant reply below into natural message segments.\n"
        f"Segmentation style: {style_name}. {style_instruction}\n"
        f"Return at most {split_limit} split positions, producing at most {segment_limit} segments.\n"
        "Use Unicode character offsets in the original reply. Each offset means "
        "split after original_reply[:offset].\n"
        f"Offsets must be strictly increasing integers where 1 <= offset < {len(text or '')}.\n"
        'Return only JSON object like {"split_after":[120,260]}.\n'
        "Do not copy or rewrite the reply. Do not return segment strings.\n"
        "Avoid split positions inside code blocks, markdown tables, URLs, and list items.\n"
        "\n"
        "<assistant_reply>\n"
        f"{text or ''}\n"
        "</assistant_reply>"
    )


def _style_instruction(style: str) -> tuple[str, str]:
    normalized = (style or "natural").strip().lower()
    if normalized == "conservative":
        return "conservative", "Use fewer, longer segments unless a clear pause exists."
    if normalized == "active":
        return "active", "Use shorter, livelier chat-like segments while preserving meaning."
    return "natural", "Use balanced segments that feel like normal conversation."


def parse_llm_segments(
    text: str,
    options: SplitOptions | None = None,
    original_text: str | None = None,
) -> SplitResult:
    opts = options or SplitOptions()
    payload = _load_json_payload(text or "")
    if payload is None:
        return SplitResult(segments=[], changed=False, used_llm=False)

    if isinstance(payload, dict):
        offset_segments = _segments_from_offsets(
            payload.get("split_after"),
            original_text,
            opts.strip_segments,
        )
        if offset_segments:
            return SplitResult(
                segments=_cap_segments(offset_segments, opts.max_segments, ""),
                changed=True,
                used_llm=True,
            )
        raw_segments = payload.get("segments")
    else:
        raw_segments = payload

    if not isinstance(raw_segments, list):
        return SplitResult(segments=[], changed=False, used_llm=False)

    segments = _clean_segments(raw_segments, opts.strip_segments)
    if not segments:
        return SplitResult(segments=[], changed=False, used_llm=False)

    return SplitResult(
        segments=_cap_segments(segments, opts.max_segments, "\n\n"),
        changed=True,
        used_llm=True,
    )


def describe_secondary_llm_failure(
    exc: BaseException,
    timeout_seconds: float,
) -> tuple[str, bool]:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return (
            f"调用分段模型超时（{timeout_seconds:g} 秒），使用本地兜底。",
            False,
        )
    return (f"调用分段模型失败：{exc}。使用本地兜底。", True)


def split_response_text(text: str, options: SplitOptions | None = None) -> SplitResult:
    opts = options or SplitOptions()
    original = text or ""

    if opts.fallback_max_chars > 0 and len(original) > opts.fallback_max_chars:
        return SplitResult(
            segments=_cap_segments(
                _split_by_length(original, opts.fallback_max_chars, opts.strip_segments),
                opts.max_segments,
                "",
            ),
            changed=True,
            used_llm=False,
        )

    return SplitResult(segments=[original], changed=False, used_llm=False)


def _load_json_payload(text: str) -> Any | None:
    candidate = _strip_json_fence(text.strip())
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except JSONDecodeError:
        return _scan_json_payload(candidate)


def _strip_json_fence(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def _scan_json_payload(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
            return payload
        except JSONDecodeError:
            continue
    return None


def _clean_segments(values: list[Any], strip_segments: bool) -> list[str]:
    segments = []
    for value in values:
        if not isinstance(value, str):
            continue
        segment = value.strip() if strip_segments else value
        if segment:
            segments.append(segment)
    return segments


def _segments_from_offsets(
    values: Any,
    original_text: str | None,
    strip_segments: bool,
) -> list[str]:
    if not isinstance(values, list) or original_text is None:
        return []

    offsets = _clean_offsets(values, len(original_text))
    if not offsets:
        return []

    segments: list[str] = []
    start = 0
    for offset in offsets:
        segment = original_text[start:offset]
        segment = segment.strip() if strip_segments else segment
        if segment:
            segments.append(segment)
        start = offset

    tail = original_text[start:]
    tail = tail.strip() if strip_segments else tail
    if tail:
        segments.append(tail)
    return segments


def _clean_offsets(values: list[Any], text_length: int) -> list[int]:
    offsets: set[int] = set()
    for value in values:
        offset = _coerce_offset(value)
        if offset is not None and 0 < offset < text_length:
            offsets.add(offset)
    return sorted(offsets)


def _coerce_offset(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value)
    return None


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
