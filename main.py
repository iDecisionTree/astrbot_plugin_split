from __future__ import annotations

import asyncio
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register

try:
    from .splitter import SplitOptions, build_split_prompt, split_response_text
except ImportError:
    from splitter import SplitOptions, build_split_prompt, split_response_text


DEFAULT_CONFIG = {
    "enabled": True,
    "only_llm_result": True,
    "inject_prompt": True,
    "max_segments": 5,
    "fallback_max_chars": 700,
    "send_separately": True,
    "send_interval_seconds": 0.8,
}

_EVENT_SEGMENTS_KEY = "_astrbot_plugin_split_segments"


@register(
    "astrbot_plugin_split",
    "DecisionTree",
    "Split long LLM text replies into marker-based segments.",
    "1.0.0",
)
class SplitPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}

    @filter.on_llm_request()
    async def inject_split_prompt(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        if not self._get_bool("enabled") or not self._get_bool("inject_prompt"):
            return

        prompt = build_split_prompt(self._get_int("max_segments"))
        system_prompt = getattr(req, "system_prompt", None) or ""
        if prompt in system_prompt:
            return

        req.system_prompt = (
            f"{system_prompt.rstrip()}\n\n{prompt}" if system_prompt.strip() else prompt
        )

    @filter.on_llm_response()
    async def clean_llm_response(
        self,
        event: AstrMessageEvent,
        resp: LLMResponse,
    ) -> None:
        if not self._get_bool("enabled") or resp is None:
            return
        if getattr(resp, "is_chunk", False):
            return

        text = self._response_text(resp)
        if not text:
            return

        split_result = split_response_text(text, self._split_options())
        if not split_result.changed:
            return

        event.set_extra(_EVENT_SEGMENTS_KEY, split_result.segments)
        if getattr(resp, "result_chain", None):
            resp.result_chain.chain = [Plain(segment) for segment in split_result.segments]
        else:
            resp.completion_text = "\n\n".join(split_result.segments)

    @filter.on_decorating_result(priority=-1000)
    async def split_decorating_result(self, event: AstrMessageEvent) -> None:
        if not self._get_bool("enabled"):
            return

        result = event.get_result()
        if result is None or not getattr(result, "chain", None):
            return
        if not self._should_process_result(result):
            return
        if not self._is_plain_chain(result.chain):
            return

        stored_segments = event.get_extra(_EVENT_SEGMENTS_KEY)
        if stored_segments:
            segments = [comp.text for comp in result.chain if comp.text.strip()]
        else:
            text = "".join(comp.text for comp in result.chain)
            split_result = split_response_text(text, self._split_options())
            if not split_result.changed:
                return
            segments = split_result.segments

        if len(segments) <= 1 or not self._get_bool("send_separately"):
            result.chain = [Plain(segment) for segment in segments]
            return

        await self._send_segments(event, result, segments)
        event.clear_result()

    async def _send_segments(
        self,
        event: AstrMessageEvent,
        result: Any,
        segments: list[str],
    ) -> None:
        interval = self._get_float("send_interval_seconds")
        for index, segment in enumerate(segments):
            await event.send(result.derive([Plain(segment)]))
            if interval > 0 and index < len(segments) - 1:
                await asyncio.sleep(interval)

    def _response_text(self, resp: LLMResponse) -> str:
        chain = getattr(getattr(resp, "result_chain", None), "chain", None)
        if chain:
            if not self._is_plain_chain(chain):
                return ""
            return "".join(comp.text for comp in chain)
        return getattr(resp, "completion_text", "") or ""

    def _split_options(self) -> SplitOptions:
        return SplitOptions(
            max_segments=self._get_int("max_segments"),
            fallback_max_chars=self._get_int("fallback_max_chars"),
        )

    def _should_process_result(self, result: Any) -> bool:
        if not self._get_bool("only_llm_result"):
            return True

        checker = getattr(result, "is_model_result", None)
        if callable(checker):
            return bool(checker())

        checker = getattr(result, "is_llm_result", None)
        if callable(checker):
            return bool(checker())

        logger.debug("Skip split: result type cannot be identified as LLM output.")
        return False

    @staticmethod
    def _is_plain_chain(chain: list[Any]) -> bool:
        return bool(chain) and all(isinstance(comp, Plain) for comp in chain)

    def _get_value(self, key: str) -> Any:
        if hasattr(self.config, "get"):
            return self.config.get(key, DEFAULT_CONFIG[key])
        return DEFAULT_CONFIG[key]

    def _get_bool(self, key: str) -> bool:
        value = self._get_value(key)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_int(self, key: str) -> int:
        try:
            return max(0, int(self._get_value(key)))
        except (TypeError, ValueError):
            return int(DEFAULT_CONFIG[key])

    def _get_float(self, key: str) -> float:
        try:
            return max(0.0, float(self._get_value(key)))
        except (TypeError, ValueError):
            return float(DEFAULT_CONFIG[key])
