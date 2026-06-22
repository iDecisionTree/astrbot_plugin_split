from __future__ import annotations

import asyncio
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, register

try:
    from .splitter import (
        SplitOptions,
        build_segmentation_prompt,
        parse_llm_segments,
        split_response_text,
    )
except ImportError:
    from splitter import (
        SplitOptions,
        build_segmentation_prompt,
        parse_llm_segments,
        split_response_text,
    )


DEFAULT_SPLIT_SYSTEM_PROMPT = (
    "You are a strict text segmenter. Return only valid JSON. "
    "Do not summarize, translate, rewrite, add, or remove content."
)

DEFAULT_CONFIG = {
    "enabled": True,
    "only_llm_result": True,
    "split_provider_id": "",
    "split_model_system_prompt": DEFAULT_SPLIT_SYSTEM_PROMPT,
    "split_model_timeout_seconds": 30,
    "max_segments": 5,
    "fallback_max_chars": 700,
    "send_separately": True,
    "send_interval_seconds": 0.8,
}


@register(
    "astrbot_plugin_split",
    "DecisionTree",
    "Split long LLM text replies with a secondary LLM.",
    "1.0.0",
)
class SplitPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}

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

        text = "".join(comp.text for comp in result.chain)
        if not text.strip():
            return

        segments = await self._split_with_secondary_llm(event, text)
        if not segments:
            fallback = split_response_text(text, self._split_options())
            if not fallback.changed:
                return
            segments = fallback.segments

        if len(segments) <= 1 or not self._get_bool("send_separately"):
            result.chain = [Plain(segment) for segment in segments]
            return

        await self._send_segments(event, result, segments)
        event.clear_result()

    async def _split_with_secondary_llm(
        self,
        event: AstrMessageEvent,
        text: str,
    ) -> list[str]:
        provider_id = self._get_split_provider_id(event)
        if not provider_id:
            logger.warning("Split provider is not configured; using local fallback.")
            return []
        if not callable(getattr(self.context, "llm_generate", None)):
            logger.warning("Context.llm_generate is unavailable; using local fallback.")
            return []

        prompt = build_segmentation_prompt(text, self._get_int("max_segments"))
        system_prompt = self._get_str("split_model_system_prompt")
        timeout = self._get_float("split_model_timeout_seconds")

        try:
            task = self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt,
            )
            response = await asyncio.wait_for(task, timeout) if timeout > 0 else await task
        except Exception:
            logger.warning("Secondary LLM segmentation failed; using local fallback.", exc_info=True)
            return []

        split_result = parse_llm_segments(
            self._response_text(response),
            self._split_options(),
        )
        if split_result.changed:
            return split_result.segments

        logger.warning("Secondary LLM returned invalid segmentation JSON; using local fallback.")
        return []

    def _get_split_provider_id(self, event: AstrMessageEvent) -> str:
        configured = self._get_str("split_provider_id").strip()
        if configured:
            return configured

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if provider is None:
            return ""

        try:
            return str(provider.meta().id or "")
        except Exception:
            return str(getattr(provider, "provider_config", {}).get("id", ""))

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

    def _response_text(self, response: Any) -> str:
        chain = getattr(getattr(response, "result_chain", None), "chain", None)
        if chain:
            plain_texts = [comp.text for comp in chain if isinstance(comp, Plain)]
            if plain_texts:
                return "".join(plain_texts)
        return getattr(response, "completion_text", "") or ""

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

    def _get_str(self, key: str) -> str:
        value = self._get_value(key)
        if value is None:
            return ""
        return str(value)
