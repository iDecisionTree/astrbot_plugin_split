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
        describe_secondary_llm_failure,
        parse_llm_segments,
        split_response_text,
    )
except ImportError:
    from splitter import (
        SplitOptions,
        build_segmentation_prompt,
        describe_secondary_llm_failure,
        parse_llm_segments,
        split_response_text,
    )


DEFAULT_SPLIT_SYSTEM_PROMPT = (
    "You are a strict text segmenter. Return only a valid JSON object with "
    "split_after offsets. Do not copy, summarize, translate, rewrite, add, "
    "or remove content."
)

DEFAULT_CONFIG = {
    "basic_settings": {
        "enabled": True,
        "only_llm_result": True,
        "min_length": 15,
    },
    "model_settings": {
        "provider_id": "",
        "style": "natural",
        "system_prompt": DEFAULT_SPLIT_SYSTEM_PROMPT,
        "temperature": 0.3,
        "max_tokens": 256,
        "timeout_seconds": 20.0,
    },
    "split_settings": {
        "max_segments": 8,
        "fallback_max_chars": 700,
    },
    "send_settings": {
        "send_separately": True,
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    },
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
        if not self._get_bool("basic_settings.enabled", "enabled"):
            logger.debug("分段插件跳过处理：插件未启用。")
            return

        result = event.get_result()
        if result is None or not getattr(result, "chain", None):
            logger.debug("分段插件跳过处理：消息结果为空。")
            return
        if not self._should_process_result(result):
            logger.debug("分段插件跳过处理：不是模型回复。")
            return
        if not self._is_plain_chain(result.chain):
            logger.debug("分段插件跳过处理：消息链包含非文本组件。")
            return

        text = "".join(comp.text for comp in result.chain)
        if not text.strip():
            logger.debug("分段插件跳过处理：文本为空。")
            return
        min_length = self._get_int("basic_settings.min_length", "min_length")
        if len(text) < min_length:
            logger.debug(
                "分段插件跳过处理：文本长度 %s 小于最小触发长度 %s。",
                len(text),
                min_length,
            )
            return

        segments = await self._split_with_secondary_llm(event, text)
        if not segments:
            fallback = split_response_text(text, self._split_options())
            if not fallback.changed:
                logger.info(
                    "分段插件保留原文：分段模型没有返回可用分段，且未触发本地兜底。",
                )
                return
            segments = fallback.segments
            logger.info(
                "分段插件使用本地兜底：文本长度=%s，分段数=%s。",
                len(text),
                len(segments),
            )

        if len(segments) <= 1 or not self._get_bool(
            "send_settings.send_separately",
            "send_separately",
        ):
            result.chain = [Plain(segment) for segment in segments]
            logger.info(
                "分段插件已替换消息链：分段数=%s，未启用逐条发送。",
                len(segments),
            )
            return

        logger.info("分段插件开始逐条发送：分段数=%s。", len(segments))
        await self._send_segments(event, result, segments)
        event.clear_result()

    async def _split_with_secondary_llm(
        self,
        event: AstrMessageEvent,
        text: str,
    ) -> list[str]:
        provider_id = self._get_split_provider_id(event)
        if not provider_id:
            logger.warning("未配置分段模型 Provider，使用本地兜底。")
            return []
        if not callable(getattr(self.context, "llm_generate", None)):
            logger.warning("当前 AstrBot Context 不支持 llm_generate，使用本地兜底。")
            return []

        style = self._get_str("model_settings.style", "style")
        max_segments = self._get_int("split_settings.max_segments", "max_segments")
        prompt = build_segmentation_prompt(
            text,
            max_segments,
            style,
        )
        system_prompt = self._get_str(
            "model_settings.system_prompt",
            "split_model_system_prompt",
        )
        timeout = self._get_float(
            "model_settings.timeout_seconds",
            "split_model_timeout_seconds",
        )

        try:
            logger.info(
                "分段插件开始请求分段模型：provider_id=%s，文本长度=%s，"
                "最大分段数=%s，风格=%s。",
                provider_id,
                len(text),
                max_segments,
                style,
            )
            kwargs = {
                "temperature": self._get_float(
                    "model_settings.temperature",
                    "temperature",
                ),
                "max_tokens": self._get_int(
                    "model_settings.max_tokens",
                    "max_tokens",
                ),
            }
            task = self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )
            response = await asyncio.wait_for(task, timeout) if timeout > 0 else await task
        except (TimeoutError, asyncio.TimeoutError) as exc:
            message, include_trace = describe_secondary_llm_failure(exc, timeout)
            logger.info(message)
            if include_trace:
                logger.debug("分段模型异常详情。", exc_info=True)
            return []
        except Exception as exc:
            message, include_trace = describe_secondary_llm_failure(exc, timeout)
            logger.warning(message)
            if include_trace:
                logger.debug("分段模型异常详情。", exc_info=True)
            return []

        split_result = parse_llm_segments(
            self._response_text(response),
            self._split_options(),
            original_text=text,
        )
        if split_result.changed:
            logger.info(
                "分段模型分段成功：分段数=%s。",
                len(split_result.segments),
            )
            return split_result.segments

        logger.warning(
            "分段模型返回的 JSON 不可用，使用本地兜底。",
        )
        return []

    def _get_split_provider_id(self, event: AstrMessageEvent) -> str:
        configured = self._get_str(
            "model_settings.provider_id",
            "split_provider_id",
        ).strip()
        if not configured:
            legacy_provider_id = self._read_config_path("provider_id", "")
            configured = str(legacy_provider_id or "").strip()
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
        for index, segment in enumerate(segments):
            await event.send(result.derive([Plain(segment)]))
            interval = self._calc_delay(segment)
            if interval > 0 and index < len(segments) - 1:
                await asyncio.sleep(interval)

    def _calc_delay(self, segment: str) -> float:
        legacy_interval = self._get_optional_float("send_interval_seconds")
        if legacy_interval is not None:
            return legacy_interval

        base = self._get_float("send_settings.delay_base")
        per_char = self._get_float("send_settings.delay_per_char")
        max_delay = self._get_float("send_settings.delay_max")
        delay = base + len(segment) * per_char
        if max_delay > 0:
            return min(delay, max_delay)
        return delay

    def _response_text(self, response: Any) -> str:
        chain = getattr(getattr(response, "result_chain", None), "chain", None)
        if chain:
            plain_texts = [comp.text for comp in chain if isinstance(comp, Plain)]
            if plain_texts:
                return "".join(plain_texts)
        return getattr(response, "completion_text", "") or ""

    def _split_options(self) -> SplitOptions:
        return SplitOptions(
            max_segments=self._get_int("split_settings.max_segments", "max_segments"),
            fallback_max_chars=self._get_int(
                "split_settings.fallback_max_chars",
                "fallback_max_chars",
            ),
        )

    def _should_process_result(self, result: Any) -> bool:
        if not self._get_bool("basic_settings.only_llm_result", "only_llm_result"):
            return True

        checker = getattr(result, "is_model_result", None)
        if callable(checker):
            return bool(checker())

        checker = getattr(result, "is_llm_result", None)
        if callable(checker):
            return bool(checker())

        logger.debug("分段插件跳过处理：无法识别该结果是否为模型输出。")
        return False

    @staticmethod
    def _is_plain_chain(chain: list[Any]) -> bool:
        return bool(chain) and all(isinstance(comp, Plain) for comp in chain)

    def _get_value(self, key: str, legacy_key: str | None = None) -> Any:
        missing = object()
        value = self._read_config_path(key, missing)
        if value is not missing:
            return value
        if legacy_key:
            value = self._read_config_path(legacy_key, missing)
            if value is not missing:
                return value
        return self._read_default_path(key)

    def _read_config_path(self, key: str, default: Any) -> Any:
        if not hasattr(self.config, "get"):
            return default
        current: Any = self.config
        for part in key.split("."):
            if not hasattr(current, "get"):
                return default
            current = current.get(part, default)
            if current is default:
                return default
        return current

    @staticmethod
    def _read_default_path(key: str) -> Any:
        current: Any = DEFAULT_CONFIG
        for part in key.split("."):
            current = current[part]
        return current

    def _get_bool(self, key: str, legacy_key: str | None = None) -> bool:
        value = self._get_value(key, legacy_key)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_int(self, key: str, legacy_key: str | None = None) -> int:
        try:
            return max(0, int(self._get_value(key, legacy_key)))
        except (TypeError, ValueError):
            return int(self._read_default_path(key))

    def _get_float(self, key: str, legacy_key: str | None = None) -> float:
        try:
            return max(0.0, float(self._get_value(key, legacy_key)))
        except (TypeError, ValueError):
            return float(self._read_default_path(key))

    def _get_optional_float(self, key: str) -> float | None:
        missing = object()
        value = self._read_config_path(key, missing)
        if value is missing:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    def _get_str(self, key: str, legacy_key: str | None = None) -> str:
        value = self._get_value(key, legacy_key)
        if value is None:
            return ""
        return str(value)
