"""DeepSeek V4 via OpenRouter's OpenAI-compatible endpoint.

Why this file exists
--------------------
OpenRouter returns a model's chain-of-thought in two non-standard fields --
``reasoning`` (plaintext) and ``reasoning_details`` (structured) -- and requires
the structured one to be echoed back when a conversation continues, especially
across tool calls:

    "When passing back `reasoning_details`, preserve the exact sequence returned
     by the model -- no rearranging or modification permitted."
    -- https://openrouter.ai/docs/use-cases/reasoning-tokens

``langchain_openai`` drops both. From the docstring of its own
``_convert_message_to_dict``:

    "Non-standard response fields added by third-party providers (e.g.
     `reasoning_content`, `reasoning_details`) are *not* extracted or preserved."

So a stock ``ChatOpenAI`` in an agent loop reasons on the first call and then
loses that reasoning the moment a tool result comes back.

``ChatOpenRouter`` below closes the round trip. (It is a small local class, not
the separate ``langchain-openrouter`` package.)

The streaming half has a trap. ``reasoning_details`` arrive as deltas, and
LangChain merges chunk ``additional_kwargs`` with ``merge_lists``, which
field-merges any two list items sharing an integer ``index`` -- concatenating
*every* string field, not just the payload. Two chunks of one detail would
merge to ``format: "unknownunknown"``, corrupting exactly the structure
OpenRouter says to send back unmodified. So deltas are stored as fragments with
``index`` renamed to ``__or_index``, which makes ``merge_lists`` append them
verbatim, and they are stitched back together at request time.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

from .config import Settings

#: Structured reasoning; what OpenRouter wants echoed back.
DETAILS_KEY = "reasoning_details"
#: Plaintext reasoning; used for display, and as passback fallback.
TEXT_KEY = "reasoning"
#: Our stand-in for OpenRouter's "index", chosen so merge_lists won't field-merge.
FRAGMENT_INDEX = "__or_index"
#: The per-detail fields that actually stream in pieces.
PAYLOAD_FIELDS = ("text", "summary", "data")


def _field(source: Any, name: str) -> Any:
    """Read a non-standard field off a dict or an openai pydantic model."""
    if isinstance(source, dict):
        return source.get(name)
    value = getattr(source, name, None)
    if value is None:
        extra = getattr(source, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return value


def _to_fragments(details: Any) -> list[dict]:
    """Rename ``index`` so chunk merging appends instead of field-merging."""
    if not isinstance(details, list):
        return []
    fragments = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        fragment = {k: v for k, v in detail.items() if k != "index"}
        if "index" in detail:
            fragment[FRAGMENT_INDEX] = detail["index"]
        fragments.append(fragment)
    return fragments


def _coalesce(fragments: Any) -> list[dict]:
    """Stitch accumulated fragments back into OpenRouter reasoning details.

    Fragments sharing an index/id/type are one detail split across chunks;
    their payload fields are concatenated in arrival order, which is the
    "exact sequence" OpenRouter asks for. Everything else is passed through
    untouched.
    """
    if not isinstance(fragments, list):
        return []
    details: list[dict] = []
    by_key: dict[tuple, dict] = {}
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        detail = {k: v for k, v in fragment.items() if k != FRAGMENT_INDEX}
        if FRAGMENT_INDEX in fragment:
            detail["index"] = fragment[FRAGMENT_INDEX]

        index = detail.get("index")
        key = (index, detail.get("id"), detail.get("type"))
        target = by_key.get(key) if index is not None else None
        if target is None:
            by_key[key] = detail
            details.append(detail)
            continue
        for name in PAYLOAD_FIELDS:
            if name in detail:
                target[name] = (target.get(name) or "") + (detail[name] or "")
    return details


class ChatOpenRouter(ChatOpenAI):
    """``ChatOpenAI`` plus a reasoning round trip for OpenRouter.

    Set ``reasoning_enabled`` to match the ``reasoning`` block passed in
    ``extra_body``; it decides whether reasoning is replayed or stripped.
    """

    reasoning_enabled: bool = Field(default=False)

    # -- capture: non-streaming ------------------------------------------------

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        choices = _field(response, "choices") or []
        for generation, choice in zip(result.generations, choices):
            message = _field(choice, "message")
            if message is None:
                continue
            extras = generation.message.additional_kwargs
            fragments = _to_fragments(_field(message, DETAILS_KEY))
            if fragments:
                extras[DETAILS_KEY] = fragments
            # OpenRouter exposes the plaintext as `reasoning`; DeepSeek's own
            # name for it, `reasoning_content`, is accepted as an alias.
            text = _field(message, TEXT_KEY) or _field(message, "reasoning_content")
            if text:
                extras[TEXT_KEY] = text
        return result

    # -- capture: streaming ----------------------------------------------------

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None
        choices = chunk.get("choices") or []
        if not choices:
            return generation_chunk
        delta = choices[0].get("delta") or {}
        extras = generation_chunk.message.additional_kwargs

        fragments = _to_fragments(delta.get(DETAILS_KEY))
        if fragments:
            extras[DETAILS_KEY] = fragments
        text = delta.get(TEXT_KEY) or delta.get("reasoning_content")
        if text:
            # Plain strings concatenate across chunks for free.
            extras[TEXT_KEY] = text
        return generation_chunk

    # -- passback: the half nothing upstream implements -------------------------

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        outgoing = payload.get("messages")
        if not isinstance(outgoing, list):
            # Responses API path has no "messages" key; nothing to patch.
            return payload

        originals = self._convert_input(input_).to_messages()
        if len(originals) != len(outgoing):
            # Conversion is 1:1 today. If that ever stops holding, leave the
            # payload untouched rather than pairing reasoning to wrong messages.
            return payload

        for original, message in zip(originals, outgoing):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if not self.reasoning_enabled:
                # Stale reasoning from an earlier reasoning-enabled session
                # would be replayed into a request that asked for none.
                message.pop(DETAILS_KEY, None)
                message.pop(TEXT_KEY, None)
                continue

            extras = getattr(original, "additional_kwargs", {}) or {}
            details = _coalesce(extras.get(DETAILS_KEY))
            if details:
                message[DETAILS_KEY] = details
            elif extras.get(TEXT_KEY):
                # No structured details (some providers omit them) -- fall back
                # to the plaintext form, which OpenRouter also accepts.
                message[TEXT_KEY] = extras[TEXT_KEY]

        return payload


def reasoning_text(extras: dict) -> str:
    """Human-readable reasoning from a message's ``additional_kwargs``."""
    if extras.get(TEXT_KEY):
        return str(extras[TEXT_KEY])
    parts = []
    for detail in _coalesce(extras.get(DETAILS_KEY)):
        for name in PAYLOAD_FIELDS:
            if detail.get(name):
                parts.append(str(detail[name]))
    return "".join(parts)


def build_llm(settings: Settings) -> ChatOpenRouter:
    """Construct the chat model from settings.

    Provider-specific body params go in ``extra_body``, not ``model_kwargs`` --
    LangChain routes the latter through OpenAI's own parameter validation.
    """
    return ChatOpenRouter(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        reasoning_enabled=settings.reasoning_enabled,
        extra_body=settings.reasoning_body,
        default_headers=settings.headers or None,
    )
