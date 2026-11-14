"""OpenAI-compatible chat client (Groq by default). The agent decides; this only transports."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class ToolFormatError(LLMError):
    """The provider returned a tool call we cannot parse. The agent asks the model to retry."""

    def __init__(self, message: str, failed: str = ""):
        super().__init__(message, 200)
        self.failed = failed


@dataclass
class Reply:
    content: str
    tool_calls: list
    model: str
    usage: dict | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw: dict | None = None


@dataclass
class LLMTurn:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openai": "https://api.openai.com/v1",
}


class LLM:
    """Used by agent.py, web.py and the eval harness."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        fallback_models: list[str] | None = None,
        timeout: float = 60,
        sleep=time.sleep,
    ):
        self.provider = (provider or os.environ.get("LLM_PROVIDER") or "groq").strip().lower()
        self.api_key = (api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")).strip()
        self.base_url = (
            (base_url or os.environ.get("LLM_BASE_URL") or PROVIDERS.get(self.provider) or PROVIDERS["groq"])
            .rstrip("/")
        )
        self.model = (model or os.environ.get("LLM_MODEL") or "openai/gpt-oss-20b").strip()
        raw_fallback = os.environ.get("LLM_FALLBACK_MODELS", "")
        self.fallback_models = fallback_models or [
            item.strip() for item in raw_fallback.split(",") if item.strip()
        ]
        if not self.fallback_models:
            self.fallback_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        self.timeout = timeout
        self.sleep = sleep

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages, tools=None, **kw) -> Reply:
        if not self.configured:
            raise LLMError("LLM_API_KEY is empty", 401)
        models = [self.model] + [m for m in self.fallback_models if m != self.model]
        last: Exception | None = None
        for index, model in enumerate(models):
            try:
                return self._complete(model, messages, tools, kw)
            except ToolFormatError:
                raise
            except LLMError as exc:
                last = exc
                if exc.status in {429, 500, 502, 503, 504} and index < len(models) - 1:
                    self.sleep(min(8, 1.5 * (index + 1)))
                    continue
                if exc.status == 400 and "model" in str(exc).lower() and index < len(models) - 1:
                    continue
                raise
        raise last or LLMError("all models failed")

    def ping(self) -> str:
        # gpt-oss spends some completion budget on hidden reasoning before content.
        reply = self.chat(
            [{"role": "user", "content": "Reply with the single word ok."}],
            None,
            temperature=0,
            max_tokens=128,
        )
        return (reply.content or "").strip()

    def probe_tools(self) -> dict | None:
        tool = [{
            "type": "function",
            "function": {
                "name": "echo_word",
                "description": "Echo one word back.",
                "parameters": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}},
                    "required": ["word"],
                },
            },
        }]
        reply = self.chat(
            [{"role": "user", "content": "Call echo_word with the word ready."}],
            tool,
            temperature=0,
            max_tokens=200,
            tool_choice="auto",
        )
        return reply.tool_calls[0] if reply.tool_calls else None

    def list_models(self) -> list[str]:
        body = self._request("GET", "/models", None)
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return []
        return [str(row["id"]).removeprefix("models/") for row in rows if isinstance(row, dict) and row.get("id")]

    def _complete(self, model: str, messages, tools, kw) -> Reply:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kw.get("temperature", 0.2),
            "max_tokens": kw.get("max_tokens", 800),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kw.get("tool_choice", "auto")
        body = self._request("POST", "/chat/completions", payload)
        return _parse_reply(body, model)

    def _request(self, method: str, path: str, payload: dict | None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "acme-intake-agent/1.0",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        delays = (0.0, 1.0, 2.0)
        last_error: LLMError | None = None
        for attempt, delay in enumerate(delays):
            if delay:
                self.sleep(delay)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                    status = response.status
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code in {429, 500, 502, 503, 504} and attempt < len(delays) - 1:
                    last_error = LLMError(detail or str(exc.code), exc.code)
                    continue
                raise LLMError(detail or str(exc.code), exc.code) from exc
            except urllib.error.URLError as exc:
                if attempt < len(delays) - 1:
                    last_error = LLMError(str(exc.reason), 0)
                    continue
                raise LLMError(str(exc.reason), 0) from exc
            if status != 200:
                raise LLMError(raw[:500], status)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LLMError("Model response was not JSON", status) from exc
        raise last_error or LLMError("request failed")


class OpenAICompatible:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60, reasoning_effort: str = "", max_tokens: int = 2048):
        self._inner = LLM(api_key=api_key, model=model, base_url=base_url, timeout=timeout)
        self.base_url = self._inner.base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMTurn:
        reply = self._inner.chat(messages, tools, max_tokens=self.max_tokens)
        calls = [
            ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in reply.tool_calls
        ]
        return LLMTurn(text=reply.content or None, tool_calls=calls)

    def ping(self) -> str:
        return self._inner.ping()

    def probe_tools(self):
        call = self._inner.probe_tools()
        if not call:
            return None
        return ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])

    def list_models(self) -> list[str]:
        return self._inner.list_models()


class ScriptedModel:
    def __init__(self, steps: list[LLMTurn]):
        self.steps = list(steps)
        self.seen: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMTurn:
        del tools
        self.seen.append([dict(message) for message in messages])
        if not self.steps:
            raise LLMError("scripted model ran out of replies")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def assistant_message(turn: LLMTurn) -> dict:
    if not turn.tool_calls:
        return {"role": "assistant", "content": turn.text or ""}
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in turn.tool_calls
        ],
    }


def _parse_reply(body: dict, fallback_model: str) -> Reply:
    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Model response had no message: {str(body)[:300]}", 200) from exc
    content = _text(message.get("content")) or ""
    calls = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        arguments = function.get("arguments", "{}")
        raw_args = arguments if isinstance(arguments, str) else json.dumps(arguments)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ToolFormatError("tool arguments were not valid JSON", raw_args[:200]) from exc
        if not isinstance(arguments, dict):
            raise ToolFormatError("tool arguments must be a JSON object", str(arguments)[:200])
        calls.append(
            {
                "id": str(raw.get("id") or f"call_{index}"),
                "name": str(function.get("name") or ""),
                "arguments": arguments,
            }
        )
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    return Reply(content=content, tool_calls=calls, model=str(body.get("model") or fallback_model), usage=usage)


def _text(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        joined = "\n".join(part for part in parts if part).strip()
        return joined or None
    return str(content)
