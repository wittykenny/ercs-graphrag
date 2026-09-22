from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class LLMClient:
    """Small OpenAI-compatible chat client.

    Set OPENAI_API_KEY and LLM_MODEL to enable it. OPENAI_BASE_URL is optional
    and defaults to the OpenAI chat completions endpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 90,
        extra_body: dict | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        base = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.url = base.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.extra_body = extra_body if extra_body is not None else load_extra_body()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
        if not self.enabled:
            raise RuntimeError("LLM is disabled. Set OPENAI_API_KEY and LLM_MODEL.")
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.extra_body:
            payload.update(self.extra_body)
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc}") from exc
        return data["choices"][0]["message"]["content"]

    def validate_json_array(self) -> list[dict]:
        """Run a minimal JSON-array completion to verify credentials and model."""
        text = self.complete(
            system="Return JSON only.",
            user=(
                "Return exactly this JSON array, with no extra text: "
                '[{"head":"OpenAI","relation":"created","tail":"ChatGPT","evidence":"OpenAI created ChatGPT."}]'
            ),
            max_tokens=120,
        )
        return parse_json_array(text)


def parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.I | re.M).strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array.")
    return [x for x in data if isinstance(x, dict)]


def load_extra_body() -> dict:
    raw = os.getenv("LLM_EXTRA_BODY_JSON", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM_EXTRA_BODY_JSON must be a JSON object.")
    return data
