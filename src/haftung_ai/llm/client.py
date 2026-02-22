"""Groq API client with rate limiting, retries, and JSON mode."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Generator

from pydantic import BaseModel

from haftung_ai.config.settings import get_settings

logger = logging.getLogger(__name__)


class GroqClient:
    """Groq LLM client for LLaMA 3.3 70B with structured output support."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
    ):
        settings = get_settings()
        self.api_key = settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        self.max_tokens = max_tokens or settings.GROQ_MAX_TOKENS
        self.max_retries = max_retries if max_retries is not None else settings.GROQ_MAX_RETRIES
        self._rate_limit_rpm = settings.GROQ_RATE_LIMIT_RPM
        self._last_call_time = 0.0
        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def _rate_limit(self) -> None:
        if self._rate_limit_rpm <= 0:
            return
        min_interval = 60.0 / self._rate_limit_rpm
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("Groq API attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Groq API call failed after all retries")

    def invoke_json(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        """Generate JSON completion using Groq's JSON mode."""
        messages = []
        sys_msg = (system_prompt or "") + "\nYou must respond with valid JSON only."
        messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content.strip()
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed on attempt %d: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    raise
            except Exception as e:
                logger.warning("Groq API attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Groq JSON API call failed after all retries")

    def invoke_structured(self, prompt: str, output_model: type[BaseModel], system_prompt: str | None = None) -> BaseModel:
        """Generate structured output validated against a Pydantic model."""
        schema_str = json.dumps(output_model.model_json_schema(), indent=2)
        enhanced_prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_str}"
        raw = self.invoke_json(enhanced_prompt, system_prompt=system_prompt)
        return output_model.model_validate(raw)

    def stream(self, prompt: str, system_prompt: str | None = None) -> Generator[str, None, None]:
        """Stream text tokens."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        self._rate_limit()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def judge(self, prompt: str) -> str:
        """Deterministic evaluation call (temperature=0)."""
        messages = [{"role": "user", "content": prompt}]
        self._rate_limit()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()
