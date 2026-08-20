"""
llm_client.py
--------------
Thin OpenAI-compatible API wrapper (model, conversation history, calls).
Split out of climbing_coach.py so a provider swap (e.g. to DeepInfra)
only touches this file.
"""

from __future__ import annotations

import logging

from openai import OpenAI

log = logging.getLogger("climbing_coach")
log.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around an OpenAI-compatible chat API.
    Maintains message history for the current session.
    Model-agnostic: works with Mistral, OpenAI, Groq, etc.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._history: list[dict] = []    # user/assistant turns only
        self._system: str = ""

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def set_system(self, system_prompt: str) -> None:
        """Set (or replace) the system prompt for the current session."""
        self._system = system_prompt

    def reset_history(self) -> None:
        """Clear conversation history (start a new session)."""
        self._history = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send a user message, get the assistant reply, update history.
        Returns the assistant's text response.
        """
        self._history.append({"role": "user", "content": user_message})

        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        reply = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def chat_stream(self, user_message: str):
        """
        Send a user message, yield the assistant reply incrementally as it is
        generated, and update history once the stream completes.
        """
        self._history.append({"role": "user", "content": user_message})

        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)

        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            stream=True,
        )

        chunks: list[str] = []
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                chunks.append(delta)
                yield delta

        self._history.append({"role": "assistant", "content": "".join(chunks)})

    def call_once(self, system: str, user_message: str) -> str:
        """
        Single stateless call — does NOT affect history.
        Used for extraction and other one-shot tasks.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.1,   # low temp for structured extraction
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Factory methods for common providers
    # ------------------------------------------------------------------

    @classmethod
    def mistral(cls, api_key: str, model: str = "mistral-large-latest", **kwargs) -> "LLMClient":
        return cls(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            model=model,
            **kwargs,
        )

    @classmethod
    def openai(cls, api_key: str, model: str = "gpt-4o-mini", **kwargs) -> "LLMClient":
        return cls(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            **kwargs,
        )

    @classmethod
    def groq(cls, api_key: str, model: str = "llama-3.1-8b-instant", **kwargs) -> "LLMClient":
        return cls(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model,
            **kwargs,
        )


