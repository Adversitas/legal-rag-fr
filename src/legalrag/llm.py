"""Pluggable LLM backends behind one tiny interface.

- AnthropicLLM: Claude through the official Anthropic SDK.
- OpenAICompatLLM: any OpenAI-compatible server - LM Studio, Ollama, vLLM - for free,
  private, local inference.

The pipeline and the evaluation harness only see `LLM.generate`, so backends can be
compared on identical retrieval results.
"""

from typing import Protocol

from .config import Settings

# Models that accept the server-side refusal fallback parameter
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 4096, effort: str | None = "low"):
        import anthropic

        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or `ant auth login`
        self.name = model
        self.max_tokens = max_tokens
        self.effort = effort

    def generate(self, system: str, user: str) -> str:
        kwargs: dict = {}
        if self.effort and "haiku" not in self.name:
            kwargs["output_config"] = {"effort": self.effort}
        if self.name in _FALLBACK_MODELS:
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        response = self.client.beta.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"Model declined to answer ({response.stop_details})")
        return "".join(b.text for b in response.content if b.type == "text").strip()


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str, api_key: str = "not-needed", max_tokens: int = 4096):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.name = model
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()


def make_llm(settings: Settings, model: str | None = None) -> LLM:
    model = model or settings.llm_model
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(model, settings.max_tokens, settings.llm_effort)
    return OpenAICompatLLM(model, settings.llm_base_url, settings.llm_api_key, settings.max_tokens)
