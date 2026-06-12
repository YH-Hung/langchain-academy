"""Local environment + LLM configuration for LangChain Academy.

Points ``langchain_openai.ChatOpenAI`` at a local OpenAI-compatible server
(LM Studio by default, Ollama as an alternative) so the course runs offline with
no cloud API keys.
"""
import base64
import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Defaults applied by ``setup_env()`` only when a variable is not already set.
DEFAULTS = {
    # LM Studio local server. Ollama: "http://localhost:11434/v1" — mlx-lm: "http://localhost:8080/v1"
    "OPENAI_BASE_URL": "http://localhost:1234/v1",
    # Local servers ignore the key, but the OpenAI client requires it to be non-empty.
    "OPENAI_API_KEY": "local",
    # Must match a tool-capable model you have loaded in LM Studio / pulled in Ollama.
    "LLM_MODEL": "qwen2.5-7b-instruct",
    # Mock Tavily/Wikipedia so the course runs fully offline. Set to "false" for real APIs.
    "USE_MOCKS": "true",
}


def setup_env() -> None:
    """Load a local ``.env`` (if present) and apply offline-friendly defaults.

    Safe to call multiple times. Existing environment variables always win, so a
    ``.env`` file or a shell ``export`` overrides the defaults above. Tracing
    stays disabled unless explicitly opted in: ``LANGFUSE_TRACING=true`` routes
    traces to a local Langfuse (see ``langfuse/docker-compose.yml``), while
    ``LANGSMITH_API_KEY`` keeps the stock cloud LangSmith path.
    """
    load_dotenv(find_dotenv(usecwd=True))
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    _setup_tracing()


def _setup_tracing() -> None:
    """Route LangChain's automatic tracing to local Langfuse, cloud, or off.

    Langfuse mode reuses the ``langsmith`` SDK's OTLP-only export
    (``LANGSMITH_TRACING_MODE=otel``): no LangSmith cloud calls are made, and
    LangChain/LangGraph's built-in tracer sends nested spans to Langfuse's
    OpenTelemetry endpoint instead.
    """
    if os.environ.get("LANGSMITH_API_KEY"):
        return  # explicit cloud LangSmith opt-in: stock env-var path unchanged
    if os.environ.get("LANGFUSE_TRACING", "").lower() in ("1", "true", "yes"):
        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
        public = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-langchain-academy")
        secret = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-langchain-academy")
        auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
        # langsmith passes this endpoint verbatim to OTLPSpanExporter, so it
        # must include the full /v1/traces path.
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_ENDPOINT", f"{host}/api/public/otel/v1/traces"
        )
        os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", f"Authorization=Basic {auth}")
        os.environ["LANGSMITH_TRACING"] = "true"  # activates LangChain's tracer
        os.environ["LANGSMITH_TRACING_MODE"] = "otel"  # OTLP only, no cloud calls
        os.environ.setdefault("LANGSMITH_PROJECT", "langchain-academy")
    else:
        # Fully offline default: tracing stays off.
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)


def _describe_tool(schema):
    """Convert ``schema`` to an OpenAI tool dict, ensuring a non-empty description.

    Local models often will not emit a tool call for a tool whose description is
    empty/null (and some local servers reject a null description outright), so we
    synthesize one from the tool name when the schema has no docstring.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tool = convert_to_openai_tool(schema)
    fn = tool["function"]
    if not fn.get("description"):
        fn["description"] = f"Return a {fn['name']} object."
    return tool


# tool_choice values local servers understand ("any" is LangChain's alias for "required").
_SERVER_TOOL_CHOICES = ("auto", "none", "any", "required")


class LocalChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` tuned for local OpenAI-compatible servers.

    Local servers (LM Studio, Ollama) generally don't implement OpenAI's native
    ``json_schema`` structured output, and they only accept ``tool_choice`` in
    {none, auto, required} — so they reject the named-tool ``tool_choice`` used by
    both the ``json_schema`` and ``function_calling`` methods. This forces a single
    bound tool with ``tool_choice="required"`` and parses the resulting tool call.

    ``bind_tools`` likewise normalizes any named ``tool_choice`` (used by trustcall
    in Modules 5-6, including its internal ``PatchDoc`` update tool) to
    ``"required"``. Trustcall only names a tool when it binds a single one, so with
    one tool bound ``"required"`` is equivalent to forcing it by name.
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if isinstance(tool_choice, dict) or (
            isinstance(tool_choice, str) and tool_choice not in _SERVER_TOOL_CHOICES
        ):
            tool_choice = "required"
        return super().bind_tools(
            [_describe_tool(t) for t in tools], tool_choice=tool_choice, **kwargs
        )

    def with_structured_output(self, schema=None, *, method=None, include_raw=False, **kwargs):
        from langchain_core.output_parsers.openai_tools import (
            JsonOutputKeyToolsParser,
            PydanticToolsParser,
        )

        bound = self.bind_tools([_describe_tool(schema)], tool_choice="required")
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
        else:
            name = _describe_tool(schema)["function"]["name"]
            parser = JsonOutputKeyToolsParser(key_name=name, first_tool_only=True)
        return bound | parser


def chat_model(model: str | None = None, temperature: float = 0, **kwargs) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` bound to the local OpenAI-compatible endpoint."""
    return LocalChatOpenAI(
        model=model or os.environ.get("LLM_MODEL", DEFAULTS["LLM_MODEL"]),
        base_url=os.environ.get("OPENAI_BASE_URL", DEFAULTS["OPENAI_BASE_URL"]),
        api_key=os.environ.get("OPENAI_API_KEY", DEFAULTS["OPENAI_API_KEY"]),
        temperature=temperature,
        **kwargs,
    )
