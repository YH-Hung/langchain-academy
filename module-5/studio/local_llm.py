"""Local-server-friendly ``ChatOpenAI`` for the module-5 studio graphs.

Local servers (LM Studio, Ollama) only accept ``tool_choice`` in {none, auto,
required}, so they reject the named-tool ``tool_choice`` that trustcall uses to
force its extraction schema (and its internal ``PatchDoc`` update tool). Trustcall
only names a tool when it binds a single one, so normalizing the named choice to
``tool_choice="required"`` is equivalent and keeps the request server-compatible.
"""
import os

from langchain_openai import ChatOpenAI

# tool_choice values local servers understand ("any" is LangChain's alias for "required").
_SERVER_TOOL_CHOICES = ("auto", "none", "any", "required")


class LocalChatOpenAI(ChatOpenAI):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if isinstance(tool_choice, dict) or (
            isinstance(tool_choice, str) and tool_choice not in _SERVER_TOOL_CHOICES
        ):
            tool_choice = "required"
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)


def chat_model(temperature: float = 0, **kwargs) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` bound to the local OpenAI-compatible endpoint."""
    return LocalChatOpenAI(
        model=os.environ.get("LLM_MODEL", "qwen2.5-7b-instruct"),
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "local"),
        temperature=temperature,
        **kwargs,
    )
