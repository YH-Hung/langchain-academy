"""Local-server-friendly structured output for the module-4 studio graphs.

LM Studio / Ollama only accept ``tool_choice`` in {none, auto, required}, so they
reject the named-tool ``tool_choice`` that ``ChatOpenAI.with_structured_output``
normally uses. This helper forces a single bound tool with ``tool_choice="required"``
and parses the resulting tool call into the pydantic ``schema``. It also injects a
description when the schema has none, since local models often won't emit a tool call
for a tool with an empty/null description.
"""
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool


def structured_output(llm, schema):
    """Return a runnable that yields an instance of ``schema`` (a pydantic model)."""
    tool = convert_to_openai_tool(schema)
    fn = tool["function"]
    if not fn.get("description"):
        fn["description"] = f"Return a {fn['name']} object."
    return llm.bind_tools([tool], tool_choice="required") | PydanticToolsParser(
        tools=[schema], first_tool_only=True
    )
