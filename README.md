![LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba1020525eea7873f96_LCA-big-green%20(2).svg)

## Introduction

Welcome to LangChain Academy, Introduction to LangGraph!
This is a growing set of modules focused on foundational concepts within the LangChain ecosystem.
Module 0 is basic setup and Modules 1 - 5 focus on building in LangGraph, progressively adding more advanced themes.  Module 6 addresses deploying your agents.
In each module folder, you'll see a set of notebooks. A link to the LangChain Academy lesson is at the top of each notebook to guide you through the topic. Each module also has a `studio` subdirectory, with a set of relevant graphs that we will explore using the LangGraph API and Studio.

> **This fork runs entirely locally.** Instead of the OpenAI API it uses a local,
> OpenAI-compatible LLM server (LM Studio or Ollama); it manages its environment with
> [uv](https://docs.astral.sh/uv/); it runs in **JupyterLab**; and it ships offline
> **mocks** for the external search APIs (Tavily, Wikipedia) so the course works with no
> cloud accounts or API keys. See [Local LLM](#set-up-a-local-llm) and
> [Running offline](#running-offline-with-mocks) below.

## Setup

### Install uv

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

```
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uv reads `.python-version` (3.11) and will download a matching interpreter automatically —
you do not need to install Python yourself.

### Clone repo
```
git clone https://github.com/langchain-ai/langchain-academy.git
cd langchain-academy
```

### Create an environment and install dependencies

```
uv sync
```

This creates a `.venv/` with all dependencies (including JupyterLab) and installs the
small `lc_local` helper package used by the notebooks. Run course commands either by
prefixing them with `uv run` (e.g. `uv run jupyter lab`) or by activating the venv:

```
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### Set up a local LLM

The course talks to any **OpenAI-compatible** local server. Pick one:

#### Option A — LM Studio (default)
1. Install [LM Studio](https://lmstudio.ai/) and download a **tool-capable** instruct model
   (e.g. `qwen2.5-7b-instruct` or `llama-3.1-8b-instruct`). Tool calling and structured
   output are used throughout the course, so the model must support tools.
2. In LM Studio, go to the **Developer / Local Server** tab and **Start Server**
   (default `http://localhost:1234/v1`).
3. Set `LLM_MODEL` (see `.env` below) to the model identifier shown in LM Studio.

#### Option B — Ollama
1. Install [Ollama](https://ollama.com/) and pull a tool-capable model:
   ```
   ollama pull llama3.1
   ```
2. Ollama serves an OpenAI-compatible endpoint at `http://localhost:11434/v1`.
3. In `.env`, set `OPENAI_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=llama3.1`.

#### Option C — mlx-lm server (Apple Silicon)
1. Install the optional extra (Apple Silicon Macs only):
   ```
   uv sync --extra mlx
   ```
2. Start the server with a tool-capable MLX model (downloaded from Hugging Face on first use):
   ```
   uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
   ```
3. In `.env`, set `OPENAI_BASE_URL=http://localhost:8080/v1` and
   `LLM_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit`.

If the Hugging Face model download hangs partway, retry with the Xet backend disabled:
`HF_HUB_DISABLE_XET=1 uv run hf download <model>` before starting the server.

> **Caveat — experimental:** mlx-lm works for the plain-chat lessons, but its tool-calling
> support is currently unreliable: it ignores `tool_choice` entirely (never *forces* a tool
> call), and as of mlx-lm 0.31.x the server often fails to return `tool_calls` even when the
> model emits a valid call (see e.g. [mlx-lm#1262](https://github.com/ml-explore/mlx-lm/issues/1262),
> [mlx-lm#1293](https://github.com/ml-explore/mlx-lm/issues/1293)). Since most modules rely on
> tool calling and structured output, prefer LM Studio (Option A) — on Apple Silicon it can run
> the same MLX models via its MLX engine, with working tool calls.

### Configure environment variables

Copy the example file and edit it to match your setup:

```
cp .env.example .env
```

| Variable          | Default                        | Purpose                                                        |
| ----------------- | ------------------------------ | -------------------------------------------------------------- |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1`     | Local server URL (LM Studio; Ollama = `…:11434/v1`; mlx-lm = `…:8080/v1`). |
| `OPENAI_API_KEY`  | `local`                        | Any non-empty string — local servers ignore it.               |
| `LLM_MODEL`       | `qwen2.5-7b-instruct`          | Must match the model you loaded/pulled (tool-capable).         |
| `USE_MOCKS`       | `true`                         | Use offline mocks for Tavily/Wikipedia (see below).            |

The notebooks call `setup_env()` from the `lc_local` package, which loads `.env` and
applies these defaults, so a fresh checkout runs out of the box once a local model is
serving.

### Running notebooks

```
uv run jupyter lab
```

Make sure the notebook kernel is this project's `.venv` (the default when launched with
`uv run`) so that `import lc_local` resolves.

### Running offline (with mocks)

Some lessons (Module 0 and Module 4) use external web search (Tavily) and Wikipedia. To
keep the course fully offline, those calls are routed through drop-in **mocks** that
return deterministic placeholder data. Mocks are **on by default** (`USE_MOCKS=true`).

To use the real services instead, set `USE_MOCKS=false` in `.env` and provide a
`TAVILY_API_KEY` (sign up at [tavily.com](https://tavily.com/) — Wikipedia needs no key).

### Optional: LangSmith tracing

Tracing is **off by default**. To enable it, add the following to `.env`:

```
LANGSMITH_API_KEY=lsv2-...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-academy
```

Sign up for LangSmith [here](https://docs.langchain.com/langsmith/create-account-api-key#create-an-account-and-api-key).

### Local model notes & troubleshooting

The course leans heavily on **tool calling** and **structured output**, so model choice matters:

* **Use a tool-capable _instruct_ model** (e.g. `qwen2.5-7b-instruct`, `llama-3.1-8b-instruct`).
* **Avoid "reasoning" models** (e.g. the QwQ / qwen3.x *reasoning* variants) with LM Studio for
  this course. They tend to over-reason and return empty content under structured-output /
  forced-tool-call constraints, so `with_structured_output` (Module 4) won't return anything.
* **How structured output works here:** local servers reject the *named-tool* `tool_choice`
  that `with_structured_output` normally uses (LM Studio only accepts `none`/`auto`/`required`).
  The `lc_local` helper and the Module-4 studio graphs route structured output through a single
  tool with `tool_choice="required"` and parse the result — no code change needed in your lessons.
* **Modules 5-6 use [trustcall](https://github.com/hinthornw/trustcall)**, which forces a
  *specific* tool by name — a `tool_choice` form LM Studio rejects (and Ollama / mlx-lm
  silently ignore, so switching servers doesn't help). The local helpers (`lc_local` for
  notebooks, `local_llm.py` in the module-5/6 studio folders) normalize that named
  `tool_choice` to `required`, which is equivalent when a single tool is bound, so trustcall
  works against LM Studio out of the box. A hosted OpenAI-spec endpoint remains the only
  option if you need true named `tool_choice`.
* If a graph hangs or returns empty output, it's almost always the model — switch to a smaller,
  non-reasoning instruct model that fits comfortably in your RAM/VRAM.

### Set up Studio

* Studio is a custom IDE for viewing and testing agents.
* Studio can be run locally and opened in your browser on Mac, Windows, and Linux.
* See documentation [here](https://docs.langchain.com/langsmith/studio#local-development-server) on the local Studio development server.
* Graphs for LangGraph Studio are in the `module-x/studio/` folders for modules 1-5 (and `module-6/deployment/`).
* The studio graphs read the same local-LLM variables from a `.env` file in their own
  folder. Create those from the templates (run from the repo root):

```
for d in module-1/studio module-2/studio module-3/studio module-4/studio module-5/studio module-6/deployment; do
  cp "$d/.env.example" "$d/.env"
done
```

* To start the local development server, activate the venv and run the following command
  in the `studio` directory of each module:

```
langgraph dev
```

You should see the following output:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.
