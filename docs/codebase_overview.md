# Codebase Analysis Summary

## 1. Project Overview
- **Project Name**: ThinkChain
- **Primary Purpose**: An extensible, tool-aware conversational agent that connects Large Language Models (LLMs) to a rich plugin system, giving users an interactive CLI/terminal chat experience with live tool calls, streaming responses and rich UI.
- **Technology Stack**: Python 3, Anthropic Claude API, Rich, Prompt-Toolkit, Click, SSEClient, dotenv, HTTPX, MCP (remote tool registry)
- **Architecture Pattern**: Monolithic CLI application with a Plugin/Tool discovery subsystem

## 2. Directory Structure
```
thinkchain/
├── run.py                 # Smart launcher – picks best UI implementation
├── thinkchain.py          # Enhanced (Rich) chat UI powered by Anthropic
├── thinkchain_cli.py      # Minimal CLI chat UI (no Rich)
├── tool_discovery.py      # Auto-discovers local & MCP tools, validates & executes
├── tools/                 # Built-in tool implementations
│   ├── base.py            # BaseTool ABC (interface)
│   ├── *.py               # Individual tool classes (weathertool, duckduckgotool…)
│   └── toolcreator.py     # LLM-driven code-generation tool
├── ui_components.py       # Re-usable Rich UI helpers for thinkchain.py
├── mcp_integration.py     # Optional remote-tool (MCP) support
├── docs/                  # Project & research docs
│   ├── P1_research_codebase.md  # Your current instructions template
│   └── …
├── requirements.txt / pyproject.toml
└── .env                   # Holds Anthropic API key, etc.
```

## 3. Core Components & Architecture
### 3.1 Chat Launchers
* **run.py** – Detects installed dependencies and launches either the rich UI (`thinkchain.py`) or fallback CLI (`thinkchain_cli.py`).

### 3.2 UI Implementations
* **thinkchain.py** – Rich-based interface providing coloured output, spinners, progress bars, paneled help, etc. Handles:
  * configuration (`CONFIG` dict)
  * Anthropic streaming loop with thinking & tool_use handling
  * enhanced command processing (/tools, /config …)

* **thinkchain_cli.py** – Text-only version implementing the same logic without Rich.

Both implementations share the same helper functions (`stream_once`, `ask`, config parsing) and rely on **tool_discovery.py** for tool management.

### 3.3 Tool System
* **tools/base.py** – `BaseTool` ABC defining `name`, `description`, `input_schema`, `execute()`.
* **tool_discovery.py** –
  * Discovers local tool classes under `tools/` (via inspection) and optional MCP remote tools.
  * Validates tool schema, builds registry, exposes helpers (`get_claude_tools`, `execute_tool_sync`, etc.).
  * Provides async refresh & execution wrappers.

### 3.4 Built-in Tools
Representative examples:
* `weathertool.py` – fetches weather.
* `duckduckgotool.py` – web search.
* `toolcreator.py` – generates new tools using Claude.
* Several file manipulation tools (create / read / edit).

### 3.5 External Integrations
* **Anthropic Messages API** – Streaming with beta headers for thinking & fine-grained tool streaming.
* **MCP servers** – Optional, supply remote tools; integrated asynchronously.

### Typical Control Flow
1. **run.py** decides UI.
2. UI module starts, builds initial `chat_history` with tool awareness message.
3. User types input ➜ added to transcript.
4. `stream_once()` sends transcript + tool definitions to Anthropic.
5. Claude may emit `tool_use` blocks ➜ local `run_tool()` executes via `tool_discovery` ➜ results appended to transcript.
6. Loop continues until final text response printed.

## 4. Key Patterns & Conventions
- **Plugin Discovery**: Dynamic import & validation of tool classes.
- **Function-Calling (tool_use) Streaming**: Model emits tool instructions mid-stream; program executes & continues the conversation.
- **Separation of UIs**: Same engine logic duplicated in rich & plain-text variants.
- **Config via /commands**: In-chat slash-command pattern for config & tool management.
- **Error Handling**: try/except with fallback to non-streaming request.
- **Concurrency**: Async functions for MCP; sync wrappers for CLI.

## 5. Data Architecture
- **Storage**: No DB; in-memory transcript list per session.
- **Models**: Simple dict structures (`{"role": "user"|"assistant", "content": …}`) mirroring Anthropic schema.
- **Access Patterns**: Passed directly to API call; persisted only for session duration.

## 6. Integration Points for New Feature (Gemini Parallel Version)
| Component | Why it needs change |
|-----------|--------------------|
| `thinkchain.py` / `thinkchain_cli.py` | Hard-wired to `anthropic.Anthropic`. Need siblings `thinkchain_gemini.py`, `thinkchain_cli_gemini.py` using `google.genai` SDK and Gemini 2.5 models. |
| `run.py` | Add detection flag & new launch branch to start Gemini variant (e.g. `--gemini` CLI arg or env var). |
| `toolcreator.py` | Clone to `gemini_toolcreator.py` generating code with Gemini instead of Claude. |
| `requirements.txt` | Add `google-genai` package. |

Implementation approach: Re-use **tool_discovery.py**, **ui_components.py** and all tools unchanged; only swap the LLM client & streaming interface. Gemini supports similar function-calling (`tools=` param) via SDK 0.6+.

## 7. Historical Context & Rationale
Initial commits focus on Anthropic integration; later commits added rich UI, MCP support and tool discovery refresh. No prior Gemini experimentation found.

## 8. Open Questions for Product Owner / Tech Lead
1. Should the launcher automatically choose Gemini if an ENV var like `GEMINI_API_KEY` is present?
2. Required Gemini model (e.g., `gemini-1.5-flash-latest` vs `gemini-1.5-pro-latest`)?
3. Desired parity for thinking / streaming visuals? (Gemini lacks `thinking` block concept.)
4. Rate-limit & cost considerations when running both Anthropic & Gemini variants concurrently.

## 9. Initial Hypothesis for Integration
- Fork `thinkchain.py` ➜ `thinkchain_gemini.py` and switch:
  * `import anthropic` ➜ `import google.genai as genai`
  * `client = genai.Client(...)` (tools param etc.)
  * Replace SSE streaming loop with Gemini SDK `client.models.generate_content_stream()`.
- Similar fork for CLI variant.
- Provide new CLI entry (`run_gemini.py`) or extend existing `run.py` to accept `--gemini` flag.
- Add `gemini_toolcreator.py` using Gemini.

## ✅ Self-check Log
- [x] Directory tree paths verified with `list_dir` tool.
- [x] Confirmed key modules & their responsibilities via `view_file`.
- [x] Identified exact integration points for Gemini feature.
- [x] Summary follows the structure required by P1 template.
- [x] No ambiguous terms or unresolved placeholders remain.
