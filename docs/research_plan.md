# Feature Implementation Research Plan and Findings

## Executive Summary
This research investigates how to add a **Gemini / google-genai–based** execution path to ThinkChain alongside the existing Anthropic path.  The key result is that the new SDK (`google-genai >= 0.6.0`) already supports **streaming**, **function-calling tools** and **automatic function execution**, allowing near-drop-in replacement of Claude logic.  Recommended approach is to fork the two UI entry points (`thinkchain.py`, `thinkchain_cli.py`) into Gemini variants and adapt the request/stream loop to:
1. Use `genai.Client()` and `client.models.generate_content_stream()`.
2. Register tools via `tools=[genai.types.FunctionDeclaration(...)]`.
3. Handle streamed chunks (`for chunk in stream:`) that may include `chunk.candidates[0].content.parts`, `FunctionCall` or `FunctionResponse` parts.
4. Keep the existing `tool_discovery` registry unchanged; only adapt the message mapping layer.

The SDK expects the **API key in `GOOGLE_API_KEY`** (or explicit `genai.configure(api_key="…")`).  Unlike Anthropic, Gemini has no “thinking” block; we can synthesise a spinner until the first text chunk arrives.

---

## Part 1: Research Plan

### 1. Feature Breakdown
| # | Requirement | Technical Challenge | Priority | Codebase Constraint |
|---|-------------|--------------------|----------|---------------------|
|1|Call Gemini models with streaming responses|Adapting Claude SSE reader to Gemini iterator|High|`thinkchain.*` stream loop deeply tied to Anthropic API|
|2|Expose tool/function calling|Translate tool schemas to `genai.types.FunctionDeclaration`|High|`tool_discovery` returns JSON schema already |
|3|Automatic execution of tools|Mirror current `tool_use` handling but for `FunctionCall` parts|High|Keep same registry & execution helpers|
|4|CLI flag / env detection|Allow `--gemini` or env var to switch engines|Medium|`run.py` selects UI based on deps|
|5|Progress UI parity|Replace Anthropic “thinking” events|Low|Rich spinners in `ui_components.py`|
|6|API key management|Read from `.env` or system env|Medium|Existing dotenv loader for Anthropic key|
|7|Cost/latency tuning|Select model (`gemini-1.5-flash` vs `pro`) & generation config|Low|Expose `/config` command similar to existing|

### 2. Research Areas

#### Area A: google-genai SDK – Streaming & Basic Usage
Objective: Confirm how to create client, call `generate_content_stream`, parse chunks, and read `.text`.
Key Questions:
- Which client class & method provide streaming?
- What structure do stream chunks have?
Search Queries:
1. "generate_content_stream google genai python" – expect official sample (fallback: "streamGenerateContent Python").
2. "google genai Client models.generate_content example" – expect README.
Validation: source must be official docs / GitHub; dated ≥ 2024-06.
Integration: drives rewrite of `stream_once`.

#### Area B: SDK Function Calling / Tools
Objective: Map ThinkChain JSON schemas to Gemini function declarations and parse calls.
Key Questions:
- How to declare tools in the request?
- How are `FunctionCall` & `FunctionResponse` represented in stream?
Search Queries:
1. "google genai python function calling" (fallback: vertex AI function calling python).
2. "google/genai automatic function calling".
Validation: docs & sample notebook.

#### Area C: Auth & Configuration
Objective: Understand env vars, client options, rate limits.
Key Questions: which env var, how to set model version, generation config fields.
Search Queries: "GOOGLE_API_KEY genai", "generation_config python genai".

#### Area D: Performance & Cost Optimisation
Objective: Compare flash vs pro, token limits, streaming chunk size.
Queries: "gemini flash vs pro token limit", "google genai max_output_tokens".

#### Area E: CLI UX Patterns for Streaming LLMs
Objective: Best practices for rendering live chunks in Rich.
Queries: "python rich live streaming generator", "cli spinner while waiting".

### Validation Criteria (all areas)
- Source credibility: Official Google docs/blog/GitHub – High; StackOverflow – Medium.
- Applicability: Must reference SDK ≥ 0.6 (post-May 2025 migration).
- Recency: Published 2025 or late 2024.

---

## Part 2: Research Findings

### Area A – Streaming
Search Query: "generate_content_stream google genai python"
- **Source 1**: *API reference – models.streamGenerateContent* (<https://ai.google.dev/api/generate-content>) – 2025-05 (High)
  - `client.models.generate_content_stream(model="gemini-1.5-flash", contents="…")`
  - Iterate: `for chunk in stream: chunk.text` or inspect `chunk.candidates[0].content.parts`.
  - Supports `GenerationConfig` (temperature, max_output_tokens).
  - Applicability: High; exactly replaces SSE loop.
- **Source 2**: *python-genai README* (<https://github.com/googleapis/python-genai>) – 2025-05 (High)
  - Demo with `for chunk in client.models.generate_content_stream(...)` printing incremental text.
  - Shows `genai.configure(api_key="…")` shortcut.

Synthesis: Use iterator; first non-empty `.text` can stop spinner.  Streaming yields both text and function-call parts.

### Area B – Function Calling / Tools
Search Query: "google genai python function calling"
- **Source 1**: *Function Calling docs* (<https://ai.google.dev/gemini-api/docs/function-calling>) – 2025-06 (High)
  - Declare tools via `tools=[{
      "function_declarations": [{"name":…, "description":…, "parameters":{…}}]
    }]` or using helper `genai.functions.FunctionDeclaration`.
  - Stream chunks may include `FunctionCall` → `name`, `args`, then model expects a `FunctionResponse` part.
- **Source 2**: *Colab sample* (<https://colab.research.google.com/github/google/generative-ai-docs/.../function-calling/python.ipynb>) – 2025-05 (High)
  - Demonstrates automatic function calling with python callable passed in `tools`.

Synthesis: Convert each ThinkChain tool’s JSON schema into `FunctionDeclaration`; on `FunctionCall`, invoke `tool_discovery.execute_tool_sync` and send back `FunctionResponse` via `stream.send_tool_response()` (SDK helper).

### Area C – Auth & Config
- **Source**: *SDK doc site* (<https://googleapis.github.io/python-genai/>) – 2025-05 (High)
  - `export GOOGLE_API_KEY=<key>` or `genai.configure(api_key="…")`.
  - Optional `http_options=genai.types.HttpOptions(api_version="v1beta")`.

### Area D – Performance
- **Source**: *Vertex AI inference guide* (<https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference>) – 2025-04 (Medium)
  - `gemini-1.5-flash`: 128k context, lower latency/cost.
  - `gemini-1.5-pro`: higher quality, higher cost.
  - Recommend default to `flash` with `/config` override.

### Area E – CLI UX Patterns
- **Source**: *Rich live update docs* (<https://rich.readthedocs.io/en/stable/live.html>) – 2024 (High)
  - Use `Live` context manager to update panel per chunk.

---

## 3. Cross-Cutting Insights
- **Design Pattern**: *Adapter* – keep common tool & chat history logic, plug different LLM adapters (AnthropicAdapter, GeminiAdapter).
- **Testing**: Mock `genai.Client` using dependency injection; unit-test parsing of `FunctionCall` parts.
- **Security**: Never log API key; enforce `safety_settings` like `HarmBlockThreshold`.

## 4. Technical Decisions Matrix
| Decision | Option 1 | Option 2 | Recommendation | Rationale |
|----------|----------|----------|----------------|-----------|
|Model default|`gemini-1.5-flash`|`gemini-1.5-pro`|Flash|Lower cost, adequate for CLI assistant|
|Tool declaration|Manual JSON dict|`genai.types.FunctionDeclaration` helper|Helper|Less boilerplate|
|Streaming loop|Separate thread|Sync iterator in async wrapper|Iterator|Simpler, mirrors current sync loop|
|API key config|Require env|Allow CLI `--key` param|Env|Matches existing pattern|

## 5. Implementation Recommendations
1. **Create `llm_adapter.py`** with abstract base and two concrete classes – defer choice to runtime.
2. **Fork UIs**: `thinkchain_gemini.py` & `thinkchain_cli_gemini.py` import `GeminiAdapter`.
3. **Update `run.py`** to detect `--gemini` flag or `GEMINI_DEFAULT=true` env.
4. **Add `google-genai>=0.6.0`** to `requirements.txt`.
5. **Map tools**: Extend `tool_discovery.get_claude_tools()` to return Gemini declarations as well.
6. **Spinner**: Use Rich `Live` until first chunk.
7. **Docs**: Update README with setup instructions & env var.

## 6. Gaps & Uncertainties
- Automatic vs manual function calling limits (max 10 automatic calls) – may need loop.
- Streaming error handling (`chunk.usage_metadata`) shape.
- Whether `stream.send_tool_response()` is synchronous in SDK ≥ 0.6.

## 7. References
1. Generating content & streaming – <https://ai.google.dev/api/generate-content>
2. python-genai README – <https://github.com/googleapis/python-genai>
3. Function Calling docs – <https://ai.google.dev/gemini-api/docs/function-calling>
4. Function-calling Colab sample – Google generative-ai docs notebook.
5. Vertex AI inference guide – <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference>
6. Rich live update – <https://rich.readthedocs.io/en/stable/live.html>
