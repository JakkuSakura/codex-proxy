# Codex GPT Model Research & Gemini Porting Guide

This document maps out the architecture, behavior, and protocol of the native Codex GPT model as implemented in `codex-rs` and `codex-cli`. It serves as the definitive reference for ensuring 1:1 parity in the `codex-proxy` for the Gemini model.

## 1. Protocol Architecture: The Responses API

The native Codex model primarily communicates via the **Responses API**, which is a high-level orchestration layer over standard chat completions.

### Wire Protocols
- **Responses API (Modern)**: Endpoint `/v1/responses`. Uses a granular, item-based SSE stream.
- **Chat Completions (Legacy)**: Endpoint `/v1/chat/completions`. Used for backward compatibility with standard OpenAI providers.

### Request Structure (`ResponsesApiRequest`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `model` | String | Model ID (e.g., `gpt-5.1-preview`). |
| `instructions` | String | System-level instructions for the agent. |
| `input` | `ResponseItem[]` | Conversation history including messages, tool calls, and tool outputs. |
| `tools` | JSON[] | Function definitions. |
| `parallel_tool_calls`| Boolean | Whether the model can invoke multiple tools at once. |
| `reasoning` | Object | Config for reasoning effort and summary detail. |
| `store` | Boolean | Enables server-side conversation storage. |
| `stream` | Boolean | Enables SSE streaming. |
| `text` | Object | Controls verbosity (`low`, `medium`, `high`) and JSON schema output. |

## 2. Granular Message Model: `ResponseItem`

Unlike standard APIs that only use `role` and `content`, the Responses API uses a polymorphic `ResponseItem` structure.

| Type | Fields | Behavior |
| :--- | :--- | :--- |
| `message` | `role`, `content` | Standard assistant/user/developer messages. |
| `reasoning` | `summary[]`, `content` | Detailed "thought" blocks. `summary` is used for UI progress bars. |
| `function_call` | `name`, `arguments`, `call_id` | Standard tool invocation. |
| `web_search_call` | `action`, `status` | Specific trigger for web search (Search, OpenPage, FindInPage). |
| `local_shell_call` | `action`, `status` | Triggers local terminal command execution. |
| `function_call_output`| `call_id`, `output` | Tool results. Output can be a string or an array of `ContentItem` (text/images). |

## 3. SSE Stream Events

The Responses API stream (`text/event-stream`) emits structured events rather than raw text chunks.

1.  **`response.created`**: Fired when the session starts.
2.  **`response.output_item.added`**: Signals a new item (e.g., a reasoning block or a tool call) has started.
3.  **`response.output_text.delta`**: Incremental text for the current message.
4.  **`response.reasoning_text.delta`**: Incremental text for the reasoning block.
5.  **`response.reasoning_summary_text.delta`**: Updates the summary string in the UI.
6.  **`response.output_item.done`**: Finalizes a `ResponseItem` (includes the full object).
7.  **`response.completed`**: Final event with full session summary and token usage.
8.  **`response.failed`**: Error event with specific Codex error codes.

## 4. Gemini Mapping Strategy (Deep Parity)

To achieve native-level parity, `codex-proxy` maps Gemini's internal capabilities to these structures.

### Reasoning & Thinking
- **Gemini `thought`**: Mapped to `reasoning` items.
- **Summary Extraction**: The proxy uses Regex (`\*\*(.*?)\*\*`) to extract headers from Gemini's thoughts and re-emits them as `reasoning_summary_text.delta` events to drive the Codex UI progress indicators.

### Tool & Shell Execution
- **`shell` / `container.exec`**: Gemini tool calls with these names are automatically mapped to `local_shell_call` to trigger Codex's native sandboxed terminal.

### Session State
- **`x-codex-turn-state`**: The proxy forwards this header bidirectionally. It allows the Gemini backend to persist context across turns without the client holding the full state.

### Context Compaction
- **Subagent Detection**: If `x-openai-subagent` is `compact`, the proxy switches to a faster model (e.g., `gemini-2.5-flash-lite`) and uses a specialized prompt to summarize the history into a `compaction_summary` item.

## 5. Model Variant: gpt-5.2-codex

`gpt-5.2-codex` (internal codename `bengalfox`) is the latest flagship model variant. It introduces several refinements over the base GPT-5 Codex model.

### Key Characteristics
- **Context Window**: 272,000 tokens.
- **Reasoning Levels**: Supports an additional `xhigh` reasoning effort level (`supported_reasoning_level_low_medium_high_xhigh`).
- **Shell Type**: Uses `ShellCommand` (rather than the default or legacy types), enabling more robust terminal interactions.
- **Patching**: Uses `ApplyPatchToolType::Freeform`, allowing the model to generate and apply patches without strict schema enforcement.
- **Parallel Tool Calls**: Explicitly enabled.
- **Reasoning Summaries**: Native support for driving UI progress indicators via SSE events.

### Specialized Instructions & Personality
`gpt-5.2-codex` uses a modular instruction system:
- **Base Instructions**: Defined in `gpt-5.2-codex_prompt.md`. Focuses on:
    - Preferring `rg` over `grep` for speed.
    - ASCII-first editing constraints.
    - Git-aware behavior (NEVER revert user changes, don't amend unless asked).
    - Frontend design mandates: "Avoid AI slop," aim for bold, intentional UI (expressive fonts, gradients, motion).
- **Template System**: Uses `gpt-5.2-codex_instructions_template.md` which injects a `{{ personality }}` variable.
- **Personalities**: Supports two distinct modes:
    - **Friendly**: More conversational and collaborative.
    - **Pragmatic**: Direct, professional, and efficiency-focused.

## 6. Parity Checklist for Future Work
- [ ] **Error Mapping**: Ensure `response.failed` codes like `context_length_exceeded` and `rate_limit_exceeded` match Gemini's 400/429 status codes.
- [ ] **Image Support**: Fully map `InputImage` items in `ResponseItem` to Gemini's multimodal parts.
- [ ] **Search Grounding**: Ensure `groundingMetadata` from Gemini correctly populates `web_search_call` actions.
- [ ] **Verbosity Control**: Ensure the `text.verbosity` field from Codex correctly modifies the Gemini system instruction.
- [ ] **GPT-5.2 Persona Injection**: Map Codex's personality requests (Friendly/Pragmatic) to Gemini's system instructions.
