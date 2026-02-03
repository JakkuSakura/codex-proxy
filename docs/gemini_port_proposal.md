# Technical Specification: 1:1 Gemini Port to GPT-5.2-Codex Parity

This document defines the exact implementation requirements to replicate the `gpt-5.2-codex` (Bengalfox) behavior in `codex-proxy`. All "abstract" goals have been converted into concrete file operations and logic specifications.

## 1. Asset Management (Direct File Sync)
We will store the native Codex prompts within the proxy to ensure bit-for-bit instruction parity.

| Source Path (reference/codex/) | Destination Path (src/codex_proxy/assets/prompts/) |
| :--- | :--- |
| `codex-rs/core/gpt-5.2-codex_prompt.md` | `gpt_5_2_base.md` |
| `codex-rs/core/templates/model_instructions/gpt-5.2-codex_instructions_template.md` | `gpt_5_2_template.md` |
| `codex-rs/core/templates/personalities/gpt-5.2-codex_friendly.md` | `personality_friendly.md` |
| `codex-rs/core/templates/personalities/gpt-5.2-codex_pragmatic.md` | `personality_pragmatic.md` |

## 2. Instruction Engine (`src/codex_proxy/providers/gemini_utils.py`)
Replace the existing instruction logic with a strict template-based builder.

### `get_gpt52_instructions(personality_type: str) -> str`
1.  Load `gpt_5_2_template.md`.
2.  If `personality_type` is "friendly", load `personality_friendly.md`.
3.  If `personality_type` is "pragmatic", load `personality_pragmatic.md`.
4.  If none, use an empty string.
5.  Replace `{{ personality }}` in the template with the loaded personality text.
6.  Prepend the content of `gpt_5_2_base.md` to the resulting string.
7.  Append the "High-Quality Output Rules" (Header Title Case, Flat Bullets, Patch Grammar) as seen in `src/codex_proxy/providers/gemini.py`.

## 3. Protocol Mapping (Deep Parity)

### Reasoning Effort (`src/codex_proxy/providers/gemini.py`)
Map the Codex `reasoning.effort` field to Gemini's `thinkingConfig`.
- **`low`**: `thinkingLevel: "low"` (Gemini 3) or `thinkingBudget: 4096` (Gemini 2.0).
- **`medium`**: `thinkingLevel: "medium"` or `thinkingBudget: 16384`.
- **`high`**: `thinkingLevel: "high"` or `thinkingBudget: 32768`.
- **`xhigh`**: `thinkingLevel: "high"` + explicit "Be extremely thorough" system instruction + `thinkingBudget: 65536`.

### Tool Calling Logic
- **Parallelism**: Force `toolConfig.functionCallingConfig.mode = "ANY"` when `supports_parallel_tool_calls` is detected or for `gpt-5.2-codex`.
- **Patching**: For `ApplyPatchToolType::Freeform`, do NOT provide a strict JSON schema for the patch; allow the model to output the diff format specified in `gpt-5.2-codex_prompt.md`.

## 4. Item-Based SSE Events (`src/codex_proxy/providers/gemini_stream.py`)

### Summary Extraction Regex
Update the extraction to use the exact patterns expected by the Codex UI:
- **Pattern**: `re.compile(r'\*\*(.*?)\*\*')`
- **Logic**: For every match in the `thought` stream:
    1.  If the header is new, emit `response.reasoning_summary_part.added`.
    2.  Emit `response.reasoning_summary_text.delta` with the header text.
    3.  Emit `response.reasoning_text.delta` for the raw thought content.

### Quota & Model Cap Headers
If Gemini returns a 429 error:
-   **Status Code**: Return `200 OK` (to keep the SSE stream alive).
-   **Event**: Emit `response.failed` with `code: "usage_limit_exceeded"`.
-   **Headers**: 
    - `x-codex-model-cap-model`: The requested model slug.
    - `x-codex-model-cap-reset-after-seconds`: Derived from Gemini's `retryDelay` (default 60).

## 5. Task Checklist

- [ ] **Filesystem**: Create `src/codex_proxy/assets/prompts/` and copy the 4 native files.
- [ ] **Config**: Add `DEFAULT_PERSONALITY = "pragmatic"` to `src/codex_proxy/config.py`.
- [ ] **Utils**: Implement `get_gpt52_instructions` and integrate it into `map_messages` in `gemini_utils.py`.
- [ ] **Provider**:
    - [ ] Add personality detection from `x-codex-personality` header.
    - [ ] Implement the 4-tier `reasoning.effort` mapping.
    - [ ] Enforce `mode: ANY` for tool calls.
- [ ] **Streaming**:
    - [ ] Update `stream_responses_loop` to match the regex logic in Section 4.
    - [ ] Ensure `models_etag` is emitted as `v1-gemini-gpt-5-2-parity`.
- [ ] **Verification**: Create `tests/integration/test_gpt52_parity.py` and verify:
    - [ ] System prompt contains the "ASCII-first" and "Anti-slop" sections.
    - [ ] Reasoning summaries drive the `response.reasoning_summary_text.delta` event.
    - [ ] `x-codex-turn-state` is forwarded.