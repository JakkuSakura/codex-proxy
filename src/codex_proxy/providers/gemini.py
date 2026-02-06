import json
import time
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple, cast

from ..config import config
from ..auth import GeminiAuth, AuthError
from ..utils import create_session, json_loads, json_dumps
from .base import BaseProvider
from .gemini_utils import map_messages, sanitize_params
from .gemini_stream import stream_responses_loop

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models via multiple authentication types."""

    def __init__(self):
        self.auth = GeminiAuth()
        self.session = create_session()

    def handle_request(self, data: Dict[str, Any], handler: Any) -> None:
        """Entry point for standard Gemini requests."""
        if config.debug_mode:
            try:
                # Save last request for debugging
                with open("/tmp/last_proxy_request.json", "w") as f:
                    json.dump(data, f)
            except Exception:
                pass

        self._stream_request(data, handler)

    def handle_compact(self, data: Dict[str, Any], handler: Any) -> None:
        """Handle context compaction using configured compaction model."""
        try:
            auth_ctx = self.auth.get_auth_context()

            compaction_model = config.compaction_model or self._get_compaction_model(
                data
            )

            # Map messages specifically for compaction
            contents, system_instruction = map_messages(
                data.get("input", []), compaction_model
            )

            compaction_prompt = data.get(
                "instructions", "Summarize the conversation history concisely."
            )
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"Perform context compaction. instructions: {compaction_prompt}"
                        }
                    ],
                }
            )

            gen_config = {"temperature": 0.1, "maxOutputTokens": 4096}

            if auth_ctx["type"] == "internal":
                url = f"{config.gemini_api_internal}/v1internal:streamGenerateContent?alt=sse"
                headers = {"Authorization": f"Bearer {auth_ctx['access_token']}"}
                request_body = {
                    "model": compaction_model,
                    "project": auth_ctx["project_id"],
                    "request": {
                        "contents": contents,
                        "generationConfig": gen_config,
                    },
                }
                if system_instruction:
                    request_body["request"]["systemInstruction"] = system_instruction
            else:
                # Public API
                url = f"{config.gemini_api_public}/v1beta/models/{compaction_model}:streamGenerateContent?alt=sse&key={auth_ctx['api_key']}"
                headers = {}
                request_body = {
                    "contents": contents,
                    "generationConfig": gen_config,
                }
                if system_instruction:
                    request_body["systemInstruction"] = system_instruction

            resp = self.session.post(
                url,
                data=json_dumps(request_body),
                headers=headers,
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()

            final_text = self._collect_sync_text(resp)

            # Aligned with OpenAI Responses API compaction spec
            result = {
                "output": [{"type": "compaction", "encrypted_content": final_text}]
            }

            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json_dumps(result))

        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            handler.send_error(500, str(e))

    def _collect_sync_text(self, resp: requests.Response) -> str:
        """Helper to collect text from a streaming response synchronously."""
        final_text = ""
        for line in resp.iter_lines():
            if line.startswith(b"data: "):
                try:
                    d = json_loads(line[6:])
                    # Public and internal API return slightly different paths for candidates
                    candidates = d.get("candidates") or d.get("response", {}).get(
                        "candidates"
                    )
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "text" in p:
                            final_text += p["text"]
                except Exception:
                    continue
        return final_text

    def _get_compaction_model(self, req_data: Dict[str, Any]) -> str:
        """Get appropriate compaction model from request or config."""
        requested_model = req_data.get("model", "")
        if requested_model and config.compaction_model:
            return config.compaction_model
        return config.models[0] if config.models else "gemini-2.5-flash-lite"

    def _stream_request(self, req_data: Dict[str, Any], handler: Any) -> None:
        """Managed streaming request with retry and fallback logic."""
        is_responses_api = bool(req_data.get("_is_responses_api", False))
        requested_model = req_data.get(
            "model", config.models[0] if config.models else "gemini-2.5-flash"
        )

        # Primary attempt with potential retry
        for attempt in range(2):
            try:
                self._execute_stream(requested_model, req_data, handler)
                return
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt == 0:
                    retry_delay = self._get_retry_delay(e.response)
                    logger.warning(
                        f"Model {requested_model} rate limited. Waiting {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue

                if e.response.status_code != 429:
                    self._report_error(handler, e, is_responses_api)
                    return
                break
            except Exception as e:
                self._report_error(handler, e, is_responses_api)
                return

        # Fallback logic
        fallback_model = config.fallback_models.get(
            requested_model, self._smart_fallback(requested_model)
        )

        if not fallback_model:
            logger.warning(f"No fallback model configured for {requested_model}")
            error = Exception(f"No fallback available for model {requested_model}")
            self._report_error(handler, error, is_responses_api)
            return

        logger.info(f"Trying fallback model: {fallback_model}")
        try:
            self._execute_stream(
                fallback_model, req_data, handler, display_model=requested_model
            )
        except Exception as e:
            setattr(e, "model", requested_model)
            self._report_error(handler, e, is_responses_api)

    def _get_retry_delay(self, response: requests.Response) -> float:
        """Extract retry delay from Google API error response."""
        try:
            error_data = response.json()
            for detail in error_data.get("error", {}).get("details", []):
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    delay_str = detail.get("retryDelay", "1s")
                    return float(delay_str.rstrip("s"))
        except Exception:
            pass
        return 1.0

    def _smart_fallback(self, model: str) -> Optional[str]:
        """Smart fallback based on model name patterns."""
        if model == "gemini-2.5-flash":
            return "gemini-2.5-flash-lite"
        elif model.startswith("gemini-3-"):
            return "gemini-2.5-flash"
        elif model.startswith("glm-"):
            return None  # Z.AI handles its own fallbacks
        return None

    def _execute_stream(
        self,
        model: str,
        req_data: Dict[str, Any],
        handler: Any,
        display_model: Optional[str] = None,
    ) -> None:
        """Inner execution logic for Gemini streaming."""
        auth_ctx = self.auth.get_auth_context()

        headers_dict = req_data.get("_headers") or {}
        subagent = headers_dict.get("x-openai-subagent")

        if subagent in ("compact", "review"):
            subagent_model = config.fallback_models.get(
                f"subagent:{subagent}",
                config.compaction_model or config.models[0]
                if config.models
                else "gemini-2.5-flash-lite",
            )
            if model != subagent_model:
                model = subagent_model
                logger.info(
                    f"Subagent '{subagent}' detected. Using faster model: {model}"
                )

        display_model = display_model or model

        messages = req_data.get("messages") or []
        contents, system_instruction = map_messages(messages, model)
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "..."}]}]

        gen_config = self._build_gen_config(req_data, model, system_instruction)

        # Determine URL and Request Body based on Auth Type
        if auth_ctx["type"] == "internal":
            url = (
                f"{config.gemini_api_internal}/v1internal:streamGenerateContent?alt=sse"
            )
            headers = {
                "Authorization": f"Bearer {auth_ctx['access_token']}",
                "User-Agent": f"GeminiCLI/0.26.0/{model} (linux; x64)",
            }
            request_body = {
                "model": model,
                "project": auth_ctx["project_id"],
                "user_prompt_id": f"u-{int(time.time())}",
                "request": {
                    "contents": contents,
                    "generationConfig": gen_config,
                    "session_id": str(
                        headers_dict.get("session_id")
                        or req_data.get("conversation_id")
                        or f"s-{int(time.time())}"
                    ),
                },
            }
            if system_instruction:
                request_body["request"]["systemInstruction"] = system_instruction

            # Apply Tools to the "request" sub-object
            self._apply_tools(req_data, request_body["request"])

            if req_data.get("store"):
                headers["x-codex-store"] = "true"
                ts = headers_dict.get("x-codex-turn-state")
                if ts:
                    headers["x-codex-turn-state"] = ts
        else:
            # Public API
            url = f"{config.gemini_api_public}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={auth_ctx['api_key']}"
            headers = {
                "User-Agent": f"GeminiCLI/0.26.0/{model} (linux; x64)",
            }
            request_body = {
                "contents": contents,
                "generationConfig": gen_config,
            }
            if system_instruction:
                request_body["systemInstruction"] = system_instruction

            # Apply Tools directly to the body
            self._apply_tools(req_data, request_body)

        with self.session.post(
            url,
            data=json_dumps(request_body),
            headers=headers,
            stream=True,
            timeout=(config.request_timeout_connect, config.request_timeout_read),
        ) as resp:
            if resp.status_code != 200:
                raise requests.exceptions.HTTPError(
                    f"Gemini API Error {resp.status_code}", response=resp
                )

            turn_state = resp.headers.get("x-codex-turn-state")
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
            handler.send_header("Connection", "keep-alive")
            if turn_state:
                handler.send_header("x-codex-turn-state", turn_state)
            handler.send_header("x-reasoning-included", "true")
            handler.end_headers()

            if req_data.get("_is_responses_api"):
                return stream_responses_loop(
                    resp, handler, display_model, int(time.time()), req_data
                )

            return self._stream_chat_loop(
                resp, handler, display_model, int(time.time())
            )

    def _build_gen_config(
        self, req_data: Dict[str, Any], model: str, system_instruction: Optional[Dict]
    ) -> Dict[str, Any]:
        """Build the generation configuration payload."""
        max_tokens = int(
            req_data.get("max_tokens") or req_data.get("maxOutputTokens") or 8192
        )
        gen_config = {
            "temperature": float(req_data.get("temperature", 1.0)),
            "topP": 0.95,
            "topK": 64,
            "maxOutputTokens": max_tokens,
        }

        effort = str((req_data.get("reasoning") or {}).get("effort", "medium"))
        self._apply_thinking_config(gen_config, effort, model, system_instruction)

        text_ctrl = req_data.get("text") or {}
        if text_ctrl.get("format", {}).get("type") == "json_schema":
            schema = text_ctrl["format"].get("schema")
            if schema:
                gen_config["responseMimeType"] = "application/json"
                gen_config["responseSchema"] = schema
                if system_instruction:
                    system_instruction["parts"][0]["text"] += (
                        "\nOutput response strictly as JSON."
                    )

        verbosity = str(text_ctrl.get("verbosity", "medium"))
        if verbosity == "low" and system_instruction:
            system_instruction["parts"][0]["text"] += "\nRespond briefly."
        elif verbosity == "high" and system_instruction:
            system_instruction["parts"][0]["text"] += "\nRespond comprehensively."

        return gen_config

    def _apply_thinking_config(
        self,
        gen_config: Dict,
        effort: str,
        model: str,
        system_instruction: Optional[Dict],
    ):
        """Map reasoning effort to Gemini thinkingConfig using config."""
        effort_levels = config.reasoning.get("effort_levels", {})
        default_effort = config.reasoning.get("default_effort", "medium")
        effort_config = effort_levels.get(
            effort,
            effort_levels.get(default_effort, {"budget": 16384, "level": "MEDIUM"}),
        )
        budget = effort_config.get("budget", 16384)
        level = effort_config.get("level", "MEDIUM")

        if model.startswith("gemini-3"):
            gen_config["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingLevel": level,
            }
        else:
            gen_config["thinkingConfig"] = {
                "thinkingBudget": budget,
                "includeThoughts": True,
            }

        if effort == "xhigh" and system_instruction:
            system_instruction["parts"][0]["text"] += "\nProvide deep reasoning."

    def _apply_tools(self, req_data: Dict[str, Any], target_dict: Dict[str, Any]):
        """Apply tool configuration to the target dictionary (either body or 'request' sub-object)."""
        tools = []
        for t in req_data.get("tools") or []:
            f = (
                t.get("function")
                if t.get("type") == "function"
                else (t if "name" in t else None)
            )
            if f:
                tools.append(
                    {
                        "name": f["name"],
                        "description": f.get("description", ""),
                        "parameters": sanitize_params(f.get("parameters", {})),
                    }
                )

        if tools:
            if "tools" not in target_dict:
                target_dict["tools"] = []
            target_dict["tools"].append({"functionDeclarations": tools})

            tc = req_data.get("tool_choice", "auto")
            mode = (
                "ANY"
                if tc in ("auto", "required")
                else ("NONE" if tc == "none" else "ANY")
            )
            target_dict["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

        if "search" in (req_data.get("include") or []):
            if "tools" not in target_dict:
                target_dict["tools"] = []
            target_dict["tools"].append({"googleSearch": {}})

    def _report_error(self, handler: Any, error: Exception, is_responses_api: bool):
        """Standardized error reporting."""
        logger.error(f"Request failed: {error}")
        err_msg = str(error)
        status_code = 500
        codex_code = "internal_server_error"

        if (
            isinstance(error, requests.exceptions.HTTPError)
            and error.response is not None
        ):
            status_code = error.response.status_code
            if status_code == 429:
                codex_code = "usage_limit_exceeded"
            elif status_code == 400:
                codex_code = "bad_request"

        if is_responses_api:
            try:
                fail_evt = {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {"code": codex_code, "message": err_msg},
                    },
                }
                handler.wfile.write(b"data: " + json_dumps(fail_evt) + b"\n\n")
                handler.wfile.flush()
            except Exception:
                pass
        else:
            handler.send_error(status_code, err_msg)

    def _stream_chat_loop(
        self, resp: requests.Response, handler: Any, model: str, created_ts: int
    ):
        """Standard OpenAI-compatible chat stream loop."""
        write, flush = handler.wfile.write, handler.wfile.flush
        sent_role = False

        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                if line == b"data: [DONE]":
                    break
                try:
                    data = json_loads(line[6:])
                    candidates = data.get("candidates") or data.get("response", {}).get(
                        "candidates"
                    )
                    if not candidates:
                        continue
                    cand = candidates[0]
                    parts = cand.get("content", {}).get("parts", [])
                    finish = cand.get("finishReason")

                    buf = ""
                    tcs = []
                    for p in parts:
                        if "text" in p:
                            buf += p["text"]
                        elif "functionCall" in p:
                            fc = p["functionCall"]
                            tcs.append(
                                {
                                    "index": 0,
                                    "id": f"call_{created_ts}",
                                    "type": "function",
                                    "function": {
                                        "name": fc["name"],
                                        "arguments": json.dumps(fc["args"]),
                                    },
                                }
                            )

                    chunk = {
                        "id": f"chatcmpl-{data.get('traceId', created_ts)}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": self._map_finish_reason(finish)
                                if not tcs
                                else "tool_calls",
                            }
                        ],
                    }

                    delta = chunk["choices"][0]["delta"]
                    if not sent_role:
                        delta["role"] = "assistant"
                        sent_role = True
                    if buf:
                        delta["content"] = buf
                    if tcs:
                        delta["tool_calls"] = tcs

                    write(b"data: " + json_dumps(chunk) + b"\n\n")
                    flush()
                except Exception:
                    continue

    def _map_finish_reason(self, reason: Optional[str]) -> Optional[str]:
        if not reason:
            return None
        return {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
        }.get(reason, "stop")
