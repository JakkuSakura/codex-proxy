use bytes::Bytes;
use futures::StreamExt;
use futures::stream::Stream;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

use crate::schema::json_value::JsonValue;
use crate::schema::openai::ChatRequest;
use crate::schema::sse::{
    CreditsData, FunctionCallItem, InputTokensDetails, MessageItem, OutputContentPart, OutputItem,
    OutputTokensDetails, RateLimitsData, ReasoningContentPart, ReasoningItem,
    ResponseCompletedData, ResponseCreatedData, ResponseEvent, ResponseObject,
    ResponseOutputItemAddedData, ResponseOutputItemDoneData, ResponseOutputTextDeltaData,
    ServerReasoningIncludedData, SummaryPart, Usage,
};

pub fn stream_responses_sse(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    resp_id: &str,
    model: &str,
    created_ts: i64,
    request: &ChatRequest,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let resp_id = resp_id.to_string();
    let model = model.to_string();
    let req = request.clone();
    let re_header = Regex::new(r"\*\*(.*?)\*\*").unwrap();

    Box::pin(async_stream::stream! {
        let mut seq_num: u64 = 0;
        let mut global_next_idx: usize = 0;

        let mut active_message: Option<(usize, OutputItem)> = None;
        let mut active_reasoning: Option<(usize, OutputItem)> = None;
        let mut completed_items: Vec<(usize, OutputItem)> = Vec::new();
        let mut final_usage: Option<Usage> = None;

        yield Ok(encode_event(
            &mut seq_num,
            "response.created",
            ResponseCreatedData {
                response: response_in_progress(&resp_id, &model, created_ts, &req),
            },
        ));

        yield Ok(encode_event(
            &mut seq_num,
            "models_etag",
            crate::schema::sse::ModelsEtagData {
                etag: "v1-gemini-gpt-5-2-parity",
            },
        ));
        yield Ok(encode_event(
            &mut seq_num,
            "server_reasoning_included",
            ServerReasoningIncludedData { included: true },
        ));
        yield Ok(encode_event(
            &mut seq_num,
            "rate_limits",
            RateLimitsData {
                primary: Some(JsonValue::Null),
                secondary: Some(JsonValue::Null),
                credits: CreditsData {
                    has_credits: true,
                    unlimited: false,
                    balance: Some(JsonValue::Null),
                },
            },
        ));

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut line_buf = String::new();
        let mut saw_done = false;
        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => { debug!("Stream error: {}", e); break; }
            };
            let text = String::from_utf8_lossy(&chunk);
            line_buf.push_str(&text);

            // Some providers may omit the trailing newline for the final DONE sentinel.
            if line_buf.trim_end_matches('\r') == "data: [DONE]" {
                break;
            }

            while let Some(nl) = line_buf.find('\n') {
                let line = line_buf[..nl].trim_end_matches('\r').to_string();
                line_buf = line_buf[nl + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }
                if line == "data: [DONE]" {
                    saw_done = true;
                    break;
                }

                let data_str = &line[6..];
                let chunk: GeminiChunk = match serde_json::from_str(data_str) {
                    Ok(d) => d,
                    Err(e) => {
                        warn!("Gemini chunk parse failed: {e}, raw: {data_str}");
                        continue;
                    }
                };
                let resp_part = chunk.response();

                if let Some(usage) = &resp_part.usage_metadata {
                    let it = usage.prompt_token_count.unwrap_or(0);
                    let ot = usage.candidates_token_count.unwrap_or(0);
                    let rt = usage.thinking_token_count.unwrap_or(0);
                    let cached = usage.cached_content_token_count.unwrap_or(0);
                    final_usage = Some(Usage {
                        input_tokens: it,
                        input_tokens_details: Some(InputTokensDetails { cached_tokens: cached }),
                        output_tokens: ot,
                        output_tokens_details: Some(OutputTokensDetails { reasoning_tokens: rt }),
                        total_tokens: it + ot + rt,
                    });
                }

                let cand = match resp_part.candidates.first() {
                    Some(c) => c,
                    None => continue,
                };
                let parts = match cand.content.as_ref() {
                    Some(c) => c.parts.as_slice(),
                    None => continue,
                };

                let mut text_buf = String::new();
                let mut reasoning_text = String::new();

                for p in parts {
                    if let Some(fc) = &p.function_call {
                        let fc_name = fc.name.as_str();
                        let fc_args_str = serde_json::to_string(&fc.args).unwrap_or_else(|_| "{}".into());
                        let call_id = format!("call_{}_{}", now_ms(), global_next_idx);

                        let item = if fc_name == "shell" || fc_name == "container.exec" || fc_name == "shell_command" {
                            let cmd = extract_command_from_args(&fc_args_str);
                            OutputItem::LocalShellCall(crate::schema::sse::LocalShellCallItem {
                                id: call_id.clone(),
                                status: "in_progress".into(),
                                name: fc_name.to_string(),
                                arguments: fc_args_str,
                                call_id: call_id.clone(),
                                action: crate::schema::sse::ShellAction { action_type: "exec", command: cmd },
                                thought_signature: p.thought_signature.clone(),
                            })
                        } else {
                            OutputItem::FunctionCall(FunctionCallItem {
                                id: call_id.clone(),
                                status: "in_progress".into(),
                                name: fc_name.to_string(),
                                arguments: fc_args_str,
                                call_id: call_id.clone(),
                                thought_signature: p.thought_signature.clone(),
                            })
                        };

                        let idx = global_next_idx;
                        global_next_idx += 1;
                        yield Ok(encode_event(
                            &mut seq_num,
                            "response.output_item.added",
                            ResponseOutputItemAddedData {
                                response_id: resp_id.clone(),
                                output_index: idx,
                                item: item.clone(),
                            },
                        ));

                        let mut done_item = item;
                        set_item_status(&mut done_item, "completed");
                        yield Ok(encode_event(
                            &mut seq_num,
                            "response.output_item.done",
                            ResponseOutputItemDoneData {
                                response_id: resp_id.clone(),
                                output_index: idx,
                                item: done_item.clone(),
                            },
                        ));
                        completed_items.push((idx, done_item));
                        continue;
                    }

                    let is_reasoning = match &p.thought {
                        Some(GeminiThought::Text(_)) => true,
                        Some(GeminiThought::Bool(true)) => true,
                        _ => false,
                    };

                    let text_chunk = match &p.thought {
                        Some(GeminiThought::Text(s)) => s.as_str(),
                        _ => p.text.as_deref().unwrap_or(""),
                    };

                    if is_reasoning && !text_chunk.is_empty() {
                        reasoning_text.push_str(text_chunk);
                    } else if !matches!(p.thought, Some(GeminiThought::Bool(true))) {
                        if let Some(tx) = p.text.as_deref() {
                            if !tx.is_empty() {
                                text_buf.push_str(tx);
                            }
                        }
                    }
                }

                if !reasoning_text.is_empty() {
                    if active_reasoning.is_none() {
                        let idx = global_next_idx;
                        global_next_idx += 1;
                        let item = OutputItem::Reasoning(ReasoningItem {
                            id: format!("rs_{}_{}", now_ms(), idx),
                            status: "in_progress".into(),
                            summary: Vec::new(),
                            content: vec![ReasoningContentPart::ReasoningText { text: String::new() }],
                        });
                        active_reasoning = Some((idx, item));
                    }
                    if let Some((_idx, ref mut item)) = active_reasoning {
                        if let OutputItem::Reasoning(r) = item {
                            let current = match r.content.first() {
                                Some(ReasoningContentPart::ReasoningText { text }) => text.clone(),
                                _ => String::new(),
                            };
                            r.content = vec![ReasoningContentPart::ReasoningText { text: format!("{current}{reasoning_text}") }];
                        }
                    }
                }

                if !text_buf.is_empty() {
                    if active_message.is_none() {
                        let idx = global_next_idx;
                        global_next_idx += 1;
                        let item = OutputItem::Message(MessageItem {
                            id: format!("msg_{}_{}", now_ms(), idx),
                            role: "assistant",
                            status: "in_progress".into(),
                            content: vec![OutputContentPart::OutputText { text: String::new() }],
                        });
                        active_message = Some((idx, item));
                    }

                    if let Some((idx, ref mut item)) = active_message {
                        if let OutputItem::Message(m) = item {
                            let current = match m.content.first() {
                                Some(OutputContentPart::OutputText { text }) => text.clone(),
                                _ => String::new(),
                            };
                            let merged = format!("{current}{text_buf}");
                            m.content = vec![OutputContentPart::OutputText { text: merged.clone() }];
                            yield Ok(encode_event(
                                &mut seq_num,
                                "response.output_text.delta",
                                ResponseOutputTextDeltaData {
                                    response_id: resp_id.clone(),
                                    item_id: m.id.clone(),
                                    output_index: idx as i64,
                                    content_index: 0,
                                    delta: text_buf.clone(),
                                },
                            ));

                            // Update reasoning summary from full reasoning text.
                            if let Some((_ridx, ref mut r_item)) = active_reasoning {
                                if let OutputItem::Reasoning(r) = r_item {
                                    let full = match r.content.first() {
                                        Some(ReasoningContentPart::ReasoningText { text }) => text.as_str(),
                                        _ => "",
                                    };
                                    let headers: Vec<String> = re_header
                                        .captures_iter(full)
                                        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
                                        .collect();
                                    if !headers.is_empty() {
                                        r.summary = headers
                                            .iter()
                                            .map(|h| SummaryPart::SummaryText { text: h.clone() })
                                            .collect();
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(citations) = cand.citation_metadata.as_ref().map(|c| c.citations.as_slice()) {
                    for c in citations {
                        if let (Some(title), Some(uri)) = (c.title.as_deref(), c.uri.as_deref()) {
                            if let Some((_idx, ref mut item)) = active_message {
                                if let OutputItem::Message(m) = item {
                                    let current = match m.content.first() {
                                        Some(OutputContentPart::OutputText { text }) => text.clone(),
                                        _ => String::new(),
                                    };
                                    m.content = vec![OutputContentPart::OutputText { text: format!("{current}\n({title}) {uri}") }];
                                }
                            }
                        }
                    }
                }
            }

            if saw_done {
                break;
            }
        }

        if let Some((idx, mut item)) = active_reasoning.take() {
            set_item_status(&mut item, "completed");
            yield Ok(encode_event(
                &mut seq_num,
                "response.output_item.done",
                ResponseOutputItemDoneData {
                    response_id: resp_id.clone(),
                    output_index: idx,
                    item: item.clone(),
                },
            ));
            completed_items.push((idx, item));
        }
        if let Some((idx, mut item)) = active_message.take() {
            set_item_status(&mut item, "completed");
            yield Ok(encode_event(
                &mut seq_num,
                "response.output_item.done",
                ResponseOutputItemDoneData {
                    response_id: resp_id.clone(),
                    output_index: idx,
                    item: item.clone(),
                },
            ));
            completed_items.push((idx, item));
        }

        completed_items.sort_by_key(|(idx, _)| *idx);
        let final_output: Vec<OutputItem> = completed_items.into_iter().map(|(_, item)| item).collect();

        let mut final_resp = response_in_progress(&resp_id, &model, created_ts, &req);
        final_resp.status = "completed".into();
        final_resp.completed_at = Some(now_secs());
        final_resp.usage = Some(final_usage.unwrap_or_else(|| Usage {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            input_tokens_details: None,
            output_tokens_details: None,
        }));
        final_resp.output = final_output;

        yield Ok(encode_event(
            &mut seq_num,
            "response.completed",
            ResponseCompletedData { response: final_resp },
        ));
    })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum GeminiChunk {
    Wrapped { response: GeminiResponse },
    Direct(GeminiResponse),
}

impl GeminiChunk {
    fn response(&self) -> &GeminiResponse {
        match self {
            GeminiChunk::Wrapped { response } => response,
            GeminiChunk::Direct(r) => r,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiResponse {
    #[serde(rename = "usageMetadata", default)]
    usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: Option<u64>,
    #[serde(rename = "thinkingTokenCount", default)]
    thinking_token_count: Option<u64>,
    #[serde(rename = "cachedContentTokenCount", default)]
    cached_content_token_count: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiCandidateContent>,
    #[serde(rename = "citationMetadata", default)]
    citation_metadata: Option<CitationMetadata>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiCandidateContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiPart {
    #[serde(rename = "functionCall", default)]
    function_call: Option<GeminiFunctionCall>,
    #[serde(rename = "thoughtSignature", default)]
    thought_signature: Option<String>,
    #[serde(default)]
    thought: Option<GeminiThought>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default)]
    args: JsonValue,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum GeminiThought {
    Bool(bool),
    Text(String),
}

#[derive(Clone, Debug, Deserialize)]
struct CitationMetadata {
    #[serde(default)]
    citations: Vec<Citation>,
}

#[derive(Clone, Debug, Deserialize)]
struct Citation {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    uri: Option<String>,
}

fn response_in_progress(
    resp_id: &str,
    model: &str,
    created_ts: i64,
    req: &ChatRequest,
) -> ResponseObject {
    ResponseObject {
        id: resp_id.to_string(),
        object: "response",
        created_at: created_ts,
        completed_at: None,
        model: model.to_string(),
        status: "in_progress".into(),
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| "auto".into()),
        tools: req.tools.clone(),
        parallel_tool_calls: true,
        store: req.store,
        metadata: req.metadata.clone(),
        output: Vec::new(),
        usage: None,
    }
}

fn set_item_status(item: &mut OutputItem, status: &str) {
    match item {
        OutputItem::Message(m) => m.status = status.to_string(),
        OutputItem::Reasoning(r) => r.status = status.to_string(),
        OutputItem::FunctionCall(c) => c.status = status.to_string(),
        OutputItem::LocalShellCall(c) => c.status = status.to_string(),
    }
}

fn extract_command_from_args(args: &str) -> Vec<String> {
    let parsed: Result<JsonValue, _> = serde_json::from_str(args);
    match parsed {
        Ok(JsonValue::Object(map)) => match map.get("command") {
            Some(JsonValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn encode_event<T: Serialize>(seq_num: &mut u64, evt_type: &'static str, data: T) -> Bytes {
    *seq_num += 1;
    let evt = ResponseEvent {
        id: format!("evt_{}_{}", now_ms(), seq_num),
        object: "response.event",
        event_type: evt_type,
        created_at: now_secs(),
        sequence_number: *seq_num,
        data,
    };
    Bytes::from(format!(
        "event: {}\ndata: {}\n\n",
        evt_type,
        serde_json::to_string(&evt).unwrap_or_default()
    ))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
