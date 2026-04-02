use bytes::Bytes;
use futures::StreamExt;
use futures::stream::Stream;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

use crate::schema::json_value::JsonValue;
use crate::schema::openai::ChatRequest;
use crate::schema::sse::{
    FunctionCallItem, LocalShellCallItem, MessageItem, OutputContentPart, OutputItem,
    ResponseCompletedData, ResponseCreatedData, ResponseEvent, ResponseObject,
    ResponseOutputItemAddedData, ResponseOutputItemDoneData, ResponseOutputTextDeltaData, Usage,
};

pub fn stream_responses_sse(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: &str,
    created_ts: i64,
    request: &ChatRequest,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let model = model.to_string();
    let resp_id = format!("resp_{created_ts}");
    let req = request.clone();

    Box::pin(async_stream::stream! {
        let mut seq_num: u64 = 0;
        let mut full_content = String::new();
        let mut message: Option<MessageState> = None;
        let mut next_idx: usize = 0;
        let mut tool_calls: HashMap<usize, ToolCallState> = HashMap::new();
        let mut final_usage: Option<Usage> = None;

        yield Ok(encode_event(
            &mut seq_num,
            "response.created",
            ResponseCreatedData {
                response: response_in_progress(&resp_id, &model, created_ts, &req),
            },
        ));

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut line_buf = String::new();
        let mut saw_done = false;
        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    debug!("ZAI stream error: {}", e);
                    break;
                }
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
                let data: ZaiStreamChunk = match serde_json::from_str(data_str) {
                    Ok(d) => d,
                    Err(e) => {
                        warn!("ZAI chunk parse failed: {e}, raw: {data_str}");
                        continue;
                    }
                };
                debug!("ZAI STREAM DELTA: {}", data_str);

                if let Some(usage) = data.usage {
                    let it = usage.prompt_tokens.unwrap_or(0);
                    let ot = usage.completion_tokens.unwrap_or(0);
                    final_usage = Some(Usage {
                        input_tokens: it,
                        output_tokens: ot,
                        total_tokens: usage.total_tokens.unwrap_or(it + ot),
                        input_tokens_details: None,
                        output_tokens_details: None,
                    });
                }

                let choice = match data.choices.first() {
                    Some(c) => c,
                    None => continue,
                };
                let delta = choice.delta.clone().unwrap_or_default();

                // Tool calls
                for tc_delta in delta.tool_calls {
                    let idx = tc_delta.index.unwrap_or(0) as usize;

                    let is_new = !tool_calls.contains_key(&idx);
                    let entry = tool_calls.entry(idx).or_insert_with(|| {
                        let output_idx = next_idx;
                        next_idx += 1;
                        let call_id = tc_delta
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}_{}", now_ms(), output_idx));

                        let item = OutputItem::FunctionCall(FunctionCallItem {
                            id: call_id.clone(),
                            status: "in_progress".into(),
                            name: String::new(),
                            arguments: String::new(),
                            call_id: call_id.clone(),
                            thought_signature: None,
                        });

                        ToolCallState {
                            output_index: output_idx,
                            call_id,
                            name: String::new(),
                            arguments: String::new(),
                            item,
                        }
                    });

                    if is_new {
                        yield Ok(encode_event(
                            &mut seq_num,
                            "response.output_item.added",
                            ResponseOutputItemAddedData {
                                response_id: resp_id.clone(),
                                output_index: entry.output_index,
                                item: entry.item.clone(),
                            },
                        ));
                    }

                    if let Some(fn_delta) = tc_delta.function {
                        if let Some(name_part) = fn_delta.name {
                            entry.name.push_str(&name_part);
                        }
                        if let Some(args_part) = fn_delta.arguments {
                            entry.arguments.push_str(&args_part.to_json_fragment());
                        }
                    }
                }

                // Content
                if let Some(content) = delta.content {
                    if content.is_empty() {
                        continue;
                    }
                    full_content.push_str(&content);

                    if message.is_none() {
                        let output_idx = next_idx as i64;
                        next_idx += 1;
                        let item_id = format!("msg_{}_{}", now_ms(), output_idx);
                        let item = OutputItem::Message(MessageItem {
                            id: item_id.clone(),
                            role: "assistant",
                            status: "in_progress".into(),
                            content: vec![OutputContentPart::OutputText { text: String::new() }],
                        });

                        yield Ok(encode_event(
                            &mut seq_num,
                            "response.output_item.added",
                            ResponseOutputItemAddedData {
                                response_id: resp_id.clone(),
                                output_index: output_idx as usize,
                                item: item.clone(),
                            },
                        ));

                        message = Some(MessageState {
                            output_index: output_idx,
                            item_id,
                            item,
                        });
                    }

                    if let Some(ref mut state) = message {
                        state.set_text(&full_content);
                        yield Ok(encode_event(
                            &mut seq_num,
                            "response.output_text.delta",
                            ResponseOutputTextDeltaData {
                                response_id: resp_id.clone(),
                                item_id: state.item_id.clone(),
                                output_index: state.output_index,
                                content_index: 0,
                                delta: content,
                            },
                        ));
                    }
                }
            }

            if saw_done {
                break;
            }
        }

        // Finalize output items in output_index order.
        let mut items_to_close: Vec<(usize, OutputItem)> = Vec::new();
        if let Some(msg) = message {
            items_to_close.push((msg.output_index as usize, msg.item));
        }
        for tc in tool_calls.values() {
            items_to_close.push((tc.output_index, finalize_tool_call(tc)));
        }
        items_to_close.sort_by_key(|(idx, _)| *idx);

        let mut final_output: Vec<OutputItem> = Vec::new();
        for (out_idx, mut item) in items_to_close {
            set_item_status(&mut item, "completed");
            yield Ok(encode_event(
                &mut seq_num,
                "response.output_item.done",
                ResponseOutputItemDoneData {
                    response_id: resp_id.clone(),
                    output_index: out_idx,
                    item: item.clone(),
                },
            ));
            final_output.push(item);
        }

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

#[derive(Clone, Debug)]
struct MessageState {
    output_index: i64,
    item_id: String,
    item: OutputItem,
}

impl MessageState {
    fn set_text(&mut self, full_text: &str) {
        match &mut self.item {
            OutputItem::Message(m) => {
                if let Some(OutputContentPart::OutputText { text }) = m.content.first_mut() {
                    *text = full_text.to_string();
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug)]
struct ToolCallState {
    output_index: usize,
    call_id: String,
    name: String,
    arguments: String,
    item: OutputItem,
}

fn finalize_tool_call(tc: &ToolCallState) -> OutputItem {
    let name = tc.name.clone();
    let args = tc.arguments.clone();
    let call_id = tc.call_id.clone();
    if name == "shell" || name == "container.exec" || name == "shell_command" {
        let cmd = extract_command_from_args(&args);
        OutputItem::LocalShellCall(LocalShellCallItem {
            id: call_id.clone(),
            status: "in_progress".into(),
            name,
            arguments: args,
            call_id: call_id.clone(),
            action: crate::schema::sse::ShellAction {
                action_type: "exec",
                command: cmd,
            },
            thought_signature: None,
        })
    } else {
        OutputItem::FunctionCall(FunctionCallItem {
            id: call_id.clone(),
            status: "in_progress".into(),
            name,
            arguments: args,
            call_id: call_id.clone(),
            thought_signature: None,
        })
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
    let payload = format!(
        "event: {}\ndata: {}\n\n",
        evt_type,
        serde_json::to_string(&evt).unwrap_or_default()
    );
    Bytes::from(payload)
}

/// Deserialize a Vec field, treating null as empty vec.
fn deserialize_null_default_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: serde::de::DeserializeOwned,
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    struct NullDefaultVec<T>(std::marker::PhantomData<T>);
    impl<'de, T: serde::de::DeserializeOwned> Visitor<'de> for NullDefaultVec<T> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("Vec or null")
        }
        fn visit_none<E: de::Error>(self) -> Result<Vec<T>, E> {
            Ok(Vec::new())
        }
        fn visit_unit<E: de::Error>(self) -> Result<Vec<T>, E> {
            Ok(Vec::new())
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, seq: A) -> Result<Vec<T>, A::Error> {
            serde::Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }
    deserializer.deserialize_any(NullDefaultVec::<T>(std::marker::PhantomData))
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiStreamChunk {
    #[serde(default)]
    usage: Option<ZaiUsage>,
    #[serde(default, deserialize_with = "deserialize_null_default_vec")]
    choices: Vec<ZaiChoice>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiChoice {
    #[serde(default)]
    delta: Option<ZaiDelta>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ZaiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "deserialize_null_default_vec")]
    tool_calls: Vec<ZaiToolCallDelta>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiToolCallDelta {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ZaiToolCallFunctionDelta>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<ZaiArgumentsDelta>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum ZaiArgumentsDelta {
    Text(String),
    Json(JsonValue),
}

impl ZaiArgumentsDelta {
    fn to_json_fragment(&self) -> String {
        match self {
            ZaiArgumentsDelta::Text(s) => s.clone(),
            ZaiArgumentsDelta::Json(v) => serde_json::to_string(v).unwrap_or_default(),
        }
    }
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
