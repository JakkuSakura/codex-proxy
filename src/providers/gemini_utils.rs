use crate::schema::openai::{ChatContent, ChatMessage};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

pub fn sanitize_params(params: &Value) -> Value {
    match params {
        Value::Object(map) => {
            let filtered = map
                .iter()
                .filter(|(k, _)| {
                    ![
                        "additionalProperties",
                        "title",
                        "default",
                        "minItems",
                        "maxItems",
                        "uniqueItems",
                    ]
                    .contains(&k.as_str())
                })
                .map(|(k, v)| (k.clone(), sanitize_params(v)))
                .collect();
            Value::Object(filtered)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(sanitize_params).collect()),
        other => other.clone(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeminiSystemInstruction {
    pub parts: Vec<GeminiPart>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,

    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    pub function_call: Option<GeminiFunctionCall>,

    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GeminiFunctionResponse>,

    #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: GeminiFunctionResponseBody,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeminiFunctionResponseBody {
    pub content: String,
}

pub fn map_messages(
    messages: &[ChatMessage],
    _common: &fp_agent::AgentRequest,
    _model_name: &str,
) -> (Vec<GeminiContent>, Option<GeminiSystemInstruction>) {
    let mut contents: Vec<GeminiContent> = Vec::new();
    let mut system_parts: Vec<GeminiPart> = Vec::new();

    let mut tool_call_map: HashMap<String, String> = HashMap::new();
    for m in messages {
        for tc in &m.tool_calls {
            tool_call_map.insert(tc.id.clone(), tc.function.name.clone());
        }
    }

    for m in messages {
        let role = m.role.as_str();
        if role == "system" || role == "developer" {
            let text = chat_content_to_string(m.content.as_ref());
            if !text.is_empty() {
                system_parts.push(GeminiPart {
                    text: Some(text),
                    thought: None,
                    function_call: None,
                    function_response: None,
                    thought_signature: None,
                });
            }
            continue;
        }

        let mut parts: Vec<GeminiPart> = Vec::new();
        if let Some(reasoning) = m.reasoning_content.as_deref() {
            if !reasoning.is_empty() {
                parts.push(GeminiPart {
                    text: Some(reasoning.to_string()),
                    thought: Some(true),
                    function_call: None,
                    function_response: None,
                    thought_signature: None,
                });
            }
        }

        let text = chat_content_to_string(m.content.as_ref());
        if !text.is_empty() {
            parts.push(GeminiPart {
                text: Some(text),
                thought: None,
                function_call: None,
                function_response: None,
                thought_signature: None,
            });
        }

        for tc in &m.tool_calls {
            let args: Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(Value::Object(Default::default()));
            let thought_sig = m
                .thought_signature
                .as_deref()
                .unwrap_or("skip_thought_signature_validator")
                .to_string();
            parts.push(GeminiPart {
                text: None,
                thought: None,
                function_call: Some(GeminiFunctionCall {
                    name: tc.function.name.clone(),
                    args,
                }),
                function_response: None,
                thought_signature: Some(thought_sig),
            });
        }

        let gemini_role = if role == "assistant" { "model" } else { "user" };

        if role == "tool" {
            let tc_id = m.tool_call_id.as_deref().unwrap_or("unknown");
            let fn_name = tool_call_map
                .get(tc_id)
                .map(|s| s.as_str())
                .or_else(|| m.name.as_deref())
                .unwrap_or("unknown");
            let content = chat_content_to_string(m.content.as_ref());
            let resp_part = GeminiPart {
                text: None,
                thought: None,
                function_call: None,
                function_response: Some(GeminiFunctionResponse {
                    name: fn_name.to_string(),
                    response: GeminiFunctionResponseBody { content },
                }),
                thought_signature: None,
            };

            // Gemini expects functionResponse parts under role user.
            let tool_role = "user".to_string();
            if let Some(last) = contents.last_mut() {
                if last.role == tool_role
                    && last.parts.iter().any(|p| p.function_response.is_some())
                {
                    last.parts.push(resp_part);
                    continue;
                }
            }

            contents.push(GeminiContent {
                role: tool_role,
                parts: vec![resp_part],
            });
            continue;
        }

        if parts.is_empty() {
            continue;
        }

        if let Some(last) = contents.last_mut() {
            if last.role == gemini_role {
                last.parts.extend(parts);
                continue;
            }
        }

        contents.push(GeminiContent {
            role: gemini_role.to_string(),
            parts,
        });
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiSystemInstruction {
            parts: system_parts,
        })
    };
    (contents, system_instruction)
}

fn chat_content_to_string(content: Option<&ChatContent>) -> String {
    match content {
        None => String::new(),
        Some(ChatContent::Text(s)) => s.clone(),
        Some(ChatContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| p.text.clone())
            .collect::<Vec<_>>()
            .join(""),
    }
}
