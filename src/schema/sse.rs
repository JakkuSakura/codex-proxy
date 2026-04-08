use crate::schema::openai::Tool;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct ResponseEvent<T: Serialize> {
    pub id: String,
    pub object: &'static str,
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub created_at: i64,
    pub sequence_number: u64,
    #[serde(flatten)]
    pub data: T,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseCreatedData {
    pub response: ResponseObject,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseCompletedData {
    pub response: ResponseObject,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseFailedData {
    pub response: FailedResponseObject,
}

#[derive(Clone, Debug, Serialize)]
pub struct FailedResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub status: &'static str,
    pub model: String,
    pub error: ResponseError,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseOutputItemAddedData {
    pub response_id: String,
    pub output_index: usize,
    pub item: OutputItem,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseOutputItemDoneData {
    pub response_id: String,
    pub output_index: usize,
    pub item: OutputItem,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseOutputTextDeltaData {
    pub response_id: String,
    pub item_id: String,
    pub output_index: i64,
    pub content_index: usize,
    pub delta: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelsEtagData {
    pub etag: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerReasoningIncludedData {
    pub included: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RateLimitsData {
    pub primary: Option<Value>,
    pub secondary: Option<Value>,
    pub credits: CreditsData,
}

#[derive(Clone, Debug, Serialize)]
pub struct CreditsData {
    pub has_credits: bool,
    pub unlimited: bool,
    pub balance: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    pub model: String,
    pub status: String,

    pub temperature: f64,
    pub top_p: f64,
    pub tool_choice: String,
    pub tools: Vec<Tool>,
    pub parallel_tool_calls: bool,
    pub store: bool,
    pub metadata: BTreeMap<String, Value>,
    pub output: Vec<OutputItem>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message(MessageItem),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningItem),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionCallItem),
    #[serde(rename = "local_shell_call")]
    LocalShellCall(LocalShellCallItem),
}

#[derive(Clone, Debug, Serialize)]
pub struct MessageItem {
    pub id: String,
    pub role: &'static str,
    pub status: String,
    pub content: Vec<OutputContentPart>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReasoningItem {
    pub id: String,
    pub status: String,
    pub summary: Vec<SummaryPart>,
    pub content: Vec<ReasoningContentPart>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FunctionCallItem {
    pub id: String,
    pub status: String,
    pub name: String,
    pub arguments: String,
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalShellCallItem {
    pub id: String,
    pub status: String,
    pub name: String,
    pub arguments: String,
    pub call_id: String,
    pub action: ShellAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShellAction {
    #[serde(rename = "type")]
    pub action_type: &'static str,
    pub command: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum OutputContentPart {
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ReasoningContentPart {
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum SummaryPart {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_failed_event_serializes_error_shape() {
        let evt = ResponseEvent {
            id: "evt_test".to_string(),
            object: "response.event",
            event_type: "response.failed",
            created_at: 123,
            sequence_number: 41,
            data: ResponseFailedData {
                response: FailedResponseObject {
                    id: "resp_test".to_string(),
                    object: "response",
                    created_at: 123,
                    status: "failed",
                    model: "glm-5-turbo".to_string(),
                    error: ResponseError {
                        code: "stream_timeout".to_string(),
                        message: "Upstream stream idle".to_string(),
                    },
                    metadata: BTreeMap::new(),
                },
            },
        };

        // Assert only load-bearing keys to keep this test resilient.
        let val = serde_json::to_value(evt).expect("event must serialize");
        assert_eq!(val["type"], "response.failed");
        assert_eq!(val["response"]["status"], "failed");
        assert_eq!(val["response"]["error"]["code"], "stream_timeout");
        assert_eq!(val["response"]["error"]["message"], "Upstream stream idle");
        assert_eq!(val["response"]["model"], "glm-5-turbo");
    }
}
