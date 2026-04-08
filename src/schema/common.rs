use fp_agent::{
    AgentMessage as CommonMessage, AgentRequest as CommonRequest,
    ToolCall as CommonToolCall, ToolCallFunction as CommonToolCallFunction,
    ToolDefinition as CommonToolDefinition, ToolFunctionDefinition as CommonToolFunctionDefinition,
};
use serde_json::{Value, json};

use crate::schema::openai::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, FunctionDef, Tool, ToolCall,
};

pub fn from_chat_request(req: &ChatRequest) -> CommonRequest {
    CommonRequest {
        model: req.model.clone(),
        messages: req.messages.iter().map(from_chat_message).collect(),
        tools: req.tools.iter().map(from_tool_definition).collect(),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| "auto".to_string()),
        max_tokens: req.max_tokens.map(|v| v as u32),
        json_mode: false,
    }
}

pub fn from_chat_message(msg: &ChatMessage) -> CommonMessage {
    CommonMessage {
        role: msg.role.clone(),
        content: Some(chat_content_to_string(msg.content.as_ref())),
        tool_calls: msg.tool_calls.iter().map(from_tool_call).collect(),
        tool_call_id: msg.tool_call_id.clone(),
        name: msg.name.clone(),
    }
}

pub fn from_tool_call(call: &ToolCall) -> CommonToolCall {
    CommonToolCall {
        id: call.id.clone(),
        function: CommonToolCallFunction {
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        },
    }
}

pub fn to_openai_messages(messages: &[CommonMessage]) -> Vec<Value> {
    messages.iter().map(to_openai_message).collect()
}

pub fn to_openai_message(message: &CommonMessage) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("role".to_string(), Value::String(message.role.clone()));
    if let Some(content) = &message.content {
        object.insert("content".to_string(), Value::String(content.clone()));
    }
    if !message.tool_calls.is_empty() {
        object.insert(
            "tool_calls".to_string(),
            Value::Array(
                message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        object.insert("tool_call_id".to_string(), Value::String(tool_call_id.clone()));
    }
    if let Some(name) = &message.name {
        object.insert("name".to_string(), Value::String(name.clone()));
    }
    Value::Object(object)
}

pub fn to_openai_tools(tools: &[CommonToolDefinition]) -> Vec<Value> {
    tools.iter()
        .map(|tool| {
            json!({
                "type": tool.r#type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                }
            })
        })
        .collect()
}

pub fn to_chat_messages(messages: &[CommonMessage]) -> Vec<ChatMessage> {
    messages.iter().map(to_chat_message).collect()
}

pub fn to_chat_message(message: &CommonMessage) -> ChatMessage {
    ChatMessage {
        role: message.role.clone(),
        content: Some(if let Some(content) = &message.content {
            ChatContent::Text(content.clone())
        } else {
            ChatContent::Parts(Vec::<ChatContentPart>::new())
        }),
        reasoning_content: None,
        thought_signature: None,
        tool_calls: message.tool_calls.iter().map(to_tool_call).collect(),
        tool_call_id: message.tool_call_id.clone(),
        name: message.name.clone(),
    }
}

pub fn to_tool_call(call: &CommonToolCall) -> ToolCall {
    ToolCall {
        id: call.id.clone(),
        call_type: "function".to_string(),
        function: crate::schema::openai::ToolCallFunction {
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        },
    }
}

fn from_tool_definition(tool: &Tool) -> CommonToolDefinition {
    let function = tool.function.clone().unwrap_or(FunctionDef {
        name: tool.name.clone().unwrap_or_else(|| "unknown".to_string()),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
    });

    CommonToolDefinition {
        r#type: tool.tool_type.clone(),
        function: CommonToolFunctionDefinition {
            name: function.name,
            description: function.description.unwrap_or_default(),
            parameters: function.parameters.unwrap_or(Value::Object(Default::default())),
        },
    }
}

fn chat_content_to_string(content: Option<&ChatContent>) -> String {
    match content {
        None => String::new(),
        Some(ChatContent::Text(text)) => text.clone(),
        Some(ChatContent::Parts(parts)) => parts
            .iter()
            .filter_map(|part| part.text.clone())
            .collect::<Vec<_>>()
            .join(""),
    }
}
