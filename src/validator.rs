use crate::error::ProxyError;
use crate::schema::openai::{CompactRequest, ResponsesInput, ResponsesRequest, Tool};

pub fn validate_responses_request(req: &ResponsesRequest) -> Result<(), ProxyError> {
    validate_model(&req.model)?;

    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(ProxyError::Validation(format!(
                "temperature must be between 0 and 2, got: {t}"
            )));
        }
    }

    if let Some(mt) = req.max_tokens {
        if !(1..=128_000).contains(&mt) {
            return Err(ProxyError::Validation(format!(
                "max_tokens must be between 1 and 128000, got: {mt}"
            )));
        }
    }

    if let Some(input) = &req.input {
        validate_input(input)?;
    }

    if let Some(tools) = &req.tools {
        validate_tools(tools)?;
    }

    Ok(())
}

pub fn validate_compact_request(req: &CompactRequest) -> Result<(), ProxyError> {
    validate_input(&req.input)?;
    Ok(())
}

fn validate_model(model: &str) -> Result<(), ProxyError> {
    if model.len() > 100 {
        return Err(ProxyError::Validation(format!(
            "Invalid model name (too long): {model}"
        )));
    }
    Ok(())
}

fn validate_input(input: &ResponsesInput) -> Result<(), ProxyError> {
    match input {
        ResponsesInput::Text(_) => Ok(()),
        ResponsesInput::Items(items) => {
            if items.len() > 100 {
                return Err(ProxyError::Validation(
                    "Compaction input exceeds maximum length of 100 messages".into(),
                ));
            }
            Ok(())
        }
    }
}

fn validate_tools(tools: &[Tool]) -> Result<(), ProxyError> {
    for (i, tool) in tools.iter().enumerate() {
        match tool.tool_type.as_str() {
            "function" | "web_search" | "retrieval" => {}
            other => {
                return Err(ProxyError::Validation(format!(
                    "Tool {i} has invalid type: {other}"
                )));
            }
        }
        if tool.tool_type == "function" && tool.function.is_none() && tool.name.is_none() {
            return Err(ProxyError::Validation(format!(
                "Tool {i} of type function must have 'function' or 'name'"
            )));
        }
    }
    Ok(())
}
