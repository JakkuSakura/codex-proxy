pub mod base;
pub mod gemini;
pub mod gemini_stream;
pub mod gemini_utils;
pub mod openai;
pub mod openrouter;
pub mod zai;
pub mod zai_stream;

use std::collections::HashMap;

use crate::config::{ProviderType, with_config};
use crate::state::AppState;
use base::Provider;
use gemini::GeminiProvider;
use openai::OpenAiProvider;
use openrouter::OpenRouterProvider;
use zai::ZAIProvider;

pub struct ProviderRegistry {
    providers: HashMap<ProviderType, Box<dyn Provider + Send + Sync>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        let mut providers: HashMap<ProviderType, Box<dyn Provider + Send + Sync>> = HashMap::new();
        providers.insert(ProviderType::Gemini, Box::new(GeminiProvider::new()));
        providers.insert(ProviderType::Zai, Box::new(ZAIProvider::new()));
        providers.insert(ProviderType::OpenAi, Box::new(OpenAiProvider::new()));
        providers.insert(
            ProviderType::OpenRouter,
            Box::new(OpenRouterProvider::new()),
        );
        Self { providers }
    }

    pub fn get(&self, provider_type: ProviderType) -> Box<dyn Provider + Send + Sync> {
        if let Some(provider) = self.providers.get(&provider_type) {
            return provider.clone_box();
        }
        match provider_type {
            ProviderType::Gemini => Box::new(GeminiProvider::new()),
            ProviderType::Zai => Box::new(ZAIProvider::new()),
            ProviderType::OpenAi => Box::new(OpenAiProvider::new()),
            ProviderType::OpenRouter => Box::new(OpenRouterProvider::new()),
        }
    }
}

pub fn get_provider(state: &AppState, provider_name: &str) -> Box<dyn Provider + Send + Sync> {
    let provider_type = with_config(state.config(), |cfg| {
        cfg.provider_type(provider_name)
            .expect("provider referenced by route/account must exist in config")
    });
    state.providers().get(provider_type)
}
