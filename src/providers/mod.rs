pub mod base;
pub mod gemini;
pub mod gemini_stream;
pub mod gemini_utils;
pub mod openai;
pub mod zai;
pub mod zai_stream;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;

use crate::config::{ProviderType, CONFIG};
use base::Provider;
use gemini::GeminiProvider;
use openai::OpenAiProvider;
use zai::ZAIProvider;

struct RegistryInner {
    providers: HashMap<ProviderType, Box<dyn Provider + Send + Sync>>,
}

static REGISTRY: Lazy<RwLock<RegistryInner>> = Lazy::new(|| {
    RwLock::new(RegistryInner {
        providers: HashMap::new(),
    })
});

pub fn initialize_registry() {
    let mut reg = REGISTRY.write();
    reg.providers.clear();
    reg.providers
        .insert(ProviderType::Gemini, Box::new(GeminiProvider::new()));
    reg.providers
        .insert(ProviderType::Zai, Box::new(ZAIProvider::new()));
    reg.providers
        .insert(ProviderType::OpenAi, Box::new(OpenAiProvider::new()));
}

pub fn get_provider(provider_name: &str) -> Box<dyn Provider + Send + Sync> {
    let provider_type = CONFIG
        .provider_type(provider_name)
        .expect("provider referenced by route/account must exist in config");

    if let Some(provider) = REGISTRY.read().providers.get(&provider_type) {
        return provider.clone_box();
    }

    match provider_type {
        ProviderType::Gemini => Box::new(GeminiProvider::new()),
        ProviderType::Zai => Box::new(ZAIProvider::new()),
        ProviderType::OpenAi => Box::new(OpenAiProvider::new()),
    }
}
