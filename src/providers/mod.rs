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

use crate::account_pool::AccountProvider;
use base::Provider;
use gemini::GeminiProvider;
use openai::OpenAiProvider;
use zai::ZAIProvider;

struct RegistryInner {
    providers: HashMap<AccountProvider, Box<dyn Provider + Send + Sync>>,
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
        .insert(AccountProvider::Gemini, Box::new(GeminiProvider::new()));
    reg.providers
        .insert(AccountProvider::Zai, Box::new(ZAIProvider::new()));
    reg.providers
        .insert(AccountProvider::OpenAi, Box::new(OpenAiProvider::new()));
}

pub fn get_provider(provider: AccountProvider) -> Box<dyn Provider + Send + Sync> {
    REGISTRY
        .read()
        .providers
        .get(&provider)
        .map(|provider| provider.clone_box())
        .unwrap_or_else(|| match provider {
            AccountProvider::Gemini => Box::new(GeminiProvider::new()),
            AccountProvider::Zai => Box::new(ZAIProvider::new()),
            AccountProvider::OpenAi => Box::new(OpenAiProvider::new()),
        })
}
