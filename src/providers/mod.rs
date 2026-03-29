pub mod base;
pub mod gemini;
pub mod gemini_utils;
pub mod gemini_stream;
pub mod zai;
pub mod zai_stream;

use parking_lot::RwLock;
use once_cell::sync::Lazy;

use crate::config::CONFIG;
use base::Provider;
use gemini::GeminiProvider;
use zai::ZAIProvider;

struct ProviderEntry {
    prefix: String,
    provider: Box<dyn Provider + Send + Sync>,
}

struct RegistryInner {
    providers: Vec<ProviderEntry>,
}

static REGISTRY: Lazy<RwLock<RegistryInner>> = Lazy::new(|| {
    RwLock::new(RegistryInner { providers: Vec::new() })
});

pub fn initialize_registry() {
    let mut reg = REGISTRY.write();
    reg.providers.clear();
    reg.providers.push(ProviderEntry { prefix: "gemini".into(), provider: Box::new(GeminiProvider::new()) });
    reg.providers.push(ProviderEntry { prefix: "zai".into(), provider: Box::new(ZAIProvider::new()) });
    for (prefix, provider_key) in &CONFIG.model_prefixes {
        if reg.providers.iter().any(|e| e.prefix == *prefix) { continue; }
        let provider: Box<dyn Provider + Send + Sync> = match provider_key.as_str() {
            "gemini" => Box::new(GeminiProvider::new()),
            "zai" => Box::new(ZAIProvider::new()),
            _ => continue,
        };
        reg.providers.push(ProviderEntry { prefix: prefix.clone(), provider });
    }
}

pub fn get_provider(model_name: &str) -> Box<dyn Provider + Send + Sync> {
    let reg = REGISTRY.read();
    for entry in &reg.providers {
        if model_name.starts_with(&entry.prefix) { return entry.provider.clone_box(); }
    }
    for entry in &reg.providers {
        if entry.prefix == "zai" { return entry.provider.clone_box(); }
    }
    Box::new(ZAIProvider::new())
}
