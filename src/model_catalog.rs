use parking_lot::RwLock;
use serde::Serialize;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
pub struct ProviderModelsSnapshot {
    pub provider: String,
    pub updated_at_unix_seconds: Option<u64>,
    pub models: Vec<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
struct ProviderModelsEntry {
    updated_at: Option<SystemTime>,
    models: Vec<String>,
    last_error: Option<String>,
}

#[derive(Default)]
pub struct ModelCatalog {
    providers: RwLock<HashMap<String, ProviderModelsEntry>>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_success(&self, provider: &str, mut models: Vec<String>) {
        models.sort();
        models.dedup();
        self.providers.write().insert(
            provider.to_string(),
            ProviderModelsEntry {
                updated_at: Some(SystemTime::now()),
                models,
                last_error: None,
            },
        );
    }

    pub fn update_error(&self, provider: &str, error: String) {
        let mut guard = self.providers.write();
        let entry = guard
            .entry(provider.to_string())
            .or_insert_with(|| ProviderModelsEntry {
                updated_at: None,
                models: Vec::new(),
                last_error: None,
            });
        entry.last_error = Some(error);
        entry.updated_at = Some(SystemTime::now());
    }

    pub fn models_for_provider(&self, provider: &str) -> Option<Vec<String>> {
        self.providers
            .read()
            .get(provider)
            .map(|entry| entry.models.clone())
    }

    pub fn snapshot(&self) -> Vec<ProviderModelsSnapshot> {
        self.providers
            .read()
            .iter()
            .map(|(provider, entry)| ProviderModelsSnapshot {
                provider: provider.clone(),
                updated_at_unix_seconds: entry
                    .updated_at
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs()),
                models: entry.models.clone(),
                last_error: entry.last_error.clone(),
            })
            .collect()
    }
}
