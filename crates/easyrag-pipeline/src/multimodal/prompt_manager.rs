//! Multilingual prompt manager.
//! Global singleton with atomic language switching; missing keys in the
//! target language fall back to English.

use std::collections::HashMap;
use std::sync::Mutex;

use super::{prompts_en, prompts_zh};

struct PromptManagerInner {
    current_language: String,
    active_prompts: HashMap<String, String>,
    prompt_languages: HashMap<String, HashMap<String, String>>,
}

fn inner() -> &'static Mutex<PromptManagerInner> {
    static INNER: std::sync::OnceLock<Mutex<PromptManagerInner>> = std::sync::OnceLock::new();
    INNER.get_or_init(|| {
        let english = prompts_en::as_map();
        let mut prompt_languages = HashMap::new();
        prompt_languages.insert("en".to_string(), english.clone());
        Mutex::new(PromptManagerInner {
            current_language: "en".to_string(),
            active_prompts: english,
            prompt_languages,
        })
    })
}

/// Get the currently active prompt set.
pub fn get_prompts() -> HashMap<String, String> {
    inner().lock().unwrap().active_prompts.clone()
}

/// Get a specific prompt by key from the active prompt set.
/// Falls back to English; errors on unknown keys
///
pub fn get(key: &str) -> easyrag_core::Result<String> {
    let guard = inner().lock().unwrap();
    if let Some(v) = guard.active_prompts.get(key) {
        return Ok(v.clone());
    }
    let english = guard
        .prompt_languages
        .get("en")
        .expect("english prompts registered at init");
    english
        .get(key)
        .cloned()
        .ok_or_else(|| easyrag_core::CoreError::InvalidArgument(format!("Unknown prompt key: {key}")))
}

/// Get a prompt with `{placeholder}` substitution.
pub fn format(key: &str, args: &[(&str, &str)]) -> easyrag_core::Result<String> {
    let mut result = get(key)?;
    for (placeholder, value) in args {
        result = result.replace(&format!("{{{placeholder}}}"), value);
    }
    Ok(result)
}

/// Get the currently active prompt language code.
pub fn get_language() -> String {
    inner().lock().unwrap().current_language.clone()
}

/// Switch the active prompt language. Missing keys in the target language
/// fall back to English. `zh` is lazy-loaded on first use.
pub fn set_language(language: &str) -> easyrag_core::Result<()> {
    let lang = language.trim().to_lowercase();
    if lang.is_empty() {
        return Err(easyrag_core::CoreError::InvalidArgument(
            "Language code must be a non-empty string".to_string(),
        ));
    }

    let mut guard = inner().lock().unwrap();
    if !guard.prompt_languages.contains_key(&lang) {
        // Lazy load: zh is built-in; anything else must be registered.
        if lang == "zh" {
            guard
                .prompt_languages
                .insert("zh".to_string(), prompts_zh::as_map());
        }
    }
    let target = guard.prompt_languages.get(&lang).cloned();
    let Some(target_prompts) = target else {
        let mut available: Vec<String> = guard.prompt_languages.keys().cloned().collect();
        if !available.iter().any(|l| l == "zh") {
            available.push("zh".to_string());
        }
        available.sort();
        return Err(easyrag_core::CoreError::InvalidArgument(format!(
            "Unknown prompt language '{language}'. Available: {}. Register new languages with register_language().",
            available.join(", ")
        )));
    };

    // Resolve: target language overrides, English fallback for missing keys.
    let english = guard
        .prompt_languages
        .get("en")
        .expect("english prompts registered at init")
        .clone();
    let mut resolved = english;
    resolved.extend(target_prompts);

    // Atomic swap (single mutex guards both fields).
    guard.active_prompts = resolved;
    guard.current_language = lang.clone();
    drop(guard);
    tracing::info!("Prompt language set to '{}'", lang);
    Ok(())
}

/// Reset all prompts back to the default English templates.
pub fn reset() {
    let mut guard = inner().lock().unwrap();
    guard.active_prompts = guard
        .prompt_languages
        .get("en")
        .expect("english prompts registered at init")
        .clone();
    guard.current_language = "en".to_string();
    drop(guard);
    tracing::info!("Prompts reset to English defaults");
}

/// Register a new set of prompt templates for a language.
pub fn register_language(language_code: &str, prompts: HashMap<String, String>) {
    let lang = language_code.trim().to_lowercase();
    assert!(!lang.is_empty(), "Language code must be a non-empty string");
    let count = prompts.len();
    inner()
        .lock()
        .unwrap()
        .prompt_languages
        .insert(lang.clone(), prompts);
    tracing::info!("Registered prompt language '{}' with {} templates", lang, count);
}

/// Get all available language codes (`zh` always listed).
pub fn get_available_languages() -> Vec<String> {
    let guard = inner().lock().unwrap();
    let mut langs: Vec<String> = guard.prompt_languages.keys().cloned().collect();
    if !langs.iter().any(|l| l == "zh") {
        langs.push("zh".to_string());
    }
    langs.sort();
    langs
}
