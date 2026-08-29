//! Entity extraction prompt profile management

use crate::error::CoreError;
use crate::prompt;
use crate::JsonMap;
use std::path::{Component, Path, PathBuf};

const DEFAULT_PROMPT_DIR: &str = "./prompts";
const ENTITY_TYPE_SUBDIR: &str = "entity_type";
const ALLOWED_PROMPT_SUFFIXES: [&str; 2] = [".yml", ".yaml"];

/// Entity extraction prompt profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityExtractionPromptProfile {
    pub entity_types_guidance: String,
    pub entity_extraction_examples: Vec<String>,
    pub entity_extraction_json_examples: Vec<String>,
}

pub struct PromptProfileResolver;

impl PromptProfileResolver {
    pub fn get_default_profile() -> EntityExtractionPromptProfile {
        EntityExtractionPromptProfile {
            entity_types_guidance: prompt::DEFAULT_ENTITY_TYPES_GUIDANCE.trim_end().to_string(),
            entity_extraction_examples: prompt::ENTITY_EXTRACTION_EXAMPLES
                .iter()
                .map(|s| s.trim_end().to_string())
                .collect(),
            entity_extraction_json_examples: prompt::ENTITY_EXTRACTION_JSON_EXAMPLES
                .iter()
                .map(|s| s.trim_end().to_string())
                .collect(),
        }
    }

    pub fn get_entity_type_prompt_dir() -> PathBuf {
        let configured = std::env::var("PROMPT_DIR")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_PROMPT_DIR.to_string());
        let base = match Path::new(&configured).is_absolute() {
            true => PathBuf::from(configured),
            false => std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(configured),
        };
        normalize_path(&base.join(ENTITY_TYPE_SUBDIR))
    }

    pub fn resolve_entity_type_prompt_path(prompt_file_name: &str) -> crate::Result<PathBuf> {
        let file_name = prompt_file_name.trim();
        if file_name.is_empty() {
            return Err(CoreError::InvalidArgument(
                "ENTITY_TYPE_PROMPT_FILE must be a file name such as 'entity_type_prompt.sample.yml'."
                    .to_string(),
            ));
        }
        if file_name.contains('\\') {
            return Err(CoreError::InvalidArgument(
                "ENTITY_TYPE_PROMPT_FILE must not contain directory separators. \
                 Only file names inside PROMPT_DIR/entity_type are allowed."
                    .to_string(),
            ));
        }

        let candidate = Path::new(file_name);
        let candidate_file_name = candidate
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if candidate.is_absolute() || candidate_file_name != file_name || file_name.contains("..")
        {
            return Err(CoreError::InvalidArgument(
                "ENTITY_TYPE_PROMPT_FILE must be a file name only. \
                 Files are loaded from PROMPT_DIR/entity_type (PROMPT_DIR defaults to ./prompts)."
                    .to_string(),
            ));
        }
        let suffix = file_name
            .rsplit_once('.')
            .map(|(_, s)| s.to_lowercase())
            .unwrap_or_default();
        if !ALLOWED_PROMPT_SUFFIXES.contains(&format!(".{suffix}").as_str()) {
            return Err(CoreError::InvalidArgument(
                "ENTITY_TYPE_PROMPT_FILE must use a '.yml' or '.yaml' extension.".to_string(),
            ));
        }

        Ok(Self::get_entity_type_prompt_dir().join(&candidate_file_name))
    }

    /// Resolve and merge the configured entity extraction prompt profile.
    /// YAML loading is simplified — addon_params can provide overrides directly.
    pub fn resolve_profile(
        addon_params: Option<&JsonMap>,
        _use_json: bool,
    ) -> EntityExtractionPromptProfile {
        let default_profile = Self::get_default_profile();
        let empty = JsonMap::new();
        let params = addon_params.unwrap_or(&empty);

        let guidance_override = params
            .get("entity_types_guidance")
            .and_then(|v| v.as_str());
        let guidance = match guidance_override {
            Some(g) if !g.trim().is_empty() => g.trim_end().to_string(),
            _ => default_profile.entity_types_guidance,
        };

        let examples_override = read_string_list(params.get("entity_extraction_examples"));
        let json_examples_override =
            read_string_list(params.get("entity_extraction_json_examples"));

        EntityExtractionPromptProfile {
            entity_types_guidance: guidance,
            entity_extraction_examples: examples_override
                .map(|list| list.into_iter().map(|s| s.trim_end().to_string()).collect())
                .unwrap_or(default_profile.entity_extraction_examples),
            entity_extraction_json_examples: json_examples_override
                .map(|list| list.into_iter().map(|s| s.trim_end().to_string()).collect())
                .unwrap_or(default_profile.entity_extraction_json_examples),
        }
    }

    pub fn validate_profile_for_mode(
        profile: EntityExtractionPromptProfile,
        use_json: bool,
    ) -> crate::Result<EntityExtractionPromptProfile> {
        if use_json {
            if profile.entity_extraction_json_examples.is_empty() {
                return Err(CoreError::InvalidArgument(
                    "The resolved prompt profile must define 'entityExtractionJsonExamples' \
                     when entity extraction runs in json mode."
                        .to_string(),
                ));
            }
        } else if profile.entity_extraction_examples.is_empty() {
            return Err(CoreError::InvalidArgument(
                "The resolved prompt profile must define 'entityExtractionExamples' \
                 when entity extraction runs in text mode."
                    .to_string(),
            ));
        }
        Ok(profile)
    }
}

fn read_string_list(value: Option<&serde_json::Value>) -> Option<Vec<String>> {
    let arr = value?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        out.push(item.as_str()?.to_string());
    }
    Some(out)
}

/// Lexical path normalization (equivalent of `Path.normalize`):
/// drop `.` components and resolve `..` against preceding components.
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !matches!(out.components().last(), Some(Component::RootDir)) {
                    out.pop();
                }
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_is_non_empty() {
        let profile = PromptProfileResolver::get_default_profile();
        assert!(!profile.entity_types_guidance.is_empty());
        assert!(!profile.entity_extraction_examples.is_empty());
        assert!(!profile.entity_extraction_json_examples.is_empty());
    }

    #[test]
    fn resolve_profile_without_params_returns_defaults() {
        let profile = PromptProfileResolver::resolve_profile(None, false);
        assert_eq!(profile, PromptProfileResolver::get_default_profile());
    }

    #[test]
    fn resolve_profile_applies_guidance_override() {
        let mut params = JsonMap::new();
        params.insert(
            "entity_types_guidance".to_string(),
            serde_json::Value::String("custom guidance  ".to_string()),
        );
        let profile = PromptProfileResolver::resolve_profile(Some(&params), false);
        assert_eq!("custom guidance", profile.entity_types_guidance);
    }

    #[test]
    fn resolve_profile_ignores_blank_guidance_override() {
        let mut params = JsonMap::new();
        params.insert(
            "entity_types_guidance".to_string(),
            serde_json::Value::String("   ".to_string()),
        );
        let profile = PromptProfileResolver::resolve_profile(Some(&params), false);
        assert_eq!(
            PromptProfileResolver::get_default_profile().entity_types_guidance,
            profile.entity_types_guidance
        );
    }

    #[test]
    fn resolve_profile_applies_examples_override() {
        let mut params = JsonMap::new();
        params.insert(
            "entity_extraction_examples".to_string(),
            serde_json::json!(["example-a\n", "example-b"]),
        );
        let profile = PromptProfileResolver::resolve_profile(Some(&params), false);
        assert_eq!(vec!["example-a", "example-b"], profile.entity_extraction_examples);
    }

    #[test]
    fn validate_profile_for_mode_rejects_empty_examples() {
        let profile = EntityExtractionPromptProfile {
            entity_types_guidance: "g".to_string(),
            entity_extraction_examples: Vec::new(),
            entity_extraction_json_examples: vec!["j".to_string()],
        };
        assert!(PromptProfileResolver::validate_profile_for_mode(profile.clone(), false).is_err());
        assert!(PromptProfileResolver::validate_profile_for_mode(profile, true).is_ok());
    }

    #[test]
    fn resolve_entity_type_prompt_path_rejects_invalid_names() {
        assert!(PromptProfileResolver::resolve_entity_type_prompt_path("").is_err());
        assert!(PromptProfileResolver::resolve_entity_type_prompt_path("a.txt").is_err());
        assert!(PromptProfileResolver::resolve_entity_type_prompt_path("../evil.yml").is_err());
        assert!(PromptProfileResolver::resolve_entity_type_prompt_path("sub/a.yml").is_err());
        assert!(PromptProfileResolver::resolve_entity_type_prompt_path("a\\b.yml").is_err());
    }

    #[test]
    fn resolve_entity_type_prompt_path_accepts_yaml_name() {
        let path =
            PromptProfileResolver::resolve_entity_type_prompt_path("entity_type_prompt.sample.yml")
                .unwrap();
        assert!(path.ends_with(Path::new("entity_type/entity_type_prompt.sample.yml")));
    }
}
