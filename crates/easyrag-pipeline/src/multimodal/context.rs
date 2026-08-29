//! Context extraction for multimodal items

use easyrag_core::JsonMap;

use super::model::ContentItem;

/// Configuration for context extraction.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    pub context_window: i64,
    pub context_mode: String,
    pub max_context_tokens: usize,
    pub include_headers: bool,
    pub include_captions: bool,
    pub filter_content_types: Vec<String>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            context_window: 1,
            context_mode: "page".to_string(),
            max_context_tokens: 2000,
            include_headers: true,
            include_captions: true,
            filter_content_types: vec!["text".to_string()],
        }
    }
}

/// Universal context extractor supporting multiple content source formats.
/// Supports page/chunk modes for context extraction.
pub struct ContextExtractor {
    config: ContextConfig,
    /// Optional token counter; when absent, `max_context_tokens` is treated
    /// as a character limit.
    tokenizer: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>,
}

impl ContextExtractor {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            tokenizer: None,
        }
    }

    pub fn with_tokenizer<F>(mut self, tokenizer: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        self.tokenizer = Some(Box::new(tokenizer));
        self
    }

    /// Extract context for the current item from the content source.
    /// `current_item_info` carries `page_idx` / `index` keys.
    pub fn extract_context(
        &self,
        content_list: &[ContentItem],
        current_item_info: &JsonMap,
    ) -> String {
        if content_list.is_empty() && self.config.context_window == 0 {
            return String::new();
        }

        match self.config.context_mode.as_str() {
            "page" => self.extract_page_context(content_list, current_item_info),
            "chunk" => self.extract_chunk_context(content_list, current_item_info),
            _ => self.extract_page_context(content_list, current_item_info),
        }
    }

    /// Extract context from plain text source.
    pub fn extract_from_text(&self, text_source: &str) -> String {
        self.truncate_context(text_source)
    }

    /// Extract context from text chunks list.
    pub fn extract_from_text_chunks(
        &self,
        text_chunks: &[String],
        current_item_info: &JsonMap,
    ) -> String {
        let current_index = info_i64(current_item_info, "index").unwrap_or(0);
        let window_size = self.config.context_window;

        let start_idx = (current_index - window_size).max(0) as usize;
        let end_idx = ((current_index + window_size + 1).max(0) as usize).min(text_chunks.len());

        let mut context_texts: Vec<String> = Vec::new();
        for i in start_idx..end_idx {
            if i as i64 != current_index {
                let chunk_text = text_chunks[i].trim();
                if !chunk_text.is_empty() {
                    context_texts.push(chunk_text.to_string());
                }
            }
        }

        self.truncate_context(&context_texts.join("\n"))
    }

    fn extract_page_context(
        &self,
        content_list: &[ContentItem],
        current_item_info: &JsonMap,
    ) -> String {
        let current_page = info_i64(current_item_info, "page_idx").unwrap_or(0);
        let window_size = self.config.context_window;

        let start_page = (current_page - window_size).max(0);
        let end_page = current_page + window_size + 1;

        let mut context_texts: Vec<String> = Vec::new();

        for item in content_list {
            let item_page = item.page_idx();
            let item_type = item.modal_type().value();

            if item_page >= start_page
                && item_page < end_page
                && self.config.filter_content_types.iter().any(|t| t == item_type)
            {
                let text_content = self.extract_text_from_item(item);
                if !text_content.trim().is_empty() {
                    if item_page != current_page {
                        context_texts.push(format!("[Page {item_page}] {text_content}"));
                    } else {
                        context_texts.push(text_content);
                    }
                }
            }
        }

        self.truncate_context(&context_texts.join("\n"))
    }

    fn extract_chunk_context(
        &self,
        content_list: &[ContentItem],
        current_item_info: &JsonMap,
    ) -> String {
        let current_index = info_i64(current_item_info, "index").unwrap_or(0);
        let window_size = self.config.context_window;

        let start_idx = (current_index - window_size).max(0) as usize;
        let end_idx =
            ((current_index + window_size + 1).max(0) as usize).min(content_list.len());

        let mut context_texts: Vec<String> = Vec::new();

        for i in start_idx..end_idx {
            if i as i64 != current_index {
                let item = &content_list[i];
                if self
                    .config
                    .filter_content_types
                    .iter()
                    .any(|t| t == item.modal_type().value())
                {
                    let text_content = self.extract_text_from_item(item);
                    if !text_content.trim().is_empty() {
                        context_texts.push(text_content);
                    }
                }
            }
        }

        self.truncate_context(&context_texts.join("\n"))
    }

    fn extract_text_from_item(&self, item: &ContentItem) -> String {
        match item {
            ContentItem::Text {
                text, text_level, ..
            } => {
                if self.config.include_headers && *text_level > 0 {
                    format!("{} {}", "#".repeat(*text_level as usize), text)
                } else {
                    text.clone()
                }
            }
            ContentItem::Image { captions, .. } => {
                if self.config.include_captions && !captions.is_empty() {
                    format!("[Image: {}]", captions.join(", "))
                } else {
                    String::new()
                }
            }
            ContentItem::Table { captions, .. } => {
                if self.config.include_captions && !captions.is_empty() {
                    format!("[Table: {}]", captions.join(", "))
                } else {
                    String::new()
                }
            }
            ContentItem::Equation { .. } | ContentItem::Generic { .. } => String::new(),
        }
    }

    fn truncate_context(&self, context: &str) -> String {
        if context.is_empty() {
            return String::new();
        }

        if let Some(tokenizer) = &self.tokenizer {
            let token_count = tokenizer(context);
            if token_count <= self.config.max_context_tokens {
                return context.to_string();
            }

            // Character-based approximation: assume ~4 chars per token.
            let max_chars = self.config.max_context_tokens * 4;
            if char_len(context) <= max_chars {
                return context.to_string();
            }

            let truncated = take_chars(context, max_chars);
            truncate_at_boundary(&truncated)
        } else {
            // Fallback: treat max_context_tokens as character limit.
            if char_len(context) <= self.config.max_context_tokens {
                return context.to_string();
            }

            let truncated = take_chars(context, self.config.max_context_tokens);
            truncate_at_boundary(&truncated)
        }
    }
}

fn info_i64(info: &JsonMap, key: &str) -> Option<i64> {
    info.get(key).and_then(|v| v.as_i64())
}

/// Char-count helpers to mirror code-point semantics.
fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn truncate_at_boundary(truncated: &str) -> String {
    let chars: Vec<char> = truncated.chars().collect();
    let len = chars.len();
    let last_period = chars.iter().rposition(|c| *c == '.');
    let last_newline = chars.iter().rposition(|c| *c == '\n');

    let threshold = len as f64 * 0.8;
    if let Some(p) = last_period {
        if (p as f64) > threshold {
            return chars[..=p].iter().collect();
        }
    }
    if let Some(n) = last_newline {
        if (n as f64) > threshold {
            return chars[..n].iter().collect();
        }
    }
    format!("{truncated}...")
}
