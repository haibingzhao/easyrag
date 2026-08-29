//! Document parsers and registry.
//!
//!  libraries are replaced by Rust equivalents:
//! - PDFBox → `pdf-extract`
//! - POI XWPF/XSLF (docx/pptx) → `zip` + `quick-xml`
//! - POI XSSF (xlsx) → `calamine`
//!
//! Legacy binary formats (`.doc`/`.ppt`) are intentionally NOT supported
//! (server returns 400), unlike the parser list.

use async_trait::async_trait;
use calamine::{open_workbook_auto_from_rs, Data, Reader};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use zip::ZipArchive;

/// Document parser interface.
#[async_trait]
pub trait DocumentParser: Send + Sync {
    /// Supported file extensions (lowercase, no dot).
    fn supported_extensions(&self) -> &[&str];

    /// Parse document content from bytes (the primary entry point;
    /// [`Self::parse_file`] defaults to reading the file and delegating).
    async fn parse_bytes(&self, bytes: &[u8], file_name: &str) -> easyrag_core::Result<String>;

    /// Parse document content from a file path.
    async fn parse_file(&self, path: &Path) -> easyrag_core::Result<String> {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| easyrag_core::CoreError::Internal(format!("read file: {e}")))?;
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        self.parse_bytes(&bytes, &file_name).await
    }
}

/// Plain text parser — handles .txt and other textual formats.
pub struct PlainTextParser;

#[async_trait]
impl DocumentParser for PlainTextParser {
    fn supported_extensions(&self) -> &[&str] {
        &[
            "txt", "text", "log", "csv", "tsv", "json", "xml", "yaml", "yml", "ini", "conf",
            "cfg", "properties",
        ]
    }

    async fn parse_bytes(&self, bytes: &[u8], _file_name: &str) -> easyrag_core::Result<String> {
        Ok(String::from_utf8_lossy(bytes).to_string())
    }
}

/// Markdown parser with optional YAML frontmatter stripping.
pub struct MarkdownParser;

impl MarkdownParser {
    fn strip_frontmatter(content: &str) -> String {
        if let Some(rest) = content.strip_prefix("---") {
            // Note: uses `indexOf("---", 3)`, which strips even when
            // the closing marker immediately follows (`------` -> "").
            // We require `end > 0` to avoid treating a lone rule as
            // frontmatter; deliberate edge-case deviation.
            if let Some(end) = rest.find("---") {
                if end > 0 {
                    return rest[end + 3..].trim().to_string();
                }
            }
        }
        content.to_string()
    }
}

#[async_trait]
impl DocumentParser for MarkdownParser {
    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown", "mdx"]
    }

    async fn parse_bytes(&self, bytes: &[u8], _file_name: &str) -> easyrag_core::Result<String> {
        let content = String::from_utf8_lossy(bytes).to_string();
        Ok(Self::strip_frontmatter(&content))
    }
}

/// PDF parser using `pdf-extract`.
pub struct PdfParser;

#[async_trait]
impl DocumentParser for PdfParser {
    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }

    async fn parse_bytes(&self, bytes: &[u8], file_name: &str) -> easyrag_core::Result<String> {
        let data = bytes.to_vec();
        let name = file_name.to_string();
        tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text_from_mem(&data)
                .map_err(|e| easyrag_core::CoreError::Internal(format!("parse {name}: {e}")))
        })
        .await
        .map_err(|e| easyrag_core::CoreError::Internal(format!("parse {file_name}: {e}")))?
    }
}

/// Compute a Markdown heading prefix from a docx paragraph style
/// (: styles starting with "Heading"/"heading", first digit =
/// level clamped to 1..6).
fn heading_prefix(style: &str) -> Option<String> {
    if style.starts_with("Heading") || style.starts_with("heading") {
        let level = style
            .chars()
            .find(|c| c.is_ascii_digit())
            .and_then(|c| c.to_digit(10))
            .unwrap_or(1)
            .clamp(1, 6) as usize;
        Some(format!("{} ", "#".repeat(level)))
    } else {
        None
    }
}

/// DOCX parser: zip + quick-xml extraction of paragraphs (with heading
/// markers) and tables (rows joined by " | ") — replicates the POI
/// bodyElements traversal order.
pub struct DocxParser;

impl DocxParser {
    fn extract_docx(bytes: &[u8]) -> easyrag_core::Result<String> {
        let mut archive = ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| easyrag_core::CoreError::Internal(format!("docx zip: {e}")))?;
        let mut document = archive
            .by_name("word/document.xml")
            .map_err(|e| easyrag_core::CoreError::Internal(format!("docx document.xml: {e}")))?;
        let mut xml = Vec::new();
        std::io::Read::read_to_end(&mut document, &mut xml)
            .map_err(|e| easyrag_core::CoreError::Internal(format!("docx read: {e}")))?;

        use quick_xml::events::Event;
        let mut reader = quick_xml::Reader::from_reader(xml.as_slice());
        let mut buf = Vec::new();

        let mut out = String::new();
        // Depth counters keep the state machine correct for self-closing
        // tags (no matching End event) and nested tables, which a plain
        // boolean would mishandle.
        let mut tbl_depth = 0usize;
        let mut tc_depth = 0usize;
        let mut in_t = false;
        let mut row_cells: Vec<String> = Vec::new();
        let mut cell_buf = String::new();
        let mut para_buf = String::new();
        let mut para_style = String::new();

        loop {
            match reader
                .read_event_into(&mut buf)
                .map_err(|e| easyrag_core::CoreError::Internal(format!("docx xml: {e}")))?
            {
                Event::Start(e) => {
                    match e.local_name().as_ref() {
                        b"tbl" => tbl_depth += 1,
                        b"tr" => row_cells.clear(),
                        b"tc" => {
                            if tc_depth == 0 {
                                cell_buf.clear();
                            }
                            tc_depth += 1;
                        }
                        b"p" => {
                            para_buf.clear();
                            para_style.clear();
                        }
                        b"pStyle" => {
                            for attr in e.attributes().flatten() {
                                if attr.key.as_ref() == b"w:val" || attr.key.as_ref() == b"val" {
                                    para_style =
                                        String::from_utf8_lossy(&attr.value).to_string();
                                }
                            }
                        }
                        b"t" => in_t = true,
                        _ => {}
                    }
                }
                // Self-closing elements never carry text or open scopes;
                // only `pStyle` attributes are readable here.
                Event::Empty(e) => {
                    if e.local_name().as_ref() == b"pStyle" {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"w:val" || attr.key.as_ref() == b"val" {
                                para_style = String::from_utf8_lossy(&attr.value).to_string();
                            }
                        }
                    }
                }
                Event::Text(e) => {
                    if in_t {
                        let text = e
                            .unescape()
                            .map_err(|err| {
                                easyrag_core::CoreError::Internal(format!("docx xml: {err}"))
                            })?
                            .to_string();
                        if tc_depth > 0 {
                            cell_buf.push_str(&text);
                        } else {
                            para_buf.push_str(&text);
                        }
                    }
                }
                Event::End(e) => match e.local_name().as_ref() {
                    b"t" => in_t = false,
                    b"tc" => {
                        tc_depth = tc_depth.saturating_sub(1);
                        if tc_depth == 0 {
                            row_cells.push(cell_buf.trim().to_string());
                        }
                    }
                    b"tr" => {
                        out.push_str(&row_cells.join(" | "));
                        out.push('\n');
                    }
                    b"tbl" => {
                        tbl_depth = tbl_depth.saturating_sub(1);
                        if tbl_depth == 0 {
                            out.push('\n');
                        }
                    }
                    b"p" => {
                        if tbl_depth == 0 && tc_depth == 0 {
                            let text = para_buf.trim();
                            if !text.is_empty() {
                                if let Some(prefix) = heading_prefix(&para_style) {
                                    out.push_str(&prefix);
                                }
                                out.push_str(text);
                                out.push('\n');
                            }
                        }
                    }
                    _ => {}
                },
                Event::Eof => break,
                _ => {}
            }
            buf.clear();
        }
        Ok(out)
    }
}

#[async_trait]
impl DocumentParser for DocxParser {
    fn supported_extensions(&self) -> &[&str] {
        &["docx"]
    }

    async fn parse_bytes(&self, bytes: &[u8], file_name: &str) -> easyrag_core::Result<String> {
        let data = bytes.to_vec();
        let name = file_name.to_string();
        tokio::task::spawn_blocking(move || {
            Self::extract_docx(&data).map_err(|e| {
                easyrag_core::CoreError::Internal(format!("parse {name}: {e}"))
            })
        })
        .await
        .map_err(|e| easyrag_core::CoreError::Internal(format!("parse {file_name}: {e}")))?
    }
}

/// PPTX parser: zip + quick-xml extraction of slide text.
/// Emits `## Slide N` headers like the POI implementation.
pub struct PptxParser;

impl PptxParser {
    fn extract_slide_text(xml: &[u8]) -> easyrag_core::Result<Vec<String>> {
        use quick_xml::events::Event;
        let mut reader = quick_xml::Reader::from_reader(xml);
        let mut buf = Vec::new();
        let mut lines = Vec::new();
        let mut para_buf = String::new();
        let mut in_para = false;
        let mut in_t = false;
        loop {
            match reader
                .read_event_into(&mut buf)
                .map_err(|e| easyrag_core::CoreError::Internal(format!("pptx xml: {e}")))?
            {
                Event::Start(e) => match e.local_name().as_ref() {
                    b"p" => {
                        in_para = true;
                        para_buf.clear();
                    }
                    b"t" => in_t = true,
                    _ => {}
                },
                Event::Text(e) => {
                    if in_t && in_para {
                        let text = e
                            .unescape()
                            .map_err(|err| {
                                easyrag_core::CoreError::Internal(format!("pptx xml: {err}"))
                            })?
                            .to_string();
                        para_buf.push_str(&text);
                    }
                }
                Event::End(e) => match e.local_name().as_ref() {
                    b"t" => in_t = false,
                    b"p" => {
                        let text = para_buf.trim().to_string();
                        if !text.is_empty() {
                            lines.push(text);
                        }
                        in_para = false;
                    }
                    _ => {}
                },
                Event::Eof => break,
                _ => {}
            }
            buf.clear();
        }
        Ok(lines)
    }

    fn extract_pptx(bytes: &[u8]) -> easyrag_core::Result<String> {
        let mut archive = ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| easyrag_core::CoreError::Internal(format!("pptx zip: {e}")))?;
        // Collect slide names and sort by slide number.
        let mut slides: Vec<(u32, String)> = Vec::new();
        for name in archive.file_names() {
            let name = name.to_string();
            if let Some(rest) = name.strip_prefix("ppt/slides/slide") {
                if let Some(num) = rest.strip_suffix(".xml") {
                    if let Ok(n) = num.parse::<u32>() {
                        slides.push((n, name));
                    }
                }
            }
        }
        slides.sort_by_key(|(n, _)| *n);

        let mut out = String::new();
        for (position, (_number, slide_name)) in slides.iter().enumerate() {
            let mut slide = archive
                .by_name(slide_name)
                .map_err(|e| easyrag_core::CoreError::Internal(format!("pptx slide: {e}")))?;
            let mut xml = Vec::new();
            std::io::Read::read_to_end(&mut slide, &mut xml)
                .map_err(|e| easyrag_core::CoreError::Internal(format!("pptx read: {e}")))?;
            drop(slide);
            out.push_str(&format!("## Slide {}\n", position + 1));
            for line in Self::extract_slide_text(&xml)? {
                out.push_str(&line);
                out.push('\n');
            }
            out.push('\n');
        }
        Ok(out)
    }
}

#[async_trait]
impl DocumentParser for PptxParser {
    fn supported_extensions(&self) -> &[&str] {
        &["pptx"]
    }

    async fn parse_bytes(&self, bytes: &[u8], file_name: &str) -> easyrag_core::Result<String> {
        let data = bytes.to_vec();
        let name = file_name.to_string();
        tokio::task::spawn_blocking(move || {
            Self::extract_pptx(&data).map_err(|e| {
                easyrag_core::CoreError::Internal(format!("parse {name}: {e}"))
            })
        })
        .await
        .map_err(|e| easyrag_core::CoreError::Internal(format!("parse {file_name}: {e}")))?
    }
}

/// XLSX/XLS parser using calamine.
/// Numbers are rendered with decimal notation (e.g. "42.0").
pub struct XlsxParser;

impl XlsxParser {
    fn format_cell(value: &Data) -> String {
        match value {
            Data::Float(f) => {
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{:.1}", f)
                } else {
                    f.to_string()
                }
            }
            // Integers render with one decimal like Double.toString
            // ("42.0"); formatting the i64 directly avoids the f64
            // precision loss for values beyond 2^53.
            Data::Int(i) => format!("{:.1}", i),
            Data::Bool(b) => b.to_string(),
            Data::String(s) => s.clone(),
            _ => String::new(),
        }
    }

    fn extract_xlsx(bytes: &[u8]) -> easyrag_core::Result<String> {
        // Auto-detect the workbook type so both .xlsx (OOXML) and legacy
        // .xls (BIFF) are supported, matching POI's WorkbookFactory.
        let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes))
            .map_err(|e| easyrag_core::CoreError::Internal(format!("xlsx open: {e}")))?;
        let mut out = String::new();
        let names = workbook.sheet_names().to_vec();
        for name in names {
            out.push_str(&format!("## Sheet: {}\n", name));
            let range = workbook
                .worksheet_range(&name)
                .map_err(|e| easyrag_core::CoreError::Internal(format!("xlsx sheet: {e}")))?;
            for row in range.rows() {
                let cells: Vec<String> = row.iter().map(Self::format_cell).collect();
                out.push_str(&cells.join(" | "));
                out.push('\n');
            }
            out.push('\n');
        }
        Ok(out)
    }
}

#[async_trait]
impl DocumentParser for XlsxParser {
    fn supported_extensions(&self) -> &[&str] {
        &["xlsx", "xls"]
    }

    async fn parse_bytes(&self, bytes: &[u8], file_name: &str) -> easyrag_core::Result<String> {
        let data = bytes.to_vec();
        let name = file_name.to_string();
        tokio::task::spawn_blocking(move || {
            Self::extract_xlsx(&data).map_err(|e| {
                easyrag_core::CoreError::Internal(format!("parse {name}: {e}"))
            })
        })
        .await
        .map_err(|e| easyrag_core::CoreError::Internal(format!("parse {file_name}: {e}")))?
    }
}

/// HTML parser — regex tag stripping (verbatim strategy).
pub struct HtmlParser;

impl HtmlParser {
    fn strip_html(html: &str) -> String {
        static SCRIPT: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap());
        static STYLE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap());
        static TAGS: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());
        static SPACES: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

        let s = SCRIPT.replace_all(html, "");
        let s = STYLE.replace_all(&s, "");
        let s = TAGS.replace_all(&s, " ");
        // Decode order matches the original verbatim (`&amp;`
        // before `&lt;`/`&gt;`), kept for consistency even though decoding
        // `&amp;` last would be strictly more correct HTML behavior.
        let s = s.replace("&nbsp;", " ").replace("&amp;", "&");
        let s = s.replace("&lt;", "<").replace("&gt;", ">");
        SPACES.replace_all(&s, " ").trim().to_string()
    }
}

#[async_trait]
impl DocumentParser for HtmlParser {
    fn supported_extensions(&self) -> &[&str] {
        &["html", "htm"]
    }

    async fn parse_bytes(&self, bytes: &[u8], _file_name: &str) -> easyrag_core::Result<String> {
        let content = String::from_utf8_lossy(bytes).to_string();
        Ok(Self::strip_html(&content))
    }
}

/// Code file parser — reads source files as plain text.
pub struct CodeFileParser;

#[async_trait]
impl DocumentParser for CodeFileParser {
    fn supported_extensions(&self) -> &[&str] {
        &[
            "py", "java", "kt", "kts", "js", "ts", "jsx", "tsx", "go", "rs", "c", "cpp", "h",
            "cs", "rb", "php", "swift", "scala", "sh", "bash", "sql", "r", "dart", "vue",
            "svelte",
        ]
    }

    async fn parse_bytes(&self, bytes: &[u8], _file_name: &str) -> easyrag_core::Result<String> {
        Ok(String::from_utf8_lossy(bytes).to_string())
    }
}

/// Per-document processing options.
#[derive(Clone, Debug, Default)]
pub struct DocumentProcessingOptions {
    pub chunk_method: Option<String>,
    pub chunk_token_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub skip_kg: bool,
    pub parser_engine: Option<String>,
    pub build_structure: bool,
}

/// Parser registry — routes files to parsers by extension with optional
/// `file.[engine].ext` hint syntax.
pub struct ParserRegistry {
    parser_map: HashMap<String, Arc<dyn DocumentParser>>,
    named_engines: HashMap<String, Arc<dyn DocumentParser>>,
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::with_parsers(vec![
            Arc::new(PlainTextParser),
            Arc::new(MarkdownParser),
            Arc::new(PdfParser),
            Arc::new(DocxParser),
            Arc::new(PptxParser),
            Arc::new(XlsxParser),
            Arc::new(HtmlParser),
            Arc::new(CodeFileParser),
        ])
    }
}

impl ParserRegistry {
    pub fn with_parsers(parsers: Vec<Arc<dyn DocumentParser>>) -> Self {
        let mut parser_map = HashMap::new();
        for parser in &parsers {
            for ext in parser.supported_extensions() {
                parser_map.insert(ext.to_lowercase(), Arc::clone(parser));
            }
        }
        // Named engines for hint routing. PDFBox/POI hints map to the
        // Rust equivalents. keeps six fixed engines regardless of
        // the parser list, so missing engines fall back to defaults below.
        let mut named_engines: HashMap<String, Arc<dyn DocumentParser>> = HashMap::new();
        for parser in &parsers {
            let exts = parser.supported_extensions();
            // Route hint engines to whatever parsers are registered.
            if exts.contains(&"pdf") {
                named_engines
                    .entry("pdfbox".to_string())
                    .or_insert_with(|| Arc::clone(parser));
            }
            if exts.contains(&"docx") {
                named_engines
                    .entry("poi".to_string())
                    .or_insert_with(|| Arc::clone(parser));
            }
            if exts.contains(&"md") {
                named_engines
                    .entry("markdown".to_string())
                    .or_insert_with(|| Arc::clone(parser));
            }
            if exts.contains(&"html") {
                named_engines
                    .entry("html".to_string())
                    .or_insert_with(|| Arc::clone(parser));
            }
            if exts.contains(&"py") {
                named_engines
                    .entry("code".to_string())
                    .or_insert_with(|| Arc::clone(parser));
            }
            if exts.contains(&"log") {
                named_engines
                    .entry("plain".to_string())
                    .or_insert_with(|| Arc::clone(parser));
            }
        }
        // Fill engines absent from the registered parsers with default
        // instances so all six hint names stay available.
        named_engines
            .entry("pdfbox".to_string())
            .or_insert_with(|| Arc::new(PdfParser));
        named_engines
            .entry("poi".to_string())
            .or_insert_with(|| Arc::new(DocxParser));
        named_engines
            .entry("plain".to_string())
            .or_insert_with(|| Arc::new(PlainTextParser));
        named_engines
            .entry("markdown".to_string())
            .or_insert_with(|| Arc::new(MarkdownParser));
        named_engines
            .entry("html".to_string())
            .or_insert_with(|| Arc::new(HtmlParser));
        named_engines
            .entry("code".to_string())
            .or_insert_with(|| Arc::new(CodeFileParser));
        Self {
            parser_map,
            named_engines,
        }
    }

    /// Get parser for a file extension (leading dot tolerated).
    pub fn get_parser(&self, extension: &str) -> Option<Arc<dyn DocumentParser>> {
        self.parser_map
            .get(&extension.to_lowercase().trim_start_matches('.').to_string())
            .cloned()
    }

    /// Get parser for a filename, honoring the `file.[engine].ext` hint.
    pub fn get_parser_for_file_name(&self, file_name: &str) -> Option<Arc<dyn DocumentParser>> {
        static HINT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[(\w+)\]\.").unwrap());
        if let Some(m) = HINT.captures(file_name) {
            let engine_name = m[1].to_lowercase();
            if let Some(engine) = self.named_engines.get(&engine_name) {
                tracing::debug!("Using hinted engine '{engine_name}' for file: {file_name}");
                return Some(Arc::clone(engine));
            }
            tracing::warn!(
                "Unknown parser engine hint '{engine_name}' for file: {file_name}, \
                 falling back to extension routing"
            );
        }
        let clean_name = HINT.replace_all(file_name, ".");
        let ext = clean_name.rsplit('.').next().unwrap_or("");
        let ext = if clean_name.contains('.') { ext } else { "" };
        self.get_parser(ext)
    }

    /// Get parser for a file path.
    pub fn get_parser_for_file(&self, path: &Path) -> Option<Arc<dyn DocumentParser>> {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        self.get_parser_for_file_name(&name)
    }

    /// Check if an extension is supported.
    pub fn is_supported(&self, extension: &str) -> bool {
        self.parser_map
            .contains_key(extension.to_lowercase().trim_start_matches('.'))
    }

    /// All supported extensions.
    pub fn supported_extensions(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.parser_map.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// All registered engine names.
    pub fn registered_engines(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.named_engines.keys().cloned().collect();
        keys.sort();
        keys
    }
}
