//! Document parser tests: per-parser behavior, registry routing,
//! hint syntax, and hand-crafted docx/pptx/xlsx zip fixtures.

use easyrag_pipeline::{DocumentParser, TextParserRegistry};
use std::io::{Cursor, Write};

fn build_zip(files: &[(&str, &str)]) -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        for (name, content) in files {
            zip.start_file::<String, ()>(
                (*name).to_string(),
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
            zip.write_all(content.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
    }
    buf.into_inner()
}

#[tokio::test]
async fn plain_text_parser_reads_utf8() {
    let registry = TextParserRegistry::default();
    let parser = registry.get_parser("txt").unwrap();
    let text = parser.parse_bytes("héllo".as_bytes(), "a.txt").await.unwrap();
    assert_eq!("héllo", text);
}

#[tokio::test]
async fn markdown_parser_strips_frontmatter() {
    let parser = easyrag_pipeline::MarkdownParser;
    let with_fm = "---\ntitle: x\n---\n# Body\ntext";
    assert_eq!("# Body\ntext", parser.parse_bytes(with_fm.as_bytes(), "a.md").await.unwrap());

    let without_fm = "# Just body";
    assert_eq!("# Just body", parser.parse_bytes(without_fm.as_bytes(), "a.md").await.unwrap());

    // Unterminated frontmatter: returned as-is.
    let unterminated = "---\nno closing";
    assert_eq!(unterminated, parser.parse_bytes(unterminated.as_bytes(), "a.md").await.unwrap());
}

#[tokio::test]
async fn html_parser_strips_scripts_styles_tags_and_entities() {
    let parser = easyrag_pipeline::HtmlParser;
    let html = "<html><head><script>var x = 1;</script>\
                <style>.a{color:red}</style></head>\
                <body><h1>Title</h1><p>a&amp;b&nbsp;&lt;ok&gt;</p></body></html>";
    let text = parser.parse_bytes(html.as_bytes(), "a.html").await.unwrap();
    assert_eq!("Title a&b <ok>", text);
}

#[tokio::test]
async fn code_parser_supports_expected_extensions() {
    let parser = easyrag_pipeline::CodeFileParser;
    let exts = parser.supported_extensions();
    for expected in ["py", "java", "kt", "rs", "ts", "vue", "svelte"] {
        assert!(exts.contains(&expected), "missing {expected}");
    }
    assert_eq!(
        "fn main() {}",
        parser.parse_bytes(b"fn main() {}", "m.rs").await.unwrap()
    );
}

#[tokio::test]
async fn registry_routes_by_extension_case_insensitively() {
    let registry = TextParserRegistry::default();
    assert!(registry.get_parser("MD").is_some());
    assert!(registry.get_parser(".json").is_some());
    assert!(registry.get_parser("docx").is_some());
    assert!(registry.is_supported("PDF"));
    // Legacy binary formats are explicitly unsupported (server returns 400).
    assert!(!registry.is_supported("doc"));
    assert!(!registry.is_supported("ppt"));
    assert!(registry.get_parser("exe").is_none());
}

#[tokio::test]
async fn registry_hint_syntax_routes_to_named_engine() {
    let registry = TextParserRegistry::default();
    // Hinted engine wins over the extension.
    let parser = registry.get_parser_for_file_name("report.[markdown].pdf").unwrap();
    assert!(parser.supported_extensions().contains(&"md"));
    // Unknown hint falls back to extension routing.
    let parser = registry.get_parser_for_file_name("report.[bogus].txt").unwrap();
    assert!(parser.supported_extensions().contains(&"txt"));
    // Hinted engine wins even over the real extension (: hint
    // short-circuits before extension routing).
    let parser = registry.get_parser_for_file_name("notes.[pdfbox].md").unwrap();
    assert!(parser.supported_extensions().contains(&"pdf"));
    // Registered engines contain the names.
    let engines = registry.registered_engines();
    for name in ["pdfbox", "poi", "plain", "markdown", "html", "code"] {
        assert!(engines.iter().any(|e| e == name), "missing engine {name}");
    }
}

#[tokio::test]
async fn docx_parser_extracts_headings_paragraphs_and_tables() {
    let document_xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Title</w:t></w:r></w:p>
<w:p><w:r><w:t>Hello world</w:t></w:r></w:p>
<w:p/>
<w:tbl>
<w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>
</w:body>
</w:document>"#;
    let bytes = build_zip(&[("word/document.xml", document_xml)]);
    let parser = easyrag_pipeline::DocxParser;
    let text = parser.parse_bytes(&bytes, "d.docx").await.unwrap();
    assert_eq!("## Title\nHello world\nA | B\n1 | 2\n\n", text);
}

#[tokio::test]
async fn pptx_parser_extracts_slide_text_with_headers() {
    let slide = |text: &str| {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree><p:sp><p:txBody>
<a:p><a:r><a:t>{text}</a:t></a:r></a:p>
<a:p><a:r><a:t></a:t></a:r></a:p>
</p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"#
        )
    };
    let s1 = slide("First");
    let s2 = slide("Second");
    let bytes = build_zip(&[
        ("ppt/slides/slide1.xml", &s1),
        ("ppt/slides/slide2.xml", &s2),
    ]);
    let parser = easyrag_pipeline::PptxParser;
    let text = parser.parse_bytes(&bytes, "d.pptx").await.unwrap();
    assert_eq!("## Slide 1\nFirst\n\n## Slide 2\nSecond\n\n", text);
}

#[tokio::test]
async fn xlsx_parser_formats_rows_like_poi() {
    let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#;
    let rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#;
    let workbook = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets>
</workbook>"#;
    let workbook_rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#;
    let sheet = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c><c r="B1"><v>42</v></c></row>
<row r="2"><c r="A2" t="inlineStr"><is><t>x</t></is></c><c r="B2"><v>1.5</v></c><c r="C2" t="str"><v>true</v></c></row>
</sheetData>
</worksheet>"#;
    let bytes = build_zip(&[
        ("[Content_Types].xml", content_types),
        ("_rels/.rels", rels),
        ("xl/workbook.xml", workbook),
        ("xl/_rels/workbook.xml.rels", workbook_rels),
        ("xl/worksheets/sheet1.xml", sheet),
    ]);
    let parser = easyrag_pipeline::XlsxParser;
    let text = parser.parse_bytes(&bytes, "d.xlsx").await.unwrap();
    assert_eq!("## Sheet: Data\nName | 42.0 | \nx | 1.5 | true\n\n", text);
}

#[tokio::test]
async fn parse_file_reads_from_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("note.md");
    std::fs::write(&path, "# Hello").unwrap();
    let registry = TextParserRegistry::default();
    let parser = registry.get_parser_for_file(&path).unwrap();
    let text = parser.parse_file(&path).await.unwrap();
    assert_eq!("# Hello", text);
}

#[tokio::test]
async fn invalid_docx_returns_error() {
    let parser = easyrag_pipeline::DocxParser;
    let result = parser.parse_bytes(b"not a zip", "bad.docx").await;
    assert!(result.is_err());
}
