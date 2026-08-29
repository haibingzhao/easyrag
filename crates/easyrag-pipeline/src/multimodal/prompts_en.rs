//! English prompt templates for multimodal content processing

use std::collections::HashMap;

pub const IMAGE_ANALYSIS_SYSTEM: &str =
    "You are an expert image analyst. Provide detailed, accurate descriptions.";

pub const IMAGE_ANALYSIS_FALLBACK_SYSTEM: &str =
    "You are an expert image analyst. Provide detailed analysis based on available information.";

pub const TABLE_ANALYSIS_SYSTEM: &str =
    "You are an expert data analyst. Provide detailed table analysis with specific insights.";

pub const EQUATION_ANALYSIS_SYSTEM: &str =
    "You are an expert mathematician. Provide detailed mathematical analysis.";

pub const GENERIC_ANALYSIS_SYSTEM: &str =
    "You are an expert content analyst specializing in {content_type} content.";

pub const VISION_PROMPT: &str = r#"Please analyze this image in detail and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Always use specific names instead of pronouns",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content and its significance (max 100 words)"
    }
}

Additional context:
- Section Path: {section_path}
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Focus on providing accurate, detailed visual analysis that would be useful for knowledge retrieval.
Use a semantic entity_name; do not return file names or figure numbers such as figure_30_1 unless they are the actual title."#;

pub const VISION_PROMPT_WITH_CONTEXT: &str = r#"Please analyze this image in detail, considering the surrounding context. Provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements and how they relate to the surrounding context
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Reference connections to the surrounding content when relevant
    - Always use specific names instead of pronouns",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content, its significance, and relationship to surrounding content (max 100 words)"
    }
}

Context from surrounding content:
{context}

Document structure:
- Section Path: {section_path}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Focus on providing accurate, detailed visual analysis that incorporates the context and would be useful for knowledge retrieval.
Use a semantic entity_name; do not return file names or figure numbers such as figure_30_1 unless they are the actual title."#;

pub const TEXT_PROMPT: &str = r#"Based on the following image information, provide analysis:

Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

{vision_prompt}"#;

pub const TABLE_PROMPT: &str = r#"Please analyze this table content and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    Always use specific names and values instead of general references.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose and key findings (max 100 words)"
    }
}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data."#;

pub const TABLE_PROMPT_WITH_CONTEXT: &str = r#"Please analyze this table content considering the surrounding context, and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented in relation to surrounding context
    - How the table supports or illustrates concepts from the surrounding content
    Always use specific names and values instead of general references.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)"
    }
}

Context from surrounding content:
{context}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data in the context of the surrounding content."#;

pub const EQUATION_PROMPT: &str = r#"Please analyze this mathematical equation and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions
    - Mathematical operations and functions used
    - Application domain and context
    - Physical or theoretical significance
    - Relationship to other mathematical concepts
    - Practical applications or use cases
    Always use specific mathematical terminology.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose and significance (max 100 words)"
    }
}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance."#;

pub const EQUATION_PROMPT_WITH_CONTEXT: &str = r#"Please analyze this mathematical equation considering the surrounding context, and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions in the context of surrounding content
    - Mathematical operations and functions used
    - Application domain and context based on surrounding material
    - Physical or theoretical significance
    - Relationship to other mathematical concepts mentioned in the context
    - Practical applications or use cases
    - How the equation relates to the broader discussion or framework
    Always use specific mathematical terminology.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose, significance, and role in the surrounding context (max 100 words)"
    }
}

Context from surrounding content:
{context}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance within the broader context."#;

pub const GENERIC_PROMPT: &str = r#"Please analyze this {content_type} content and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose and key points (max 100 words)"
    }
}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval."#;

pub const GENERIC_PROMPT_WITH_CONTEXT: &str = r#"Please analyze this {content_type} content considering the surrounding context, and provide a JSON response with the following structure:

{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance in relation to surrounding content
    - How this content connects to or supports the broader discussion
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose, key points, and relationship to surrounding context (max 100 words)"
    }
}

Context from surrounding content:
{context}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval and understanding the content's role in the broader context."#;

pub const IMAGE_CHUNK: &str = r#"
Image Content Analysis:
- Section Path: {section_path}
- Neighbor Text: {neighbor_text}
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}"#;

pub const TABLE_CHUNK: &str = r#"Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"#;

pub const EQUATION_CHUNK: &str = r#"Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}"#;

pub const GENERIC_CHUNK: &str = r#"{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"#;

pub const QUERY_IMAGE_DESCRIPTION: &str =
    "Please briefly describe the main content, key elements, and important information in this image.";

pub const QUERY_IMAGE_ANALYST_SYSTEM: &str =
    "You are a professional image analyst who can accurately describe image content.";

pub const QUERY_TABLE_ANALYSIS: &str = r#"Please analyze the main content, structure, and key information of the following table data:

Table data:
{table_data}

Table caption: {table_caption}

Please briefly summarize the main content, data characteristics, and important findings of the table."#;

pub const QUERY_TABLE_ANALYST_SYSTEM: &str =
    "You are a professional data analyst who can accurately analyze table data.";

pub const QUERY_EQUATION_ANALYSIS: &str = r#"Please explain the meaning and purpose of the following mathematical formula:

LaTeX formula: {latex}
Formula caption: {equation_caption}

Please briefly explain the mathematical meaning, application scenarios, and importance of this formula."#;

pub const QUERY_EQUATION_ANALYST_SYSTEM: &str =
    "You are a mathematics expert who can clearly explain mathematical formulas.";

pub const QUERY_GENERIC_ANALYSIS: &str = r#"Please analyze the following {content_type} type content and extract its main information and key features:

Content: {content_str}

Please briefly summarize the main characteristics and important information of this content."#;

pub const QUERY_GENERIC_ANALYST_SYSTEM: &str =
    "You are a professional content analyst who can accurately analyze {content_type} type content.";

pub const QUERY_ENHANCEMENT_SUFFIX: &str =
    "\n\nPlease provide a comprehensive answer based on the user query and the provided multimodal content information.";

/// Get all prompts as a map (for PromptManager language switching).
pub fn as_map() -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert("IMAGE_ANALYSIS_SYSTEM".to_string(), IMAGE_ANALYSIS_SYSTEM.to_string());
    map.insert("IMAGE_ANALYSIS_FALLBACK_SYSTEM".to_string(), IMAGE_ANALYSIS_FALLBACK_SYSTEM.to_string());
    map.insert("TABLE_ANALYSIS_SYSTEM".to_string(), TABLE_ANALYSIS_SYSTEM.to_string());
    map.insert("EQUATION_ANALYSIS_SYSTEM".to_string(), EQUATION_ANALYSIS_SYSTEM.to_string());
    map.insert("GENERIC_ANALYSIS_SYSTEM".to_string(), GENERIC_ANALYSIS_SYSTEM.to_string());
    map.insert("vision_prompt".to_string(), VISION_PROMPT.to_string());
    map.insert("vision_prompt_with_context".to_string(), VISION_PROMPT_WITH_CONTEXT.to_string());
    map.insert("text_prompt".to_string(), TEXT_PROMPT.to_string());
    map.insert("table_prompt".to_string(), TABLE_PROMPT.to_string());
    map.insert("table_prompt_with_context".to_string(), TABLE_PROMPT_WITH_CONTEXT.to_string());
    map.insert("equation_prompt".to_string(), EQUATION_PROMPT.to_string());
    map.insert("equation_prompt_with_context".to_string(), EQUATION_PROMPT_WITH_CONTEXT.to_string());
    map.insert("generic_prompt".to_string(), GENERIC_PROMPT.to_string());
    map.insert("generic_prompt_with_context".to_string(), GENERIC_PROMPT_WITH_CONTEXT.to_string());
    map.insert("image_chunk".to_string(), IMAGE_CHUNK.to_string());
    map.insert("table_chunk".to_string(), TABLE_CHUNK.to_string());
    map.insert("equation_chunk".to_string(), EQUATION_CHUNK.to_string());
    map.insert("generic_chunk".to_string(), GENERIC_CHUNK.to_string());
    map.insert("QUERY_IMAGE_DESCRIPTION".to_string(), QUERY_IMAGE_DESCRIPTION.to_string());
    map.insert("QUERY_IMAGE_ANALYST_SYSTEM".to_string(), QUERY_IMAGE_ANALYST_SYSTEM.to_string());
    map.insert("QUERY_TABLE_ANALYSIS".to_string(), QUERY_TABLE_ANALYSIS.to_string());
    map.insert("QUERY_TABLE_ANALYST_SYSTEM".to_string(), QUERY_TABLE_ANALYST_SYSTEM.to_string());
    map.insert("QUERY_EQUATION_ANALYSIS".to_string(), QUERY_EQUATION_ANALYSIS.to_string());
    map.insert("QUERY_EQUATION_ANALYST_SYSTEM".to_string(), QUERY_EQUATION_ANALYST_SYSTEM.to_string());
    map.insert("QUERY_GENERIC_ANALYSIS".to_string(), QUERY_GENERIC_ANALYSIS.to_string());
    map.insert("QUERY_GENERIC_ANALYST_SYSTEM".to_string(), QUERY_GENERIC_ANALYST_SYSTEM.to_string());
    map.insert("QUERY_ENHANCEMENT_SUFFIX".to_string(), QUERY_ENHANCEMENT_SUFFIX.to_string());
    map
}
