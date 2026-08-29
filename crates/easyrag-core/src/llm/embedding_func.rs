//! Embedding function abstraction.
//!
//! Wraps an embedding function with dimension validation and configuration.

use async_trait::async_trait;
use std::sync::Arc;

/// Context for distinguishing between query and document embeddings.
/// Allows wrapped functions to apply different processing based on context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingContext {
    Query,
    Document,
}

/// Core embedding function interface.
/// Takes a list of texts and returns a list of float vectors (embeddings).
#[async_trait]
pub trait EmbeddingFunction: Send + Sync {
    async fn embed(
        &self,
        texts: Vec<String>,
        context: Option<EmbeddingContext>,
    ) -> crate::Result<Vec<Vec<f32>>>;
}

/// Embedding function wrapper with dimension validation.
///
/// - `embedding_dim`: expected dimension of the embeddings (for dimension
///   checking and workspace data isolation)
/// - `func`: the actual embedding function to wrap
/// - `max_token_size`: enable embedding token limit checking for description
///   summarization
/// - `send_dimensions`: whether to inject embedding_dim argument to the
///   underlying function
/// - `model_name`: model name for workspace data isolation in vector DB
/// - `supports_asymmetric`: whether the underlying function supports the
///   context parameter
pub struct EmbeddingFunc {
    pub embedding_dim: usize,
    func: Arc<dyn EmbeddingFunction>,
    pub max_token_size: Option<usize>,
    pub send_dimensions: bool,
    pub model_name: Option<String>,
    pub supports_asymmetric: bool,
}

impl std::fmt::Debug for EmbeddingFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingFunc")
            .field("embedding_dim", &self.embedding_dim)
            .field("max_token_size", &self.max_token_size)
            .field("send_dimensions", &self.send_dimensions)
            .field("model_name", &self.model_name)
            .field("supports_asymmetric", &self.supports_asymmetric)
            .finish_non_exhaustive()
    }
}

impl EmbeddingFunc {
    pub fn new(embedding_dim: usize, func: Arc<dyn EmbeddingFunction>) -> Self {
        Self {
            embedding_dim,
            func,
            max_token_size: None,
            send_dimensions: false,
            model_name: None,
            supports_asymmetric: false,
        }
    }

    pub fn with_max_token_size(mut self, max_token_size: usize) -> Self {
        self.max_token_size = Some(max_token_size);
        self
    }

    pub fn with_send_dimensions(mut self, send_dimensions: bool) -> Self {
        self.send_dimensions = send_dimensions;
        self
    }

    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    pub fn with_supports_asymmetric(mut self, supports_asymmetric: bool) -> Self {
        self.supports_asymmetric = supports_asymmetric;
        self
    }

    /// Invoke the embedding function with dimension validation.
    pub async fn invoke(
        &self,
        texts: Vec<String>,
        context: Option<EmbeddingContext>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let effective_context = if self.supports_asymmetric { context } else { None };

        let result = self.func.embed(texts.clone(), effective_context).await?;

        // Validate embedding dimensions
        for (index, embedding) in result.iter().enumerate() {
            if embedding.len() != self.embedding_dim {
                return Err(crate::CoreError::InvalidArgument(format!(
                    "Embedding dimension mismatch at index {}: expected {} but got {}",
                    index,
                    self.embedding_dim,
                    embedding.len()
                )));
            }
        }

        // Verify vector count matches input text count
        if result.len() != texts.len() {
            return Err(crate::CoreError::InvalidArgument(format!(
                "Vector count mismatch: expected {} vectors but got {} vectors",
                texts.len(),
                result.len()
            )));
        }

        Ok(result)
    }

    /// Convenience call without context.
    pub async fn invoke_simple(&self, texts: Vec<String>) -> crate::Result<Vec<Vec<f32>>> {
        self.invoke(texts, None).await
    }
}

#[cfg(test)]
mod tests {
    // Ported from EmbeddingFuncTest.kt (7 cases).
    use super::*;
    use std::sync::Mutex;

    /// Test embedding function: vector = [text.len+i for i in 0..dim].
    struct LengthBasedFunc {
        dim: usize,
    }

    #[async_trait]
    impl EmbeddingFunction for LengthBasedFunc {
        async fn embed(
            &self,
            texts: Vec<String>,
            _context: Option<EmbeddingContext>,
        ) -> crate::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    (0..self.dim)
                        .map(|i| text.chars().count() as f32 + i as f32)
                        .collect()
                })
                .collect())
        }
    }

    fn create_embedding_func(dim: usize, supports_asymmetric: bool) -> EmbeddingFunc {
        EmbeddingFunc::new(dim, Arc::new(LengthBasedFunc { dim }))
            .with_supports_asymmetric(supports_asymmetric)
    }

    #[tokio::test]
    async fn invoke_valid_dimension_returns_embeddings() {
        let func = create_embedding_func(3, false);
        let result = func
            .invoke_simple(vec!["hello".to_string(), "world".to_string()])
            .await
            .unwrap();
        assert_eq!(2, result.len());
        assert_eq!(3, result[0].len());
        assert_eq!(3, result[1].len());
    }

    #[tokio::test]
    async fn invoke_dimension_mismatch_throws_exception() {
        struct DimThreeFunc;
        #[async_trait]
        impl EmbeddingFunction for DimThreeFunc {
            async fn embed(
                &self,
                texts: Vec<String>,
                _context: Option<EmbeddingContext>,
            ) -> crate::Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|_| vec![0f32; 3]).collect())
            }
        }
        let func = EmbeddingFunc::new(5, Arc::new(DimThreeFunc)); // expects 5, returns 3
        let result = func.invoke_simple(vec!["test".to_string()]).await;
        assert!(matches!(
            result,
            Err(crate::CoreError::InvalidArgument(_))
        ));
    }

    #[tokio::test]
    async fn invoke_count_mismatch_throws_exception() {
        struct SingleVectorFunc;
        #[async_trait]
        impl EmbeddingFunction for SingleVectorFunc {
            async fn embed(
                &self,
                _texts: Vec<String>,
                _context: Option<EmbeddingContext>,
            ) -> crate::Result<Vec<Vec<f32>>> {
                Ok(vec![vec![0f32; 3]]) // always returns 1 vector
            }
        }
        let func = EmbeddingFunc::new(3, Arc::new(SingleVectorFunc));
        let result = func
            .invoke_simple(vec!["text1".to_string(), "text2".to_string()])
            .await;
        assert!(matches!(
            result,
            Err(crate::CoreError::InvalidArgument(_))
        ));
    }

    #[tokio::test]
    async fn invoke_empty_input_returns_empty_result() {
        let func = create_embedding_func(3, false);
        let result = func.invoke_simple(Vec::new()).await.unwrap();
        assert_eq!(0, result.len());
    }

    #[tokio::test]
    async fn invoke_convenience_call_works() {
        let func = create_embedding_func(2, false);
        let result = func.invoke_simple(vec!["test".to_string()]).await.unwrap();
        assert_eq!(1, result.len());
        assert_eq!(2, result[0].len());
    }

    #[tokio::test]
    async fn invoke_asymmetric_not_supported_context_ignored() {
        struct ContextCapturingFunc {
            captured: Arc<Mutex<Option<Option<EmbeddingContext>>>>,
        }
        #[async_trait]
        impl EmbeddingFunction for ContextCapturingFunc {
            async fn embed(
                &self,
                texts: Vec<String>,
                context: Option<EmbeddingContext>,
            ) -> crate::Result<Vec<Vec<f32>>> {
                *self.captured.lock().unwrap() = Some(context);
                Ok(texts.iter().map(|_| vec![0f32; 2]).collect())
            }
        }
        // Initial captured value DOCUMENT — must be replaced with None.
        let captured = Arc::new(Mutex::new(Some(Some(EmbeddingContext::Document))));
        let func = EmbeddingFunc::new(
            2,
            Arc::new(ContextCapturingFunc {
                captured: captured.clone(),
            }),
        )
        .with_supports_asymmetric(false);
        func.invoke(vec!["test".to_string()], Some(EmbeddingContext::Query))
            .await
            .unwrap();
        // Context should be None because supports_asymmetric = false
        assert_eq!(Some(None), *captured.lock().unwrap());
    }

    #[tokio::test]
    async fn invoke_asymmetric_supported_context_passed() {
        struct ContextCapturingFunc {
            captured: Arc<Mutex<Option<Option<EmbeddingContext>>>>,
        }
        #[async_trait]
        impl EmbeddingFunction for ContextCapturingFunc {
            async fn embed(
                &self,
                texts: Vec<String>,
                context: Option<EmbeddingContext>,
            ) -> crate::Result<Vec<Vec<f32>>> {
                *self.captured.lock().unwrap() = Some(context);
                Ok(texts.iter().map(|_| vec![0f32; 2]).collect())
            }
        }
        let captured = Arc::new(Mutex::new(None::<Option<EmbeddingContext>>));
        let func = EmbeddingFunc::new(
            2,
            Arc::new(ContextCapturingFunc {
                captured: captured.clone(),
            }),
        )
        .with_supports_asymmetric(true);
        func.invoke(vec!["test".to_string()], Some(EmbeddingContext::Query))
            .await
            .unwrap();
        assert_eq!(Some(Some(EmbeddingContext::Query)), *captured.lock().unwrap());
    }
}
