//! easyrag-server crate: axum HTTP/SSE layer.

pub mod auth;
pub mod config;
pub mod error;
pub mod routes;
pub mod scan_stats;
pub mod service;
pub mod state;
pub mod util;

use std::sync::Arc;

use axum::http::HeaderValue;
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

pub use config::{ServerConfig, DEFAULT_JWT_SECRET};
pub use error::{AppError, AppResult};
pub use scan_stats::AutoScanStats;
pub use state::{AppServices, AppState};

/// Assemble the full router from shared services ( controller beans +
/// security filter chain + CORS configuration source).
pub fn build_router(services: Arc<AppServices>) -> Router {
    let mut app = Router::new()
        .merge(routes::health::router())
        .merge(routes::auth::router())
        .merge(routes::ollama::router())
        .merge(routes::query::router())
        .merge(routes::documents::router())
        .merge(routes::graph::router())
        .merge(routes::evaluation::router())
        .merge(routes::tenant_config::router());
    // MultimodalController bean is conditional on RAGAnything.
    if services.rag_anything.is_some() {
        app = app.merge(routes::multimodal::router());
    }

    app = app.layer(cors_layer(&services.config.cors_origins));

    // Security filter chain: JWT enforcement only when auth is enabled.
    if let Some(jwt) = &services.jwt {
        app = app.layer(axum::middleware::from_fn_with_state(
            Arc::clone(jwt),
            auth::auth_middleware,
        ));
    }

    app.with_state(services)
}

/// CORS layer matching: configured origins,
/// methods GET/POST/PUT/DELETE/OPTIONS, all headers.
fn cors_layer(origins: &[String]) -> CorsLayer {
    let methods = [
        axum::http::Method::GET,
        axum::http::Method::POST,
        axum::http::Method::PUT,
        axum::http::Method::DELETE,
        axum::http::Method::OPTIONS,
    ];
    if origins.iter().any(|o| o == "*") {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(methods)
            .allow_headers(Any)
    } else {
        let parsed: Vec<HeaderValue> = origins.iter().filter_map(|o| o.parse().ok()).collect();
        CorsLayer::new()
            .allow_origin(parsed)
            .allow_methods(methods)
            .allow_headers(Any)
    }
}