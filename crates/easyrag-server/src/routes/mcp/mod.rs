//! MCP server mount: a stateless Streamable HTTP endpoint.
//!
//! Compiles only under the `mcp` feature. The service is attached to the shared
//! axum router at `config.mcp_mount_path` (default `/api/mcp`) so it inherits
//! the same CORS + JWT middleware chain as the REST routes. Because the path is
//! under `/api/`, the auth filter chain enforces JWT Bearer exactly as it does
//! for the REST surface — no bespoke auth code.
//!
//! The endpoint is fully stateless: [`NeverSessionManager`] rejects session
//! operations and `legacy_session_mode` is off, so 2026-07-28 clients are served
//! request-by-request with no `Mcp-Session-Id` — any instance can answer any
//! request (LB / serverless friendly).

pub mod handler;

use std::sync::Arc;

use axum::Router;
use rmcp::transport::streamable_http_server::session::never::NeverSessionManager;
use rmcp::transport::streamable_http_server::tower::StreamableHttpService;
use rmcp::transport::streamable_http_server::StreamableHttpServerConfig;

use crate::state::AppServices;

use handler::EasyRagMcp;

/// Build the MCP sub-router over the shared services. Returns a
/// `Router<Arc<AppServices>>` so it can be `.merge`d into the main application
/// router before the CORS/JWT layers are applied.
pub fn router(services: Arc<AppServices>) -> Router<Arc<AppServices>> {
    let mount_path = services.config.mcp_mount_path.clone();

    // Stateless configuration:
    // - `legacy_session_mode(false)` + `NeverSessionManager` => per-request,
    //   no session header.
    // - `json_response(true)` => plain application/json replies for the
    //   request/response tools easyai uses (no SSE needed).
    // - `disable_allowed_hosts()` => the DNS-rebinding guard defaults to
    //   loopback only, which would reject external `Host` headers in K8s /
    //   behind a proxy. The endpoint's real gate is the JWT middleware.
    // - Origin validation stays disabled unless operators pin browser origins
    //   via `mcp_allowed_origins` (server-to-server clients send no `Origin`).
    let mut config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .disable_allowed_hosts()
        .with_max_request_body_bytes(handler::MCP_MAX_BODY_BYTES);
    if !services.config.mcp_allowed_origins.is_empty() {
        config = config.with_allowed_origins(services.config.mcp_allowed_origins.clone());
    }

    let factory_services = Arc::clone(&services);
    let service: StreamableHttpService<EasyRagMcp, NeverSessionManager> =
        StreamableHttpService::new(
            move || Ok(EasyRagMcp::new(Arc::clone(&factory_services))),
            Arc::new(NeverSessionManager::default()),
            config,
        );

    Router::new().nest_service(mount_path.as_str(), service)
}
