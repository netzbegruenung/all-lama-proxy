use axum::{
    body::Bytes,
    http::{HeaderMap, Method, StatusCode},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    io::{self, Write},
    net::IpAddr,
    sync::{Arc, Mutex, RwLock},
};
use tokio::sync::{Notify, mpsc};
use tracing::{info, warn};

use crate::auth::UserRegistry;
use crate::utils::LockExt;

const BLOCKED_FILE: &str = "blocked_items.json";

/// Log entry for the in-TUI log buffer
pub struct LogEntry {
    pub level: String,
    pub message: String,
    pub timestamp: i64,
}

/// Thread-safe circular buffer for log messages
#[derive(Clone)]
pub struct LogBuffer {
    lines: Arc<RwLock<VecDeque<LogEntry>>>,
    max_lines: usize,
}

impl LogBuffer {
    pub fn new(max_lines: usize) -> Self {
        Self {
            lines: Arc::new(RwLock::new(VecDeque::with_capacity(max_lines))),
            max_lines,
        }
    }

    pub fn append(&self, level: &str, message: String) {
        if let Ok(mut lines) = self.lines.write() {
            lines.push_back(LogEntry {
                level: level.to_string(),
                message,
                timestamp: Utc::now().timestamp(),
            });
            while lines.len() > self.max_lines {
                lines.pop_front();
            }
        }
    }

    pub fn get_last_n(&self, n: usize) -> Vec<(String, i64, String)> {
        if let Ok(lines) = self.lines.read() {
            let total = lines.len();
            lines
                .iter()
                .skip(total.saturating_sub(n))
                .map(|entry| (entry.level.clone(), entry.timestamp, entry.message.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }
}

/// Writer that captures tracing output into a LogBuffer
pub struct LogBufferWriter {
    buffer: LogBuffer,
    current_line: String,
    out: std::io::Stdout,
}

impl LogBufferWriter {
    pub fn new(buffer: LogBuffer) -> Self {
        Self {
            buffer,
            current_line: String::new(),
            out: std::io::stdout(),
        }
    }

    fn parse_and_store(&mut self, line: &str) {
        // Parse format: "LEVEL: message" or extract level from line
        let (level, message) = if line.starts_with("DEBUG") {
            ("DEBUG", line[6..].trim())
        } else if line.starts_with("INFO") {
            ("INFO", line[5..].trim())
        } else if line.starts_with("WARN") {
            ("WARN", line[5..].trim())
        } else if line.starts_with("ERROR") {
            ("ERROR", line[6..].trim())
        } else {
            ("INFO", line)
        };

        self.buffer.append(level, message.to_string());
    }
}

impl Write for LogBufferWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let _ = self.out.write_all(buf);
        let _ = self.out.flush();

        if let Ok(line) = std::str::from_utf8(buf) {
            self.current_line.push_str(line);

            // Process complete lines (ending with newline)
            while let Some(pos) = self.current_line.find('\n') {
                let complete_line = self.current_line[..pos].trim().to_string();
                if !complete_line.is_empty() {
                    self.parse_and_store(&complete_line);
                }
                self.current_line = self.current_line[pos + 1..].to_string();
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let _ = self.out.flush();

        // Process any remaining line on flush
        if !self.current_line.is_empty() {
            let complete_line = self.current_line.trim().to_string();
            if !complete_line.is_empty() {
                self.parse_and_store(&complete_line);
            }
            self.current_line.clear();
        }
        Ok(())
    }
}

impl Drop for LogBufferWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// MakeWriter implementation for LogBufferWriter
impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBufferWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        LogBufferWriter {
            buffer: self.buffer.clone(),
            current_line: String::new(),
            out: std::io::stdout(),
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct BlockedConfig {
    ips: HashSet<IpAddr>,
    users: HashSet<String>,
}

/// OpenAI-compatible model response structure
#[derive(Serialize, Deserialize, Clone)]
pub struct OpenAIModel {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct OpenAIModelsList {
    pub object: &'static str,
    pub data: Vec<OpenAIModel>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

// Public-facing model info with public_name substitution
#[derive(Serialize, Clone)]
pub struct PublicModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
}

// Cached tags response
#[derive(Clone, Serialize)]
pub struct CachedTags {
    pub models: Vec<PublicModelInfo>,
}

/// Loaded from models.yaml - explicit model-to-backend mapping
#[derive(Deserialize, Clone)]
pub struct ModelConfig {
    pub models: Vec<ParsedModel>,
}

/// Validate that a backend URL is safe and reachable for proxying.
///
/// Enforces:
///   - scheme is `http` or `https` (rejects `file://`, `gopher://`, etc.)
///   - a non-empty host is present (rejects bare `http:///path`)
///   - no embedded userinfo/credentials (rejects `http://user:pass@host/`)
///   - port, if present, is a valid non-zero 16-bit port
///   - the URL parses as a valid absolute URL
///
/// This is the primary URL-injection mitigation. The outbound destination is
/// config-driven (not client-pickable), but an operator-controlled
/// `models.yaml` containing `file:///etc/shadow` would otherwise be handed
/// straight to reqwest.
///
/// There is deliberately **no IP allowlist or denylist**: loopback, RFC1918
/// ranges and the cloud-metadata address (`http://169.254.169.254/`) are all
/// accepted, because pointing at loopback and private hosts is the normal
/// deployment. Scheme restriction is the only network-level control here — see
/// the `documents_no_ip_allowlist` test.
///
/// Note: the base URL does *not* have to end with `/`, and a path prefix
/// (`http://host/ollama`, e.g. Ollama behind a reverse proxy) is accepted and
/// preserved when the request path is appended — see `backend_request_url`.
pub fn validate_backend_url(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let u = url.trim();
    if u.is_empty() {
        return Err("backend URL is empty".into());
    }

    // Split out the scheme manually (avoid pulling in a URL parser just for
    // this; we only need http/https).
    let lowered = u.to_ascii_lowercase();
    let rest = if let Some(r) = lowered.strip_prefix("http://") {
        r
    } else if let Some(r) = lowered.strip_prefix("https://") {
        r
    } else {
        return Err(format!("backend URL must start with http:// or https:// (got: {u})").into());
    };

    // The authority part is up to the first '/', '?' or '#'.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.is_empty() {
        return Err(format!(
            "backend URL '{}' is missing a host (e.g. http://127.0.0.1:11434/)",
            u
        )
        .into());
    }

    // Reject embedded credentials: http://user:pass@host/
    if authority.contains('@') {
        return Err(format!("backend URL '{}' must not embed userinfo (@)", u).into());
    }

    // Split authority into host[:port]. An IPv6 literal is bracketed
    // (`[::1]`), so its port — if any — is the ':' *after* the closing
    // bracket; a bare `rfind(':')` would land inside the address and treat
    // the trailing `1]` of `[::1]` as the port.
    let (host, port_str) = if let Some(close) = authority.rfind(']') {
        match authority[close + 1..].strip_prefix(':') {
            Some(port) => (&authority[..=close], port),
            None => (authority, ""),
        }
    } else {
        match authority.rfind(':') {
            Some(idx) if !authority[idx + 1..].contains(':') => {
                (&authority[..idx], &authority[idx + 1..])
            }
            _ => (authority, ""),
        }
    };

    if host.is_empty() {
        return Err(format!("backend URL '{}' has an empty host", u).into());
    }

    if !port_str.is_empty() {
        match port_str.parse::<u16>() {
            Ok(0) => {
                return Err(format!("backend URL '{}' has invalid port 0", u).into());
            }
            Err(_) => {
                return Err(format!("backend URL '{}' has invalid port '{}'", u, port_str).into());
            }
            Ok(_) => {}
        }
    }

    // Final sanity: the base URL should parse as a valid absolute URL.
    reqwest::Url::parse(u)
        .map_err(|e| format!("backend URL '{}' is not a valid HTTP(S) URL: {e}", u))?;

    Ok(())
}

/// Individual model configuration with explicit backends and aliases
#[derive(Deserialize, Clone)]
pub struct ParsedModel {
    /// Real model name (required, used for routing)
    pub name: String,

    /// Display name for /api/tags and TUI (optional, defaults to name)
    #[serde(default)]
    pub public_name: Option<String>,

    /// Backend URLs that serve this model
    pub backends: Vec<String>,

    /// Alias names that resolve to this model
    #[serde(default)]
    pub aliases: Vec<String>,

    /// Maximum concurrent requests per backend for this model (default: 1)
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,

    /// Whether to enable keep-alive for this model (default: true)
    /// Set to false for embedding models that don't support keep-alive
    #[serde(default = "default_keep_alive")]
    pub keep_alive: bool,
}

fn default_keep_alive() -> bool {
    true
}

fn default_max_concurrent_requests() -> usize {
    1
}

/// Append `:latest` if the model name has no explicit tag.
pub fn normalize_model_tag(name: &str) -> String {
    if name.contains(':') {
        name.to_string()
    } else {
        format!("{}:latest", name)
    }
}

/// Compare two model names treating a missing tag as `:latest`,
/// so `foo` and `foo:latest` refer to the same model.
pub fn model_names_match(a: &str, b: &str) -> bool {
    a == b || normalize_model_tag(a) == normalize_model_tag(b)
}

/// Runtime backend status
#[derive(Clone)]
pub struct BackendStatus {
    pub url: String,
    pub active_requests: usize,
    pub processed_count: usize,
    pub is_online: bool,
    pub active_models: HashMap<String, usize>,
    pub processed_models: HashMap<String, usize>,
    /// Models this backend is configured to serve (from config)
    pub configured_models: Vec<String>,
    /// Per-model verification status (true = available, false = not available)
    pub model_status: Arc<RwLock<HashMap<String, bool>>>,
}

impl BackendStatus {
    pub fn new(url: String) -> Self {
        Self {
            url,
            active_requests: 0,
            processed_count: 0,
            is_online: true,
            active_models: HashMap::new(),
            processed_models: HashMap::new(),
            configured_models: Vec::new(),
            model_status: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if this backend can serve a specific model
    pub fn can_serve_model(&self, model_name: &str) -> bool {
        // Check if any configured model matches exactly
        let has_matching_config = self
            .configured_models
            .iter()
            .any(|configured| configured == model_name);

        if !has_matching_config {
            return false;
        }

        // Check per-model verification status (exact match)
        self.model_status
            .read()
            .map(|status| status.get(model_name).copied().unwrap_or(true))
            .unwrap_or(true)
    }
}

impl ModelConfig {
    /// Load model configuration from YAML file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: ModelConfig = serde_yaml::from_str(&content)?;

        // Validate that each model has at least one backend
        for model in &config.models {
            if model.backends.is_empty() {
                return Err(format!("Model '{}' has no backends configured", model.name).into());
            }
            // Validate each backend URL: http(s) only, non-empty host, no
            // embedded userinfo, base path terminated with '/'. This is the
            // primary SSRF / URL-injection mitigation.
            for backend in &model.backends {
                validate_backend_url(backend)?;
            }
        }

        Ok(config)
    }

    /// Resolve alias to the real model name as spelled in the config,
    /// returns None if not found. Matching ignores an implicit `:latest` tag on
    /// either side, but the returned name is always the exact config spelling so
    /// it matches `BackendStatus::configured_models`.
    pub fn resolve_alias(&self, model_name: &str) -> Option<String> {
        // Check explicit aliases first
        for model in &self.models {
            if model
                .aliases
                .iter()
                .any(|alias| model_names_match(alias, model_name))
            {
                return Some(model.name.clone());
            }
        }

        // Check public_name as implicit alias
        for model in &self.models {
            if let Some(ref public_name) = model.public_name {
                if model_names_match(public_name, model_name) {
                    return Some(model.name.clone());
                }
            }
        }

        // Check if it's already a real model name
        for model in &self.models {
            if model_names_match(&model.name, model_name) {
                return Some(model.name.clone());
            }
        }

        None
    }

    /// Get ParsedModel by name or alias (including public_name)
    pub fn get_model(&self, model_name: &str) -> Option<&ParsedModel> {
        self.models.iter().find(|m| {
            model_names_match(&m.name, model_name)
                || m.aliases
                    .iter()
                    .any(|alias| model_names_match(alias, model_name))
                || m.public_name
                    .as_ref()
                    .is_some_and(|pn| model_names_match(pn, model_name))
        })
    }

    /// Get all unique backend URLs from all models
    pub fn get_all_backends(&self) -> BTreeSet<String> {
        let mut backends = BTreeSet::new();
        for model in &self.models {
            for backend in &model.backends {
                backends.insert(backend.clone());
            }
        }
        backends
    }

    /// Get models configured for a specific backend
    pub fn get_models_for_backend(&self, backend_url: &str) -> Vec<String> {
        self.models
            .iter()
            .filter(|m| m.backends.contains(&backend_url.to_string()))
            .map(|m| m.name.clone())
            .collect()
    }
}

pub enum ResponsePart {
    Status(StatusCode, HeaderMap),
    Chunk(Bytes),
    Error(reqwest::Error),
    /// A dispatch failure with no `reqwest::Error` behind it (e.g. a backend
    /// URL that could not be composed). Carries the server-side reason; the
    /// client still only sees a generic 502.
    Failed(String),
    ModelNotFound(String),
}

pub struct Task {
    pub method: Method,
    pub path: String,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub responder: mpsc::Sender<ResponsePart>,
    pub resolved_model: Option<String>,
}

pub struct AppState {
    pub queues: Mutex<HashMap<String, VecDeque<Task>>>,
    pub processing_counts: Mutex<HashMap<String, usize>>,
    pub processed_counts: Mutex<HashMap<String, usize>>,
    pub dropped_counts: Mutex<HashMap<String, usize>>,
    pub user_ips: Mutex<HashMap<String, IpAddr>>,
    pub blocked_ips: Mutex<HashSet<IpAddr>>,
    pub blocked_users: Mutex<HashSet<String>>,
    pub vip_user: Mutex<Vec<String>>,
    pub global_counter: Mutex<usize>,
    pub notify: Notify,
    pub backend_freed: Notify,
    pub backends: Mutex<Vec<BackendStatus>>,
    pub last_backend_idx: Mutex<usize>,
    pub timeout: u64,
    /// Shared reqwest client for backend requests
    pub client: reqwest::Client,
    /// User registry loaded from users.yaml
    pub user_registry: Mutex<Arc<UserRegistry>>,
    /// Model configuration: real models with backends and aliases
    pub model_config: Arc<RwLock<ModelConfig>>,
    /// Debug mode: enable verbose logging
    pub debug: bool,
    /// Shared log buffer for in-TUI logging
    pub log_buffer: LogBuffer,
    /// HTTP header to extract real client IP from
    pub ip_header: Option<String>,
    /// Cached /api/tags response with merged models
    pub cached_tags: Arc<RwLock<Option<CachedTags>>>,
    /// Health check interval in seconds
    pub health_check_interval: u64,
}

impl AppState {
    pub fn new(
        model_config_path: String,
        timeout: u64,
        registry: UserRegistry,
        debug: bool,
        log_buffer: LogBuffer,
        ip_header: Option<String>,
        health_check_interval: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (blocked_ips, blocked_users) = Self::load_blocked_items();

        // Load model configuration (required, will crash if missing)
        let model_config = ModelConfig::load(&model_config_path)?;
        info!("Loaded model configuration from {}", model_config_path);

        // Build deduplicated backend list from config
        let backend_urls = model_config.get_all_backends();

        // Create BackendStatus entries with model configuration
        let mut backends: Vec<BackendStatus> = backend_urls
            .into_iter()
            .map(|url| {
                let mut backend = BackendStatus::new(url.clone());
                backend.configured_models = model_config.get_models_for_backend(&url);

                // Initialize model_status to true for all configured models
                // (optimistic: assume available until health check proves otherwise)
                let mut initial_status = HashMap::new();
                for model_name in &backend.configured_models {
                    initial_status.insert(model_name.clone(), true);
                }
                *backend.model_status.write().expect("model_status write") = initial_status;

                backend
            })
            .collect();

        // Sort backends for consistent ordering (optional, for TUI display)
        backends.sort_by(|a, b| a.url.cmp(&b.url));

        info!("{} backends configured from model config", backends.len());
        for backend in &backends {
            info!(
                "  Backend {}: serves {} models",
                backend.url,
                backend.configured_models.len()
            );
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .expect("Failed to create reqwest client");

        Ok(Self {
            queues: Mutex::new(HashMap::new()),
            processing_counts: Mutex::new(HashMap::new()),
            processed_counts: Mutex::new(HashMap::new()),
            dropped_counts: Mutex::new(HashMap::new()),
            user_ips: Mutex::new(HashMap::new()),
            blocked_ips: Mutex::new(blocked_ips),
            blocked_users: Mutex::new(blocked_users),
            vip_user: Mutex::new(registry.get_vip_users()),
            global_counter: Mutex::new(0),
            notify: Notify::new(),
            backend_freed: Notify::new(),
            backends: Mutex::new(backends),
            last_backend_idx: Mutex::new(0),
            timeout,
            client,
            user_registry: Mutex::new(Arc::new(registry)),
            model_config: Arc::new(RwLock::new(model_config)),
            debug,
            log_buffer,
            ip_header,
            cached_tags: Arc::new(RwLock::new(None)),
            health_check_interval,
        })
    }

    /// Reload model configuration (called from SIGHUP handler)
    pub fn reload_model_config(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let new_config = ModelConfig::load(path)?;

        // Update the config
        {
            let mut write_guard = self.model_config.write().expect("model_config write");
            *write_guard = new_config.clone();
        }

        let all_new_backends = new_config.get_all_backends();

        {
            let mut backends = self.backends.lock().lock_unwrap("backends");

            // Remove backends that are no longer in the config
            backends.retain(|b| all_new_backends.contains(&b.url));

            // Update existing backends and identify which ones are already present
            let mut current_urls = HashSet::new();
            for backend in backends.iter_mut() {
                current_urls.insert(backend.url.clone());
                backend.configured_models = new_config.get_models_for_backend(&backend.url);

                let mut new_status = HashMap::new();
                for model_name in &backend.configured_models {
                    let old_status = backend
                        .model_status
                        .read()
                        .expect("model_status read")
                        .get(model_name)
                        .copied();

                    new_status.insert(model_name.clone(), old_status.unwrap_or(true));
                }
                *backend.model_status.write().expect("model_status write") = new_status;
            }

            // Add new backends
            for url in all_new_backends {
                if !current_urls.contains(&url) {
                    let mut backend = BackendStatus::new(url.clone());
                    backend.configured_models = new_config.get_models_for_backend(&url);

                    let mut initial_status = HashMap::new();
                    for model_name in &backend.configured_models {
                        initial_status.insert(model_name.clone(), true);
                    }
                    *backend.model_status.write().expect("model_status write") = initial_status;

                    backends.push(backend);
                }
            }

            backends.sort_by(|a, b| a.url.cmp(&b.url));
        }

        Ok(())
    }

    /// Spawn keep-alive for a single backend's eligible models
    pub fn spawn_keep_alive_for_backend(&self, backend: &BackendStatus) {
        use crate::health::keep_alive_specific_models;

        let config = self.model_config.read().expect("model_config read");
        let models_to_keep: Vec<String> = backend
            .configured_models
            .iter()
            .filter(|model_name| {
                config
                    .get_model(model_name)
                    .map(|m| m.keep_alive)
                    .unwrap_or(true)
            })
            .cloned()
            .collect();
        drop(config);

        if models_to_keep.is_empty() {
            return;
        }

        let backend_clone = backend.clone();
        let client_clone = self.client.clone();
        let timeout = self.timeout;

        tokio::spawn(async move {
            keep_alive_specific_models(&backend_clone, &client_clone, &models_to_keep, timeout)
                .await;
        });
    }

    /// Trigger keep-alive for all online backends
    pub async fn trigger_all_keep_alives(&self) {
        let backends = self.backends.lock().lock_unwrap("backends");

        for backend in backends.iter() {
            if backend.is_online {
                self.spawn_keep_alive_for_backend(backend);
            }
        }
    }

    #[allow(clippy::collapsible_if)]
    fn load_blocked_items() -> (HashSet<IpAddr>, HashSet<String>) {
        if let Ok(content) = fs::read_to_string(BLOCKED_FILE) {
            if let Ok(config) = serde_json::from_str::<BlockedConfig>(&content) {
                return (config.ips, config.users);
            }
        }
        (HashSet::new(), HashSet::new())
    }

    fn save_blocked_items(&self) {
        let config = BlockedConfig {
            ips: self.blocked_ips.lock().lock_unwrap("blocked_ips").clone(),
            users: self
                .blocked_users
                .lock()
                .lock_unwrap("blocked_users")
                .clone(),
        };
        if let Ok(content) = serde_json::to_string_pretty(&config) {
            let _ = fs::write(BLOCKED_FILE, content);
        }
    }

    pub fn block_ip(&self, ip: IpAddr) {
        {
            let mut ips = self.blocked_ips.lock().lock_unwrap("blocked_ips");
            ips.insert(ip);
        }
        self.save_blocked_items();
        warn!("IP blocked: {}", ip);
    }

    pub fn block_user(&self, user_id: String) {
        {
            let mut users = self.blocked_users.lock().lock_unwrap("blocked_users");
            users.insert(user_id.clone());
        }
        self.save_blocked_items();
        warn!("User blocked: {}", user_id);
    }

    pub fn is_ip_blocked(&self, ip: &IpAddr) -> bool {
        self.blocked_ips
            .lock()
            .lock_unwrap("blocked_ips")
            .contains(ip)
    }

    pub fn is_user_blocked(&self, user_id: &str) -> bool {
        self.blocked_users
            .lock()
            .lock_unwrap("blocked_users")
            .contains(user_id)
    }
}

#[cfg(test)]
mod validator_tests {
    use super::validate_backend_url;

    #[test]
    fn accepts_http_ipv4_port() {
        assert!(validate_backend_url("http://10.137.1.1:11434").is_ok());
        assert!(validate_backend_url("http://127.0.0.1:11434/").is_ok());
        assert!(validate_backend_url("http://example.com").is_ok());
        assert!(validate_backend_url("https://example.com:8443/base").is_ok());
        assert!(validate_backend_url("  https://host:9000/  ").is_ok());
    }

    #[test]
    fn rejects_bad_scheme() {
        assert!(validate_backend_url("file:///etc/shadow").is_err());
        assert!(validate_backend_url("gopher://host:25").is_err());
        assert!(validate_backend_url("ftp://host/pub").is_err());
        assert!(validate_backend_url("10.0.0.1:11434").is_err());
        assert!(validate_backend_url("example.com:11434").is_err());
        assert!(validate_backend_url("").is_err());
        assert!(validate_backend_url("   ").is_err());
    }

    #[test]
    fn documents_no_ip_allowlist() {
        // We deliberately do NOT add an IP allowlist: an operator that
        // enumerates a trusted internal backend (e.g. `http://10.0.0.5:11434/`)
        // is the intended use case. We only reject *syntactically* unsafe URLs
        // (bad scheme, no host, embedded creds, bad port). The cloud-metadata
        // URL below is therefore *accepted* by `validate_backend_url` — this
        // is a policy decision for the operator, not a syntax error, and the
        // test exists to pin that down so a future "hardening" does not
        // silently break legitimate private-network backends.
        assert!(validate_backend_url("http://169.254.169.254/").is_ok());
    }

    #[test]
    fn rejects_missing_host() {
        assert!(validate_backend_url("http:///path").is_err());
        assert!(validate_backend_url("http://").is_err());
    }

    #[test]
    fn rejects_embedded_credentials() {
        assert!(validate_backend_url("http://user:pass@host:11434/").is_err());
        assert!(validate_backend_url("https://admin@host/").is_err());
    }

    #[test]
    fn rejects_invalid_port() {
        assert!(validate_backend_url("http://host:99999").is_err());
        assert!(validate_backend_url("http://host:0").is_err());
        assert!(validate_backend_url("http://host:abc").is_err());
    }

    #[test]
    fn accepts_ipv6() {
        // URL::parse and url::Url both handle bracketed IPv6.
        assert!(validate_backend_url("http://[::1]:11434/").is_ok());
    }

    #[test]
    fn accepts_bracketed_ipv6_without_explicit_port() {
        // Regression test: previously `parse_backend_address` returned the raw
        // literal for unbracketed values (correct), but bracketed values fell
        // into a naive `rsplit(':')` that mis-parsed the bracket, leaving a
        // non-numeric "port" and rejecting perfectly valid bracketed IPv6
        // URLs without an explicit port.
        assert!(validate_backend_url("http://[::1]").is_ok());
        assert!(validate_backend_url("http://[2001:db8::ff00:42:8329]").is_ok());
        // Non-literal (hostname) bracketed hosts are rejected outright by the
        // url crate (`http://[example.org]` is not a valid special-form URL) —
        // they are not a supported form, so `is_ok` below is not asserted.
        assert!(validate_backend_url("http://[backend.example.org]").is_err());
    }
}
