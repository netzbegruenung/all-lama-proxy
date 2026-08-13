use axum::{
    body::{Body, Bytes},
    extract::{ConnectInfo, Path, State},
    http::{HeaderMap, Method, StatusCode},
    response::IntoResponse,
};
use chrono::Utc;
use futures_util::StreamExt;
use std::{
    collections::VecDeque,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    net::{IpAddr, SocketAddr},
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};

use crate::appstate::{
    AppState, CachedTags, ModelConfig, OpenAIModel, OpenAIModelsList, ResponsePart, Task,
    model_names_match,
};
use crate::auth::UserRegistry;
use crate::utils::LockExt;

/// Rendezvous (highest-random-weight) hash of a user against a backend URL.
///
/// Keying on `user_id` pins a user to a stable backend so consecutive requests
/// reuse the same kv cache, while distributing distinct users evenly. Uses
/// `DefaultHasher::new()` (fixed seed) so weights are reproducible across
/// process restarts and SIGHUP reloads, and the backend `url` rather than its
/// list index so the mapping survives backends going on/offline.
fn rendezvous_weight(user_id: &str, backend_url: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    user_id.hash(&mut hasher);
    backend_url.hash(&mut hasher);
    hasher.finish()
}

/// Format duration showing only seconds if < 1 minute, otherwise minutes and seconds
fn format_duration_short(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else {
        let mins = secs / 60;
        let remaining_secs = secs % 60;
        format!("{}m {}s", mins, remaining_secs)
    }
}

/// Extract client IP from headers or fallback to connection address
fn extract_client_ip(headers: &HeaderMap, addr: SocketAddr, ip_header: &Option<String>) -> IpAddr {
    if let Some(header_name) = ip_header {
        headers
            .get(header_name)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split(',').next().and_then(|ip| ip.trim().parse().ok()))
            .unwrap_or_else(|| addr.ip())
    } else {
        addr.ip()
    }
}

/// Authenticate request and return user ID if successful
fn authenticate_request(
    headers: &HeaderMap,
    user_registry: &Mutex<Arc<UserRegistry>>,
    ip: IpAddr,
    is_debug: bool,
) -> Option<String> {
    let raw_token = match headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    {
        Some(token) => token,
        None => {
            if is_debug {
                debug!(
                    "Rejected request from {}: missing or malformed Authorization header",
                    ip
                );
            } else {
                warn!(
                    "Rejected request from {}: missing or malformed Authorization header",
                    ip
                );
            }
            return None;
        }
    };

    match user_registry
        .lock()
        .lock_unwrap("user_registry")
        .authenticate(raw_token)
    {
        Some(uid) => {
            if is_debug {
                debug!("Authenticated user: {} from IP: {}", uid, ip);
            }
            Some(uid.to_string())
        }
        None => {
            if is_debug {
                debug!("Rejected request from {}: invalid token", ip);
            } else {
                warn!("Rejected request from {}: invalid token", ip);
            }
            None
        }
    }
}

/// Normalize proxy paths to Ollama's API paths
/// Maps `/chat/completions` → `/v1/chat/completions` for backend compatibility
fn normalize_path(path: &str) -> &str {
    match path {
        "/chat/completions" => "/v1/chat/completions",
        _ => path,
    }
}

/// Paths that carry a `model` field in their JSON body.
const MODEL_PATHS: &[&str] = &[
    "/api/generate",
    "/api/chat",
    "/api/embed",
    "/api/embeddings",
    "/chat/completions",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/images/generations",
];

/// Extract and resolve the `model` field from the request body when the path is model-aware.
/// If the model is an alias, it's replaced with the real name in the body.
/// Returns `None` for non-model endpoints or when the field is absent.
/// Returns `Some(model_name)` even if not in config (will be 503'd later).
///
/// The returned name is the exact spelling from models.yaml, so it can be used
/// as-is for backend matching and per-model bookkeeping.
fn extract_and_resolve_model(
    body: &mut Bytes,
    path: &str,
    config: &ModelConfig,
    debug_enabled: bool,
) -> Option<String> {
    if !MODEL_PATHS.iter().any(|p| path.starts_with(p)) {
        if debug_enabled {
            debug!("Path {} is not a model path", path);
        }
        return None;
    }

    let mut v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let requested_model = v.get("model")?.as_str()?.to_string();

    if debug_enabled {
        debug!("Requested model: {} on path {}", requested_model, path);
    }

    // Resolve alias to real model name
    let real_model = match config.resolve_alias(&requested_model) {
        Some(resolved) => {
            if debug_enabled {
                debug!("Resolved {} -> {}", requested_model, resolved);
            }
            resolved
        }
        None => {
            // Not in config at all - don't modify body, will 503 later
            if debug_enabled {
                debug!("Model {} not found in config", requested_model);
            }
            return Some(requested_model);
        }
    };

    // Update body with real model name
    #[allow(clippy::collapsible_if)]
    if real_model != requested_model {
        if let Some(obj) = v.as_object_mut() {
            obj.insert("model".to_string(), serde_json::json!(real_model));
            *body = Bytes::from(serde_json::to_vec(&v).ok()?);
        }
    }

    Some(real_model)
}

/// Peek at the model name from a request body without modifying it.
/// Used for routing decisions before actually consuming the task.
fn peek_model_from_body(body: &Bytes) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let requested_model = v.get("model")?.as_str()?.to_string();
    Some(requested_model)
}

/// Parse Unix timestamp from Ollama modified_at format or return current time
fn parse_created_timestamp(modified_at: &str) -> i64 {
    // Try to parse ISO 8601 format from Ollama
    // Modified_at typically looks like: "2024-01-15T10:30:00Z" or similar
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(modified_at) {
        dt.timestamp()
    } else if let Ok(dt) = chrono::DateTime::parse_from_str(modified_at, "%Y-%m-%d %H:%M:%S.%f%#z")
    {
        dt.timestamp()
    } else {
        // Fallback to current time if parsing fails
        Utc::now().timestamp()
    }
}

/// Transform cached tags into OpenAI-compatible models list
fn build_models_list(cached_tags: &Option<CachedTags>) -> OpenAIModelsList {
    let models = match cached_tags {
        Some(tags) => tags
            .models
            .iter()
            .map(|model| {
                let created = parse_created_timestamp(&model.modified_at);

                OpenAIModel {
                    id: model.name.clone(),
                    object: "model",
                    created,
                    owned_by: "all-llama-proxy".to_string(),
                }
            })
            .collect(),
        None => vec![],
    };

    OpenAIModelsList {
        object: "list",
        data: models,
    }
}

/// Find a single model by name (matches public_name, name, or alias)
fn find_model_by_name(state: &AppState, requested_model: &str) -> Option<OpenAIModel> {
    let cached_tags = state.cached_tags.read().expect("cached_tags read");

    // First, try to find in cache by name
    let model_info = cached_tags.as_ref().and_then(|tags| {
        tags.models
            .iter()
            .find(|m| model_names_match(&m.name, requested_model))
    });

    if let Some(model) = model_info {
        let created = parse_created_timestamp(&model.modified_at);
        return Some(OpenAIModel {
            id: model.name.clone(),
            object: "model",
            created,
            owned_by: "all-llama-proxy".to_string(),
        });
    }

    // Also check if it's an alias that resolves to a real model
    let config = state.model_config.read().expect("model_config read");
    if let Some(real_model) = config.resolve_alias(requested_model) {
        // Find the real model in cache
        if let Some(real_model_info) = cached_tags
            .as_ref()
            .and_then(|tags| tags.models.iter().find(|m| m.name == real_model))
        {
            let created = parse_created_timestamp(&real_model_info.modified_at);
            return Some(OpenAIModel {
                id: real_model_info.name.clone(),
                object: "model",
                created,
                owned_by: "all-llama-proxy".to_string(),
            });
        }
    }

    None
}

enum SelectionResult {
    Dispatch(String, Task, usize, String),
    ModelNotFound(Task, String),
    Wait,
}

struct DispatchTaskArgs {
    user_id: String,
    task: Task,
    backend_idx: usize,
    backend_url: String,
    state: Arc<AppState>,
    client: reqwest::Client,
}

fn dispatch_task(args: DispatchTaskArgs) {
    let DispatchTaskArgs {
        user_id,
        task,
        backend_idx,
        backend_url,
        state,
        client,
    } = args;
    let url = format!("{}{}", backend_url, task.path);

    if state.debug {
        debug!(
            "Spawning task for user {} -> backend {} ({})",
            user_id, backend_url, task.path
        );
    }

    tokio::spawn(async move {
        let start = Instant::now();

        let is_blocked = {
            let user_ips = state.user_ips.lock().lock_unwrap("user_ips");
            let blocked_ips = state.blocked_ips.lock().lock_unwrap("blocked_ips");
            let blocked_users = state.blocked_users.lock().lock_unwrap("blocked_users");
            blocked_users.contains(&user_id)
                || user_ips
                    .get(&user_id)
                    .map(|ip| blocked_ips.contains(ip))
                    .unwrap_or(false)
        };

        if is_blocked || task.responder.is_closed() {
            let mut dropped = state.dropped_counts.lock().lock_unwrap("dropped_counts");
            *dropped.entry(user_id.clone()).or_insert(0) += 1;
        } else {
            {
                let mut processing = state
                    .processing_counts
                    .lock()
                    .lock_unwrap("processing_counts");
                *processing.entry(user_id.clone()).or_insert(0) += 1;
            }

            if state.debug {
                debug!("=== BACKEND REQUEST ===");
                debug!("URL: {}", url);
                debug!("Method: {:?}", task.method);
                debug!("Headers: {:?}", task.headers);
                if let Ok(body_str) = std::str::from_utf8(&task.body) {
                    debug!("Body: {}", body_str);
                } else {
                    debug!("Body: <binary data> {} bytes", task.body.len());
                }
            }

            let res_fut = client
                .request(task.method, &url)
                .headers(task.headers)
                .body(task.body)
                .send();

            match res_fut.await {
                Ok(response) => {
                    let status = response.status();

                    if state.debug {
                        debug!(
                            "Backend {} responded with status {} for user {}",
                            backend_url, status, user_id
                        );
                    }

                    let mut headers = response.headers().clone();
                    headers.remove(axum::http::header::TRANSFER_ENCODING);
                    headers.remove(axum::http::header::CONTENT_LENGTH);

                    if task
                        .responder
                        .send(ResponsePart::Status(status, headers))
                        .await
                        .is_ok()
                    {
                        let mut stream = response.bytes_stream();
                        let mut client_disconnected = false;
                        while let Some(chunk_res) = stream.next().await {
                            match chunk_res {
                                Ok(chunk) => {
                                    if task
                                        .responder
                                        .send(ResponsePart::Chunk(chunk))
                                        .await
                                        .is_err()
                                    {
                                        client_disconnected = true;
                                        break;
                                    }
                                }
                                Err(_) => break,
                            }
                        }

                        if !client_disconnected {
                            let mut counts = state
                                .processed_counts
                                .lock()
                                .lock_unwrap("processed_counts");
                            *counts.entry(user_id.clone()).or_insert(0) += 1;

                            // Log completion
                            let model_info = task
                                .resolved_model
                                .as_ref()
                                .map(|m| format!(" using {}", m))
                                .unwrap_or_default();
                            info!(
                                "Request finished for user {}{}, duration {}",
                                user_id,
                                model_info,
                                format_duration_short(start.elapsed())
                            );
                        } else {
                            let mut dropped =
                                state.dropped_counts.lock().lock_unwrap("dropped_counts");
                            *dropped.entry(user_id.clone()).or_insert(0) += 1;
                        }
                    }
                }
                Err(e) => {
                    error!(
                        "Backend {} request failed for user {}: {}",
                        backend_url, user_id, e
                    );
                    let _ = task.responder.send(ResponsePart::Error(e)).await;
                    let mut dropped = state.dropped_counts.lock().lock_unwrap("dropped_counts");
                    *dropped.entry(user_id.clone()).or_insert(0) += 1;

                    // Log failure
                    let model_info = task
                        .resolved_model
                        .as_ref()
                        .map(|m| format!(" using {}", m))
                        .unwrap_or_default();
                    info!(
                        "Request failed for user {}{}, duration {}",
                        user_id,
                        model_info,
                        format_duration_short(start.elapsed())
                    );
                }
            }

            {
                let mut processing = state
                    .processing_counts
                    .lock()
                    .lock_unwrap("processing_counts");
                if let Some(count) = processing.get_mut(&user_id) {
                    *count = count.saturating_sub(1);
                }
            }
        }

        {
            let mut backends = state.backends.lock().lock_unwrap("backends");
            let backend = &mut backends[backend_idx];
            backend.active_requests = backend.active_requests.saturating_sub(1);
            if let Some(model) = task.resolved_model.as_ref() {
                let count = backend.active_models.entry(model.clone()).or_insert(0);
                *count = count.saturating_sub(1);
            }
            backend.processed_count += 1;
            if let Some(model) = task.resolved_model.as_ref() {
                *backend.processed_models.entry(model.clone()).or_insert(0) += 1;
            }
        }
        state.backend_freed.notify_one();
    });
}

async fn handle_model_not_found(task: Task, model_name: String) {
    warn!("No backend has model '{}', returning 503", model_name);
    let error_body = Bytes::from(
        serde_json::json!({"error": format!("Model '{}' not available on any backend", model_name)}).to_string()
    );
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    let _ = task
        .responder
        .send(ResponsePart::Status(
            StatusCode::SERVICE_UNAVAILABLE,
            headers,
        ))
        .await;
    let _ = task.responder.send(ResponsePart::Chunk(error_body)).await;
}

fn select_and_prepare_task(state: &Arc<AppState>, current_idx: &mut usize) -> SelectionResult {
    let mut queues = state.queues.lock().lock_unwrap("queues");
    let mut backends = state.backends.lock().lock_unwrap("backends");

    // 1. Find all available online backends (per-model concurrency limits apply below)
    let online_indices: Vec<usize> = backends
        .iter()
        .enumerate()
        .filter(|(_, b)| b.is_online)
        .map(|(i, _)| i)
        .collect();

    if online_indices.is_empty() {
        return SelectionResult::Wait;
    }

    let mut target_user = None;
    let vip_list = state.vip_user.lock().lock_unwrap("vip_user").clone();
    let mut counter = state.global_counter.lock().lock_unwrap("global_counter");

    let mut active_users: Vec<String> = queues
        .iter()
        .filter(|(_, q)| !q.is_empty())
        .map(|(u, _)| u.clone())
        .collect();

    if active_users.is_empty() {
        return SelectionResult::Wait;
    }

    active_users.sort_by(|a, b| {
        let a_total = state
            .processed_counts
            .lock()
            .lock_unwrap("processed_counts")
            .get(a)
            .cloned()
            .unwrap_or(0);
        let b_total = state
            .processed_counts
            .lock()
            .lock_unwrap("processed_counts")
            .get(b)
            .cloned()
            .unwrap_or(0);
        a_total.cmp(&b_total).then_with(|| a.cmp(b))
    });

    // Priority: VIP users first (in order), then round-robin
    for vip_user in &vip_list {
        if active_users.contains(vip_user) {
            target_user = Some(vip_user.clone());
            break;
        }
    }

    if target_user.is_none() {
        if *current_idx >= active_users.len() {
            *current_idx = 0;
        }
        target_user = Some(active_users[*current_idx].clone());
        *current_idx += 1;
    }

    let user_id = match target_user {
        Some(u) => u,
        None => return SelectionResult::Wait,
    };

    // Peek at the task to get the model BEFORE popping
    let task_body = match queues.get(&user_id).and_then(|q| q.front()) {
        Some(task) => task.body.clone(),
        None => {
            warn!("User {} disappeared from queues after selection", user_id);
            return SelectionResult::Wait;
        }
    };

    let is_debug = state.debug;
    let config = state.model_config.read().expect("model_config read");

    // Peek at the original requested model name
    let requested_model = peek_model_from_body(&task_body);

    // Resolve to the exact model name as spelled in models.yaml (for routing
    // decisions). Matching tolerates an implicit `:latest` tag on either side,
    // but the routing key must keep the config spelling because that is what
    // `BackendStatus::configured_models` and `model_status` are keyed by.
    let resolved_model = requested_model
        .as_deref()
        .and_then(|model| config.resolve_alias(model));

    drop(config);

    match resolved_model {
        Some(model) => {
            // Filter backends: can serve this model
            let eligible: Vec<usize> = online_indices
                .iter()
                .cloned()
                .filter(|&i| backends[i].can_serve_model(&model))
                .collect();

            if eligible.is_empty() {
                // Nothing online can serve the model right now. A backend that is
                // merely offline or temporarily missing the model will recover, so
                // wait without popping. But if no backend is configured for the
                // model at all, waiting can never succeed and the task would block
                // the head of this user's queue forever - fail it instead.
                let configured_anywhere = backends
                    .iter()
                    .any(|b| b.configured_models.iter().any(|m| m == &model));

                if !configured_anywhere {
                    let task = match queues.get_mut(&user_id).and_then(|q| q.pop_front()) {
                        Some(t) => t,
                        None => {
                            warn!("User {} disappeared from queues", user_id);
                            return SelectionResult::Wait;
                        }
                    };
                    *counter += 1;
                    return SelectionResult::ModelNotFound(task, model);
                }

                // No eligible backend - wait, do NOT pop the task
                if is_debug {
                    debug!(
                        "No backend can serve model '{}' for user {}",
                        model, user_id
                    );
                }
                SelectionResult::Wait
            } else {
                // Get per-model concurrency limit from config
                let config = state.model_config.read().expect("model_config read");
                let max_concurrency = config
                    .get_model(&model)
                    .map(|m| m.max_concurrent_requests)
                    .unwrap_or(1);
                drop(config);

                // Helper to get per-model count for this specific model
                let get_model_count = |backend_idx: usize| -> usize {
                    backends[backend_idx]
                        .active_models
                        .get(&model)
                        .copied()
                        .unwrap_or(0)
                };

                // Sticky selection: among backends still under the per-model
                // concurrency cap, pick the one with the highest rendezvous
                // weight for this user. This pins a user to a stable backend
                // (kv-cache reuse) while spreading distinct users across GPUs;
                // when the preferred backend is capped, the next-ranked one
                // takes over automatically.
                let selected_backend_idx = eligible
                    .iter()
                    .copied()
                    .filter(|&i| get_model_count(i) < max_concurrency)
                    .max_by(|&a, &b| {
                        rendezvous_weight(&user_id, &backends[a].url)
                            .cmp(&rendezvous_weight(&user_id, &backends[b].url))
                            .then(a.cmp(&b))
                    });

                let selected_backend_idx = match selected_backend_idx {
                    Some(i) => i,
                    None => {
                        // All backends at max concurrency - wait, do NOT pop the task
                        if is_debug {
                            debug!(
                                "All backends at max concurrency ({}) for model '{}', user {} waiting",
                                max_concurrency, model, user_id
                            );
                        }
                        return SelectionResult::Wait;
                    }
                };

                // NOW pop the task (backend is guaranteed available)
                let mut task = match queues.get_mut(&user_id).and_then(|q| q.pop_front()) {
                    Some(t) => t,
                    None => {
                        warn!("User {} disappeared from queues before pop", user_id);
                        return SelectionResult::Wait;
                    }
                };
                *counter += 1;

                // Resolve alias in body (mutate the body)
                let config = state.model_config.read().expect("model_config read");
                let resolved_model_name =
                    extract_and_resolve_model(&mut task.body, &task.path, &config, is_debug);

                // Store resolved model in task and log if alias was used
                if let Some(ref resolved) = resolved_model_name {
                    task.resolved_model = Some(resolved.clone());

                    // Log alias rewrite if different from requested
                    // (a tag-only difference like `foo` -> `foo:latest` is not a rewrite)
                    let is_alias_rewrite = requested_model
                        .as_ref()
                        .is_some_and(|m| !model_names_match(m, resolved));

                    if is_alias_rewrite {
                        info!(
                            "Mapped requested model {} to {} for user {}",
                            requested_model.as_deref().unwrap_or("unknown"),
                            resolved,
                            user_id
                        );
                    }
                }

                if is_debug {
                    debug!(
                        "Selected backend {} (idx {}) for user {}, model {}",
                        backends[selected_backend_idx].url, selected_backend_idx, user_id, model
                    );
                }

                backends[selected_backend_idx].active_requests += 1;
                *backends[selected_backend_idx]
                    .active_models
                    .entry(model.clone())
                    .or_insert(0) += 1;
                SelectionResult::Dispatch(
                    user_id,
                    task,
                    selected_backend_idx,
                    backends[selected_backend_idx].url.clone(),
                )
            }
        }
        None => {
            // Model not in config - pop and fail
            let task = match queues.get_mut(&user_id).and_then(|q| q.pop_front()) {
                Some(t) => t,
                None => {
                    warn!("User {} disappeared from queues", user_id);
                    return SelectionResult::Wait;
                }
            };
            *counter += 1;

            if is_debug {
                debug!("Model not in config for user {}", user_id);
            }

            SelectionResult::ModelNotFound(task, requested_model.unwrap_or("unknown".to_string()))
        }
    }
}

pub async fn run_worker(state: Arc<AppState>) {
    let mut current_idx = 0;

    // Trigger keep-alive on startup
    info!("Triggering initial model keep-alive");
    state.trigger_all_keep_alives().await;

    crate::health::spawn_health_checker(state.clone());
    crate::health::spawn_model_keeper(state.clone());

    loop {
        let selection = select_and_prepare_task(&state, &mut current_idx);

        match selection {
            SelectionResult::ModelNotFound(task, model_name) => {
                handle_model_not_found(task, model_name).await;
            }
            SelectionResult::Dispatch(user_id, task, backend_idx, backend_url) => {
                dispatch_task(DispatchTaskArgs {
                    user_id,
                    task,
                    backend_idx,
                    backend_url,
                    state: state.clone(),
                    client: state.client.clone(),
                });
            }
            SelectionResult::Wait => {
                tokio::select! {
                    _ = state.notify.notified() => {},
                    _ = state.backend_freed.notified() => {},
                }
            }
        }
    }
}

pub async fn proxy_handler(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    method: Method,
    headers: HeaderMap,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
    body: Bytes,
) -> impl IntoResponse {
    let path = uri.path().to_string();
    let ip = extract_client_ip(&headers, addr, &state.ip_header);
    let is_debug = state.debug;

    // --- Authentication ---
    let user_id = match authenticate_request(&headers, &state.user_registry, ip, is_debug) {
        Some(uid) => uid,
        None => return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response(),
    };

    if is_debug {
        debug!("Request from user: {} to {} {}", user_id, method, path);
        #[allow(clippy::collapsible_if)]
        if path.starts_with("/api/generate") || path.starts_with("/api/chat") {
            if let Ok(body_str) = std::str::from_utf8(&body) {
                debug!("Request body: {}", body_str);
            }
        }
    }

    if state.is_ip_blocked(&ip) {
        warn!("Blocked request from IP: {} for user: {}", ip, user_id);
        return (StatusCode::FORBIDDEN, "IP blocked").into_response();
    }

    if state.is_user_blocked(&user_id) {
        warn!("Blocked request from user: {} (IP: {})", user_id, ip);
        return (StatusCode::FORBIDDEN, "User blocked").into_response();
    }

    {
        let mut ips = state.user_ips.lock().lock_unwrap("user_ips");
        ips.insert(user_id.clone(), ip);
    }

    let (tx, rx) = mpsc::channel(32);
    let mut task_headers = headers.clone();
    task_headers.remove(axum::http::header::HOST);
    task_headers.remove(axum::http::header::AUTHORIZATION);
    task_headers.remove(axum::http::header::CONTENT_LENGTH);

    let normalized_path = normalize_path(&path);

    let task = Task {
        path: normalized_path.to_string(),
        method,
        headers: task_headers,
        responder: tx,
        body,
        resolved_model: None,
    };

    {
        let mut queues = state.queues.lock().lock_unwrap("queues");
        queues
            .entry(user_id.clone())
            .or_insert_with(VecDeque::new)
            .push_back(task);
    }

    state.notify.notify_one();

    if is_debug {
        debug!("Task queued for user: {}", user_id);
    }

    let mut rx = rx;
    match rx.recv().await {
        Some(ResponsePart::Status(status, headers)) => {
            if is_debug {
                debug!("Received response status {} for user: {}", status, user_id);
            }
            let stream = ReceiverStream::new(rx).map(|part| match part {
                ResponsePart::Chunk(chunk) => Ok(chunk),
                ResponsePart::Error(e) => Err(e),
                ResponsePart::ModelNotFound(_) | ResponsePart::Status(_, _) => Ok(Bytes::new()),
            });

            let mut res = Body::from_stream(stream).into_response();
            *res.status_mut() = status;
            *res.headers_mut() = headers;
            res
        }
        Some(ResponsePart::Error(e)) => {
            error!("Backend error for user {}: {}", user_id, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Backend error: {}", e),
            )
                .into_response()
        }
        _ => {
            error!("Worker failed to respond for user {}", user_id);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Worker failed to respond",
            )
                .into_response()
        }
    }
}

pub async fn tags_handler(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    _method: Method,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Extract IP
    let ip = extract_client_ip(&headers, addr, &state.ip_header);

    // Authentication
    let user_id = match authenticate_request(&headers, &state.user_registry, ip, true) {
        Some(uid) => uid,
        None => return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response(),
    };

    // Check blocking
    if state.is_ip_blocked(&ip) {
        warn!("Blocked tags request from IP: {} for user: {}", ip, user_id);
        return (StatusCode::FORBIDDEN, "IP blocked").into_response();
    }
    if state.is_user_blocked(&user_id) {
        warn!("Blocked tags request from user: {} (IP: {})", user_id, ip);
        return (StatusCode::FORBIDDEN, "User blocked").into_response();
    }

    // Return cached response (or empty list if not populated)
    let cache = state.cached_tags.read().expect("cached_tags read");
    match cache.as_ref() {
        Some(cached_tags) => (StatusCode::OK, axum::Json(cached_tags.clone())).into_response(),
        None => (
            StatusCode::OK,
            axum::Json(serde_json::json!({"models": []})),
        )
            .into_response(),
    }
}

/// OpenAI-compatible models list handler
pub async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models_list = build_models_list(&state.cached_tags.read().expect("cached_tags read"));
    (StatusCode::OK, axum::Json(models_list)).into_response()
}

/// OpenAI-compatible single model handler
pub async fn model_handler(
    State(state): State<Arc<AppState>>,
    Path(model_name): Path<String>,
) -> impl IntoResponse {
    match find_model_by_name(&state, &model_name) {
        Some(model) => (StatusCode::OK, axum::Json(model)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            axum::Json(serde_json::json!({
                "error": {
                    "code": "model_not_found",
                    "message": format!("Model '{}' not found", model_name)
                }
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::appstate::{AppState, BackendStatus, LogBuffer, ModelConfig};
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex, RwLock};
    use tokio::sync::Notify;

    fn create_test_state() -> Arc<AppState> {
        let registry = Arc::new(UserRegistry::empty());
        let log_buffer = LogBuffer::new(100);
        let config = ModelConfig { models: vec![] };
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap();

        Arc::new(AppState {
            queues: Mutex::new(HashMap::new()),
            processing_counts: Mutex::new(HashMap::new()),
            processed_counts: Mutex::new(HashMap::new()),
            dropped_counts: Mutex::new(HashMap::new()),
            user_ips: Mutex::new(HashMap::new()),
            blocked_ips: Mutex::new(HashSet::new()),
            blocked_users: Mutex::new(HashSet::new()),
            vip_user: Mutex::new(Vec::new()),
            global_counter: Mutex::new(0),
            notify: Notify::new(),
            backend_freed: Notify::new(),
            backends: Mutex::new(vec![]),
            last_backend_idx: Mutex::new(0),
            timeout: 300,
            client,
            user_registry: Mutex::new(registry),
            model_config: Arc::new(RwLock::new(config)),
            debug: false,
            log_buffer: log_buffer.clone(),
            ip_header: None,
            cached_tags: Arc::new(RwLock::new(None)),
            health_check_interval: 10,
        })
    }

    #[tokio::test]
    async fn test_worker_doesnt_deadlock() {
        // Simple smoke test: create state and call run_worker briefly
        let state = create_test_state();

        // Add a backend and user with empty queue
        {
            let mut backends = state.backends.lock().lock_unwrap("backends");
            backends.push(BackendStatus {
                url: "http://localhost:11434".to_string(),
                configured_models: vec![],
                active_requests: 0,
                processed_count: 0,
                is_online: true,
                active_models: HashMap::new(),
                processed_models: HashMap::new(),
                model_status: Arc::new(RwLock::new(HashMap::new())),
            });
        }

        // Test completes without hanging
        assert!(state.backends.lock().lock_unwrap("backends").len() == 1);
    }

    fn parsed_model(name: &str, aliases: &[&str]) -> crate::appstate::ParsedModel {
        crate::appstate::ParsedModel {
            name: name.to_string(),
            public_name: None,
            backends: vec!["http://localhost:11434".to_string()],
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            max_concurrent_requests: 1,
            keep_alive: true,
        }
    }

    fn resolve(config: &ModelConfig, requested: &str) -> (Option<String>, String) {
        let mut body = Bytes::from(format!(r#"{{"model":"{}"}}"#, requested));
        let resolved = extract_and_resolve_model(&mut body, "/api/chat", config, false);
        let sent_model: String = serde_json::from_slice::<serde_json::Value>(&body)
            .unwrap()
            .get("model")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        (resolved, sent_model)
    }

    fn test_backend(configured: &[&str]) -> BackendStatus {
        let mut backend = BackendStatus::new("http://localhost:11434".to_string());
        backend.configured_models = configured.iter().map(|s| s.to_string()).collect();
        backend
    }

    #[test]
    fn test_untagged_config_model_is_routable() {
        // Regression: an untagged models.yaml name must not be rewritten to
        // `:latest`, otherwise it never matches configured_models.
        let config = ModelConfig {
            models: vec![parsed_model("muse-glimmer", &[])],
        };
        let backend = test_backend(&["muse-glimmer"]);

        let (resolved, sent) = resolve(&config, "muse-glimmer");
        assert_eq!(resolved.as_deref(), Some("muse-glimmer"));
        assert_eq!(sent, "muse-glimmer");
        assert!(backend.can_serve_model(&resolved.unwrap()));
    }

    #[test]
    fn test_explicit_latest_resolves_to_config_spelling() {
        let config = ModelConfig {
            models: vec![parsed_model("muse-glimmer", &[])],
        };
        let backend = test_backend(&["muse-glimmer"]);

        let (resolved, sent) = resolve(&config, "muse-glimmer:latest");
        assert_eq!(resolved.as_deref(), Some("muse-glimmer"));
        assert_eq!(sent, "muse-glimmer", "body must carry the config spelling");
        assert!(backend.can_serve_model(&resolved.unwrap()));
    }

    #[test]
    fn test_tagged_config_model_keeps_its_tag() {
        let config = ModelConfig {
            models: vec![parsed_model("qwen3:35b", &["qwen3"])],
        };
        let backend = test_backend(&["qwen3:35b"]);

        let (resolved, sent) = resolve(&config, "qwen3:35b");
        assert_eq!(resolved.as_deref(), Some("qwen3:35b"));
        assert_eq!(sent, "qwen3:35b");
        assert!(backend.can_serve_model(&resolved.unwrap()));

        // Reachable via the explicit alias, which rewrites the body.
        let (resolved, sent) = resolve(&config, "qwen3");
        assert_eq!(resolved.as_deref(), Some("qwen3:35b"));
        assert_eq!(sent, "qwen3:35b");
    }

    #[test]
    fn test_bare_name_does_not_match_unrelated_tag() {
        // `foo` means `foo:latest`, never `foo:7b`.
        let config = ModelConfig {
            models: vec![parsed_model("qwen3:35b", &[])],
        };
        let (resolved, sent) = resolve(&config, "qwen3");
        assert_eq!(resolved.as_deref(), Some("qwen3"));
        assert_eq!(sent, "qwen3", "unknown model body must stay untouched");
    }

    #[tokio::test]
    async fn test_model_without_any_backend_is_failed_not_queued() {
        // A task that no backend is configured for must be popped and 503'd,
        // otherwise it blocks the head of the user's queue forever.
        let state = create_test_state();
        *state.model_config.write().unwrap() = ModelConfig {
            models: vec![parsed_model("ghost:1b", &[])],
        };
        state
            .backends
            .lock()
            .lock_unwrap("backends")
            .push(test_backend(&["other:1b"]));

        let (tx, _rx) = mpsc::channel(32);
        state
            .queues
            .lock()
            .lock_unwrap("queues")
            .entry("alice".to_string())
            .or_insert_with(VecDeque::new)
            .push_back(Task {
                method: Method::POST,
                path: "/api/chat".to_string(),
                headers: HeaderMap::new(),
                body: Bytes::from(r#"{"model":"ghost:1b"}"#),
                responder: tx,
                resolved_model: None,
            });

        let mut idx = 0;
        match select_and_prepare_task(&state, &mut idx) {
            SelectionResult::ModelNotFound(_, model) => assert_eq!(model, "ghost:1b"),
            _ => panic!("expected ModelNotFound for a model without any backend"),
        }
        assert!(
            state.queues.lock().lock_unwrap("queues")["alice"].is_empty(),
            "task must be removed from the queue"
        );
    }

    #[test]
    fn test_rendezvous_weight_deterministic() {
        let a = rendezvous_weight("alice", "http://gpu1:11434");
        let b = rendezvous_weight("alice", "http://gpu1:11434");
        assert_eq!(a, b);
        // Different inputs should (almost certainly) differ.
        assert_ne!(a, rendezvous_weight("bob", "http://gpu1:11434"));
        assert_ne!(a, rendezvous_weight("alice", "http://gpu2:11434"));
    }

    // Pick the highest-weight backend for a user, tie-broken by index,
    // mirroring the selection logic in select_and_prepare_task.
    fn pick_backend(user: &str, urls: &[&str]) -> usize {
        urls.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                rendezvous_weight(user, a)
                    .cmp(&rendezvous_weight(user, b))
                    .then(ia.cmp(ib))
            })
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn test_rendezvous_sticky_per_user() {
        let urls = [
            "http://gpu1:11434",
            "http://gpu2:11434",
            "http://gpu3:11434",
        ];
        let first = pick_backend("alice", &urls);
        // Same user always maps to the same backend.
        for _ in 0..100 {
            assert_eq!(pick_backend("alice", &urls), first);
        }
    }

    #[test]
    fn test_rendezvous_spreads_users() {
        let urls = [
            "http://gpu1:11434",
            "http://gpu2:11434",
            "http://gpu3:11434",
        ];
        let mut seen = HashSet::new();
        for n in 0..200 {
            seen.insert(pick_backend(&format!("user{}", n), &urls));
        }
        // With 200 users over 3 backends, every backend should be used.
        assert_eq!(seen.len(), urls.len());
    }

    #[test]
    fn test_rendezvous_stable_when_unrelated_backend_removed() {
        let full = [
            "http://gpu1:11434",
            "http://gpu2:11434",
            "http://gpu3:11434",
        ];
        // Find a user not pinned to the last backend, then remove that last
        // backend and confirm the user's pick is unchanged.
        let user = (0..1000)
            .map(|n| format!("user{}", n))
            .find(|u| pick_backend(u, &full) != full.len() - 1)
            .expect("expected some user not pinned to the last backend");

        let pick_full = pick_backend(&user, &full);
        let reduced = &full[..full.len() - 1];
        assert_eq!(pick_backend(&user, reduced), pick_full);
    }
}
