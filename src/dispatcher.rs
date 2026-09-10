use axum::{
    body::{Body, Bytes},
    extract::{ConnectInfo, Path, State},
    http::{HeaderMap, Method, StatusCode},
    response::IntoResponse,
};
use chrono::Utc;
use futures_util::StreamExt;
use std::{
    collections::HashMap,
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
    AppState, BackendStatus, CachedTags, ModelConfig, OpenAIModel, OpenAIModelsList, ResponsePart,
    Task, model_names_match,
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

/// How a model can be routed at this instant, independent of who is asking.
enum ModelRoute {
    /// The model is unknown to the config, or no backend is configured for it.
    /// Waiting can never help, so tasks for it are failed with the carried name.
    Unavailable(String),
    /// Backends are configured but none can take a task right now: offline,
    /// model not loaded, or at the per-model concurrency cap. Retry next pass.
    Busy,
    /// Backends that can accept a task for `model` right now.
    Ready {
        model: String,
        backend_indices: Vec<usize>,
    },
}

/// Resolve `requested_model` and collect the backends that could take it now.
///
/// The result does not depend on the requesting user, so
/// [`select_and_prepare_task`] caches it per pass: a user with 50 queued
/// requests for the same model only pays for this once.
fn route_model(
    state: &Arc<AppState>,
    backends: &[BackendStatus],
    online_indices: &[usize],
    requested_model: Option<&str>,
) -> ModelRoute {
    let config = state.model_config.read().expect("model_config read");

    // Resolve to the exact model name as spelled in models.yaml (for routing
    // decisions). Matching tolerates an implicit `:latest` tag on either side,
    // but the routing key must keep the config spelling because that is what
    // `BackendStatus::configured_models` and `model_status` are keyed by.
    let model = match requested_model.and_then(|model| config.resolve_alias(model)) {
        Some(model) => model,
        None => {
            return ModelRoute::Unavailable(requested_model.unwrap_or("unknown").to_string());
        }
    };
    let max_concurrency = config
        .get_model(&model)
        .map(|m| m.max_concurrent_requests)
        .unwrap_or(1);
    drop(config);

    let eligible: Vec<usize> = online_indices
        .iter()
        .copied()
        .filter(|&i| backends[i].can_serve_model(&model))
        .collect();

    if eligible.is_empty() {
        // A backend that is merely offline or temporarily missing the model
        // will recover, so the task keeps waiting. But if no backend is
        // configured for the model at all, waiting can never succeed.
        let configured_anywhere = backends
            .iter()
            .any(|b| b.configured_models.iter().any(|m| m == &model));

        return if configured_anywhere {
            ModelRoute::Busy
        } else {
            ModelRoute::Unavailable(model)
        };
    }

    let backend_indices: Vec<usize> = eligible
        .into_iter()
        .filter(|&i| backends[i].active_models.get(&model).copied().unwrap_or(0) < max_concurrency)
        .collect();

    if backend_indices.is_empty() {
        ModelRoute::Busy
    } else {
        ModelRoute::Ready {
            model,
            backend_indices,
        }
    }
}

/// A queued task the scheduler can act on right now.
enum TaskPick {
    Dispatch {
        queue_idx: usize,
        model: String,
        backend_idx: usize,
    },
    /// The task's model can never be served - pop it and return 503.
    Fail {
        queue_idx: usize,
        model_name: String,
    },
}

/// Find the first task in `queue` that can be acted on right now.
///
/// Tasks whose model is only temporarily unroutable (backends offline or at
/// their concurrency cap) are skipped instead of blocking everything behind
/// them, so a backlog for one model cannot stall requests for other models.
/// The scan always runs front to back, so tasks for the *same* model keep
/// their FIFO order and a skipped task is retried before any later task.
fn find_dispatchable_task(
    state: &Arc<AppState>,
    backends: &[BackendStatus],
    online_indices: &[usize],
    queue: &VecDeque<Task>,
    user_id: &str,
    routes: &mut HashMap<Option<String>, ModelRoute>,
) -> Option<TaskPick> {
    for (queue_idx, task) in queue.iter().enumerate() {
        let route = routes
            .entry(task.requested_model.clone())
            .or_insert_with(|| {
                route_model(
                    state,
                    backends,
                    online_indices,
                    task.requested_model.as_deref(),
                )
            });

        match route {
            ModelRoute::Unavailable(model_name) => {
                return Some(TaskPick::Fail {
                    queue_idx,
                    model_name: model_name.clone(),
                });
            }
            ModelRoute::Busy => continue,
            ModelRoute::Ready {
                model,
                backend_indices,
            } => {
                // Sticky selection: among backends still under the per-model
                // concurrency cap, pick the one with the highest rendezvous
                // weight for this user. This pins a user to a stable backend
                // (kv-cache reuse) while spreading distinct users across GPUs;
                // when the preferred backend is capped, the next-ranked one
                // takes over automatically.
                let backend_idx = backend_indices.iter().copied().max_by(|&a, &b| {
                    rendezvous_weight(user_id, &backends[a].url)
                        .cmp(&rendezvous_weight(user_id, &backends[b].url))
                        .then(a.cmp(&b))
                });

                if let Some(backend_idx) = backend_idx {
                    return Some(TaskPick::Dispatch {
                        queue_idx,
                        model: model.clone(),
                        backend_idx,
                    });
                }
            }
        }
    }

    None
}

fn select_and_prepare_task(state: &Arc<AppState>, current_idx: &mut usize) -> SelectionResult {
    let mut queues = state.queues.lock().lock_unwrap("queues");
    let mut backends = state.backends.lock().lock_unwrap("backends");
    let is_debug = state.debug;

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

    // 2. Users with something queued, fewest requests processed first
    let mut active_users: Vec<String> = queues
        .iter()
        .filter(|(_, q)| !q.is_empty())
        .map(|(u, _)| u.clone())
        .collect();

    if active_users.is_empty() {
        return SelectionResult::Wait;
    }

    {
        let processed = state
            .processed_counts
            .lock()
            .lock_unwrap("processed_counts");
        active_users.sort_by(|a, b| {
            let a_total = processed.get(a).copied().unwrap_or(0);
            let b_total = processed.get(b).copied().unwrap_or(0);
            a_total.cmp(&b_total).then_with(|| a.cmp(b))
        });
    }

    // 3. Candidate order: VIP users first (in config order), then the rest
    //    round-robin from `current_idx`.
    let vip_list = state.vip_user.lock().lock_unwrap("vip_user").clone();
    let mut candidates: Vec<usize> = Vec::with_capacity(active_users.len());
    for vip in &vip_list {
        if let Some(i) = active_users.iter().position(|u| u == vip) {
            if !candidates.contains(&i) {
                candidates.push(i);
            }
        }
    }
    let start = *current_idx % active_users.len();
    for offset in 0..active_users.len() {
        let i = (start + offset) % active_users.len();
        if !candidates.contains(&i) {
            candidates.push(i);
        }
    }

    // 4. Try every candidate, not just the first one: a user whose next
    //    request is blocked on a saturated model must not stall the requests
    //    that other users have for idle models.
    let mut routes: HashMap<Option<String>, ModelRoute> = HashMap::new();

    for i in candidates {
        let user_id = active_users[i].clone();
        let queue = match queues.get(&user_id) {
            Some(q) => q,
            None => continue,
        };

        let pick = match find_dispatchable_task(
            state,
            &backends,
            &online_indices,
            queue,
            &user_id,
            &mut routes,
        ) {
            Some(pick) => pick,
            None => continue,
        };

        // Advance the round-robin cursor past the user we serve. VIPs are
        // always tried first, so serving one must not move the cursor.
        if !vip_list.contains(&user_id) {
            *current_idx = i + 1;
        }

        let queue_idx = match &pick {
            TaskPick::Dispatch { queue_idx, .. } | TaskPick::Fail { queue_idx, .. } => *queue_idx,
        };
        let mut task = match queues.get_mut(&user_id).and_then(|q| q.remove(queue_idx)) {
            Some(task) => task,
            None => {
                warn!("User {} disappeared from queues before pop", user_id);
                return SelectionResult::Wait;
            }
        };
        *state.global_counter.lock().lock_unwrap("global_counter") += 1;

        match pick {
            TaskPick::Fail { model_name, .. } => {
                if is_debug {
                    debug!(
                        "No backend is configured for model '{}' (user {})",
                        model_name, user_id
                    );
                }
                return SelectionResult::ModelNotFound(task, model_name);
            }
            TaskPick::Dispatch {
                model, backend_idx, ..
            } => {
                // Resolve alias in body (mutate the body)
                let config = state.model_config.read().expect("model_config read");
                let resolved_model_name =
                    extract_and_resolve_model(&mut task.body, &task.path, &config, is_debug);
                drop(config);

                // Store resolved model in task and log if alias was used
                if let Some(resolved) = resolved_model_name {
                    // Log alias rewrite if different from requested
                    // (a tag-only difference like `foo` -> `foo:latest` is not a rewrite)
                    let is_alias_rewrite = task
                        .requested_model
                        .as_ref()
                        .is_some_and(|m| !model_names_match(m, &resolved));

                    if is_alias_rewrite {
                        info!(
                            "Mapped requested model {} to {} for user {}",
                            task.requested_model.as_deref().unwrap_or("unknown"),
                            resolved,
                            user_id
                        );
                    }

                    task.resolved_model = Some(resolved);
                }

                if is_debug {
                    debug!(
                        "Selected backend {} (idx {}) for user {}, model {}",
                        backends[backend_idx].url, backend_idx, user_id, model
                    );
                }

                backends[backend_idx].active_requests += 1;
                *backends[backend_idx]
                    .active_models
                    .entry(model)
                    .or_insert(0) += 1;
                let backend_url = backends[backend_idx].url.clone();
                return SelectionResult::Dispatch(user_id, task, backend_idx, backend_url);
            }
        }
    }

    // Nothing anywhere can run: wait for a backend to free up or a new task.
    SelectionResult::Wait
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
        requested_model: peek_model_from_body(&body),
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

    /// Queue a chat request for `model` and hand back the receiver, which the
    /// caller must keep alive for as long as the task stays queued.
    fn queue_task(
        state: &Arc<AppState>,
        user: &str,
        model: &str,
    ) -> mpsc::Receiver<crate::appstate::ResponsePart> {
        let (tx, rx) = mpsc::channel(32);
        let body = Bytes::from(format!(r#"{{"model":"{}"}}"#, model));
        state
            .queues
            .lock()
            .lock_unwrap("queues")
            .entry(user.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(Task {
                method: Method::POST,
                path: "/api/chat".to_string(),
                headers: HeaderMap::new(),
                requested_model: peek_model_from_body(&body),
                body,
                responder: tx,
                resolved_model: None,
            });
        rx
    }

    /// A backend serving `busy:1b` (saturated at its cap of 1) and `idle:1b`.
    fn saturated_backend() -> BackendStatus {
        let mut backend = test_backend(&["busy:1b", "idle:1b"]);
        backend.active_models.insert("busy:1b".to_string(), 1);
        backend.active_requests = 1;
        backend
    }

    fn two_model_config() -> ModelConfig {
        ModelConfig {
            models: vec![parsed_model("busy:1b", &[]), parsed_model("idle:1b", &[])],
        }
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

        let _rx = queue_task(&state, "alice", "ghost:1b");

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

    #[tokio::test]
    async fn test_busy_model_does_not_block_other_users() {
        // Regression: a backlog on one model used to put the single worker to
        // sleep, so nobody else got served even on completely idle models.
        let state = create_test_state();
        *state.model_config.write().unwrap() = two_model_config();
        state
            .backends
            .lock()
            .lock_unwrap("backends")
            .push(saturated_backend());

        // alice sorts first (equal processed counts, alphabetical), so the old
        // scheduler picked her blocked request and stopped there.
        let _alice = queue_task(&state, "alice", "busy:1b");
        let _bob = queue_task(&state, "bob", "idle:1b");

        let mut idx = 0;
        match select_and_prepare_task(&state, &mut idx) {
            SelectionResult::Dispatch(user, task, _, _) => {
                assert_eq!(user, "bob");
                assert_eq!(task.resolved_model.as_deref(), Some("idle:1b"));
            }
            _ => panic!("bob's request for an idle model must be dispatched"),
        }

        // alice keeps her place in line until busy:1b frees up.
        let queues = state.queues.lock().lock_unwrap("queues");
        assert_eq!(queues["alice"].len(), 1);
        assert!(queues["bob"].is_empty());
    }

    #[tokio::test]
    async fn test_busy_model_does_not_block_later_task_of_same_user() {
        // Within one user's queue a request for a saturated model must not
        // hold back their later requests for other models.
        let state = create_test_state();
        *state.model_config.write().unwrap() = two_model_config();
        state
            .backends
            .lock()
            .lock_unwrap("backends")
            .push(saturated_backend());

        let _blocked = queue_task(&state, "alice", "busy:1b");
        let _runnable = queue_task(&state, "alice", "idle:1b");

        let mut idx = 0;
        match select_and_prepare_task(&state, &mut idx) {
            SelectionResult::Dispatch(user, task, _, _) => {
                assert_eq!(user, "alice");
                assert_eq!(task.resolved_model.as_deref(), Some("idle:1b"));
            }
            _ => panic!("the second queued request must be dispatched"),
        }

        // The skipped request stays at the head, so it runs first once free.
        let queues = state.queues.lock().lock_unwrap("queues");
        assert_eq!(queues["alice"].len(), 1);
        assert_eq!(
            queues["alice"][0].requested_model.as_deref(),
            Some("busy:1b")
        );
    }

    #[tokio::test]
    async fn test_blocked_vip_does_not_stall_other_users() {
        // VIPs are tried first, but a VIP waiting on a saturated model must
        // not starve everyone else indefinitely.
        let state = create_test_state();
        *state.model_config.write().unwrap() = two_model_config();
        state
            .backends
            .lock()
            .lock_unwrap("backends")
            .push(saturated_backend());
        state
            .vip_user
            .lock()
            .lock_unwrap("vip_user")
            .push("vip".to_string());

        let _vip = queue_task(&state, "vip", "busy:1b");
        let _bob = queue_task(&state, "bob", "idle:1b");

        let mut idx = 0;
        match select_and_prepare_task(&state, &mut idx) {
            SelectionResult::Dispatch(user, _, _, _) => assert_eq!(user, "bob"),
            _ => panic!("bob must be dispatched while the VIP waits"),
        }
    }

    #[tokio::test]
    async fn test_vip_user_is_served_first() {
        // The fallback scan must not weaken VIP priority when both are runnable.
        let state = create_test_state();
        *state.model_config.write().unwrap() = two_model_config();
        state
            .backends
            .lock()
            .lock_unwrap("backends")
            .push(test_backend(&["busy:1b", "idle:1b"]));
        state
            .vip_user
            .lock()
            .lock_unwrap("vip_user")
            .push("zoe".to_string());

        let _alice = queue_task(&state, "alice", "idle:1b");
        let _zoe = queue_task(&state, "zoe", "busy:1b");

        let mut idx = 0;
        match select_and_prepare_task(&state, &mut idx) {
            SelectionResult::Dispatch(user, _, _, _) => assert_eq!(user, "zoe"),
            _ => panic!("the VIP must be dispatched first"),
        }
    }

    #[tokio::test]
    async fn test_all_models_busy_waits() {
        // With nothing runnable the worker must still park instead of spinning.
        let state = create_test_state();
        *state.model_config.write().unwrap() = two_model_config();
        state
            .backends
            .lock()
            .lock_unwrap("backends")
            .push(saturated_backend());

        let _alice = queue_task(&state, "alice", "busy:1b");
        let _bob = queue_task(&state, "bob", "busy:1b");

        let mut idx = 0;
        assert!(
            matches!(
                select_and_prepare_task(&state, &mut idx),
                SelectionResult::Wait
            ),
            "expected Wait when every model is at capacity"
        );
        assert_eq!(state.queues.lock().lock_unwrap("queues")["alice"].len(), 1);
        assert_eq!(state.queues.lock().lock_unwrap("queues")["bob"].len(), 1);
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
