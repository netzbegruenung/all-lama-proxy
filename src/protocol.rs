use bincode::config::standard;
use bincode::serde::decode_from_slice;
use bincode::serde::encode_to_vec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ───────────────
// Server → TUI types
// ───────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub queues_len: HashMap<String, usize>,
    pub processing_counts: HashMap<String, usize>,
    pub processed_counts: HashMap<String, usize>,
    pub dropped_counts: HashMap<String, usize>,
    pub user_ips: HashMap<String, String>,
    pub blocked_ips: HashSet<String>,
    pub blocked_users: HashSet<String>,
    pub vip_list: Vec<String>,
    pub user_ids: Vec<String>,
    pub backends: Vec<BackendSnapshot>,
    pub model_public_names: HashMap<String, String>,
    pub log_lines: Vec<(String, i64, String)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackendSnapshot {
    pub url: String,
    pub active_requests: usize,
    pub processed_count: usize,
    pub is_online: bool,
    pub active_models: HashMap<String, usize>,
    pub processed_models: HashMap<String, usize>,
    pub configured_models: Vec<String>,
    pub model_status: HashMap<String, bool>,
}

// ───────────────
// TUI → Server types
// ───────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DashboardCmd {
    ToggleVip(String),
    BlockUser(String),
    UnblockUser(String),
    BlockIp(String),
    UnblockIp(String),
}

// ───────────────
// Wire helpers
// ───────────────

/// Maximum length of a single wire message's payload, excluding the 4-byte
/// length header.
///
/// A peer must not advertise a length larger than this. Capping the length
/// prevents an unauthenticated dashboard socket peer from driving unbounded
/// memory growth in the read buffer (length-prefix DoS).
pub const MAX_MESSAGE_LEN: usize = 4 * 1024 * 1024;

pub fn encode<T: Serialize + std::fmt::Debug>(payload: &T) -> std::io::Result<Vec<u8>> {
    let config = standard();
    let body = encode_to_vec(payload, config).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bincode encode error: {:?}, payload: {:?}", e, payload),
        )
    })?;

    // Refuse to emit a message no peer could decode: `decode` rejects anything
    // above the cap, so writing it would strand the reader waiting for a
    // message that never completes.
    if body.len() > MAX_MESSAGE_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "encoded message of {} bytes exceeds the {}-byte wire limit",
                body.len(),
                MAX_MESSAGE_LEN
            ),
        ));
    }

    let len = (body.len() as u32).to_be_bytes();
    let mut msg = Vec::with_capacity(body.len() + 4);
    msg.extend_from_slice(&len);
    msg.extend_from_slice(&body);
    Ok(msg)
}

/// Returns the message body length from the 4-byte big-endian prefix, or
/// `Ok(None)` if the header itself is not fully buffered yet.
///
/// An advertised length above [`MAX_MESSAGE_LEN`] is an error, reported here on
/// sight: the caller can drop the peer as soon as the header arrives instead of
/// first buffering the oversized payload to discover it is too large.
fn prefix_len(buf: &[u8]) -> std::io::Result<Option<usize>> {
    if buf.len() < 4 {
        return Ok(None);
    }
    let len_bytes: [u8; 4] = [buf[0], buf[1], buf[2], buf[3]];
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > MAX_MESSAGE_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "peer advertised a {}-byte message, exceeding the {}-byte limit",
                len, MAX_MESSAGE_LEN
            ),
        ));
    }
    Ok(Some(len))
}

pub fn decode<T: for<'a> Deserialize<'a>>(buf: &[u8]) -> std::io::Result<Option<T>> {
    let Some(len) = prefix_len(buf)? else {
        return Ok(None);
    };
    if buf.len() < 4 + len {
        return Ok(None);
    }
    let data = &buf[4..4 + len];
    let config = standard();
    decode_from_slice::<T, _>(data, config)
        .map(|(value, _)| Some(value))
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bincode decode error: {}", e),
            )
        })
}

/// Returns how many bytes a complete message occupies (header + payload), or
/// `Ok(None)` if it is not fully buffered yet.
///
/// Returns `Err` if the header advertises a payload above [`MAX_MESSAGE_LEN`];
/// callers should treat that as grounds for dropping the peer.
pub fn consumed_len(buf: &[u8]) -> std::io::Result<Option<usize>> {
    let Some(len) = prefix_len(buf)? else {
        return Ok(None);
    };
    if buf.len() >= 4 + len {
        Ok(Some(4 + len))
    } else {
        Ok(None)
    }
}
