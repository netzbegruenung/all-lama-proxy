use all_llama_proxy::{DashboardCmd, DashboardSnapshot, consumed_len, decode, encode};
use std::collections::{HashMap, HashSet};
use std::io;
use std::time::Duration;

use clap::Parser;
use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    backend::CrosstermBackend,
    prelude::*,
    text::Line,
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tracing::info;

#[derive(Clone, PartialEq)]
enum CurrentPanel {
    Users,
    Blocked,
}

pub struct TuiDashboard {
    table_state: TableState,
    blocked_table_state: TableState,
    active_panel: CurrentPanel,
    show_help: bool,
    show_models: bool,
}

impl TuiDashboard {
    pub fn new() -> Self {
        Self {
            table_state: TableState::default(),
            blocked_table_state: TableState::default(),
            active_panel: CurrentPanel::Users,
            show_help: false,
            show_models: false,
        }
    }

    fn render(&mut self, f: &mut Frame, snapshot: &DashboardSnapshot) {
        if self.active_panel == CurrentPanel::Users {
            if snapshot.user_ids.is_empty() {
                self.table_state.select(None);
            } else if self.table_state.selected().is_none() {
                self.table_state.select(Some(0));
            }
        } else {
            let blocked_total = snapshot.blocked_ips.len() + snapshot.blocked_users.len();
            if blocked_total == 0 {
                self.blocked_table_state.select(None);
            } else if self.blocked_table_state.selected().is_none() {
                self.blocked_table_state.select(Some(0));
            }
        }

        let area = f.area();
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(4),
                Constraint::Length(6),
                Constraint::Length(3),
                if self.show_help {
                    Constraint::Length(12)
                } else {
                    Constraint::Length(0)
                },
            ])
            .split(area);

        f.render_widget(self.render_stats(snapshot), main_chunks[0]);

        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(40),
                Constraint::Percentage(35),
            ])
            .split(main_chunks[1]);

        f.render_widget(self.render_backends(snapshot), content_chunks[0]);
        f.render_stateful_widget(
            self.render_users(snapshot),
            content_chunks[1],
            &mut self.table_state,
        );

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(content_chunks[2]);

        f.render_stateful_widget(
            self.render_queues(snapshot, right_chunks[0].width),
            right_chunks[0],
            &mut self.table_state,
        );
        f.render_stateful_widget(
            self.render_blocked(snapshot),
            right_chunks[1],
            &mut self.blocked_table_state,
        );

        f.render_widget(self.render_logs(snapshot, main_chunks[2]), main_chunks[2]);
        f.render_widget(self.render_help(), main_chunks[3]);
        if self.show_help {
            f.render_widget(self.render_detailed_help(), main_chunks[4]);
        }
    }

    fn render_stats(&self, snapshot: &DashboardSnapshot) -> Paragraph<'static> {
        let total_queued: usize = snapshot.queues_len.values().sum();
        let total_processing: usize = snapshot.processing_counts.values().sum();
        let total_processed: usize = snapshot.processed_counts.values().sum();
        let total_dropped: usize = snapshot.dropped_counts.values().sum();

        let stats_line = vec![
            Span::styled(" all-llama-proxy ", Style::default().fg(Color::Cyan).bold()),
            Span::raw(" | "),
            Span::styled("Panel: ", Style::default().fg(Color::White)),
            Span::styled(
                if self.active_panel == CurrentPanel::Users {
                    "USERS"
                } else {
                    "BLOCKED"
                },
                Style::default().fg(Color::Yellow).bold(),
            ),
            Span::raw(" | "),
            Span::styled("Q: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                (total_queued + total_processing).to_string(),
                Style::default().fg(Color::Yellow).bold(),
            ),
            Span::raw(" | "),
            Span::styled("Done: ", Style::default().fg(Color::Green)),
            Span::styled(
                total_processed.to_string(),
                Style::default().fg(Color::Green).bold(),
            ),
            Span::raw(" | "),
            Span::styled("Drop: ", Style::default().fg(Color::Red)),
            Span::styled(
                total_dropped.to_string(),
                Style::default().fg(Color::Red).bold(),
            ),
        ];

        Paragraph::new(Line::from(stats_line)).block(Block::default().borders(Borders::ALL))
    }

    fn render_backends(&self, snapshot: &DashboardSnapshot) -> Table<'static> {
        let rows: Vec<Row> = snapshot
            .backends
            .iter()
            .flat_map(|b| {
                let url = b.url.replace("http://", "").replace("https://", "");

                let (status_sym, status_style) = if b.is_online {
                    ("● ", Style::default().fg(Color::Green))
                } else {
                    ("○ ", Style::default().fg(Color::Red))
                };

                let req_style = if b.active_requests > 0 {
                    Style::default().fg(Color::Cyan).bold()
                } else {
                    Style::default().fg(Color::Gray)
                };

                let header_row = Row::new(vec![
                    Cell::from(Line::from(vec![
                        Span::styled(status_sym, status_style),
                        Span::styled(
                            url,
                            if b.is_online {
                                Style::default().fg(Color::White)
                            } else {
                                Style::default()
                                    .fg(Color::DarkGray)
                                    .add_modifier(Modifier::CROSSED_OUT)
                            },
                        ),
                    ])),
                    Cell::from(b.active_requests.to_string()).style(req_style),
                    Cell::from(b.processed_count.to_string())
                        .style(Style::default().fg(Color::DarkGray)),
                ]);

                let mut rows = vec![header_row];

                if self.show_models {
                    if b.configured_models.is_empty() {
                        rows.push(Row::new(vec![
                            Cell::from("  (no models)").style(Style::default().fg(Color::DarkGray)),
                            Cell::from(""),
                            Cell::from(""),
                        ]));
                    } else {
                        for model in &b.configured_models {
                            let available = b.model_status.get(model).copied().unwrap_or(true);
                            let active = b.active_models.get(model).copied().unwrap_or(0);
                            let done = b.processed_models.get(model).copied().unwrap_or(0);
                            let (sym, style) = if available {
                                ("✓", Style::default().fg(Color::Green))
                            } else {
                                ("✗", Style::default().fg(Color::Red))
                            };
                            let display_name = snapshot
                                .model_public_names
                                .get(model)
                                .cloned()
                                .unwrap_or_else(|| model.clone());
                            rows.push(Row::new(vec![
                                Cell::from(format!("  {} {}", sym, display_name)).style(style),
                                Cell::from(active.to_string()),
                                Cell::from(done.to_string()),
                            ]));
                        }
                    }
                }

                rows
            })
            .collect();

        Table::new(
            rows,
            [
                Constraint::Min(10),
                Constraint::Length(4),
                Constraint::Length(6),
            ],
        )
        .header(
            Row::new(vec!["Backend", "Act", "Done"])
                .style(Style::default().fg(Color::Yellow).bold())
                .bottom_margin(1),
        )
        .block(
            Block::default()
                .title(" Ollama Instances ")
                .borders(Borders::ALL),
        )
    }

    fn render_users(&self, snapshot: &DashboardSnapshot) -> Table<'static> {
        let rows: Vec<Row> = snapshot
            .user_ids
            .iter()
            .map(|user| {
                let queue_len = snapshot.queues_len.get(user).unwrap_or(&0)
                    + snapshot.processing_counts.get(user).unwrap_or(&0);
                let processed = snapshot.processed_counts.get(user).unwrap_or(&0);
                let dropped = snapshot.dropped_counts.get(user).unwrap_or(&0);
                let ip_str = snapshot.user_ips.get(user).cloned().unwrap_or_default();
                let ip_ref = snapshot
                    .user_ips
                    .get(user)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let is_blocked = snapshot.blocked_users.contains(user.as_str())
                    || snapshot.blocked_ips.contains(ip_ref);
                let is_vip = snapshot.vip_list.contains(user);

                let (sym, style) = if is_blocked {
                    ("✖ ", Style::default().fg(Color::Red))
                } else if is_vip {
                    ("★ ", Style::default().fg(Color::Magenta))
                } else if *snapshot.processing_counts.get(user).unwrap_or(&0) > 0 {
                    ("▶ ", Style::default().fg(Color::Cyan))
                } else if *snapshot.queues_len.get(user).unwrap_or(&0) > 0 {
                    ("● ", Style::default().fg(Color::Green))
                } else {
                    ("○ ", Style::default().fg(Color::DarkGray))
                };

                let user_style = if is_blocked {
                    Style::default()
                        .fg(Color::Red)
                        .add_modifier(Modifier::CROSSED_OUT)
                } else if is_vip {
                    Style::default().fg(Color::Magenta).bold()
                } else {
                    Style::default().fg(Color::White)
                };

                let mut spans = vec![
                    Span::styled(sym, style),
                    Span::styled(user.clone(), user_style),
                ];
                if is_vip {
                    spans.push(Span::styled(
                        " [VIP]",
                        Style::default().fg(Color::Magenta).bold(),
                    ));
                }
                if is_blocked {
                    spans.push(Span::styled(
                        " [BLOCKED]",
                        Style::default().fg(Color::Red).bold(),
                    ));
                }

                Row::new(vec![
                    Cell::from(Line::from(spans)),
                    Cell::from(ip_str).style(Style::default().fg(Color::Cyan)),
                    Cell::from(queue_len.to_string()),
                    Cell::from(processed.to_string()),
                    Cell::from(dropped.to_string()),
                ])
            })
            .collect();

        Table::new(
            rows,
            [
                Constraint::Percentage(45),
                Constraint::Percentage(25),
                Constraint::Percentage(10),
                Constraint::Percentage(10),
                Constraint::Percentage(10),
            ],
        )
        .header(
            Row::new(vec!["User ID", "Last IP", "Q", "Done", "Drop"])
                .style(Style::default().fg(Color::Yellow).bold())
                .bottom_margin(1),
        )
        .row_highlight_style(
            Style::default()
                .bg(Color::Rgb(40, 40, 40))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ")
        .block(
            Block::default()
                .title(" Active Users ")
                .borders(Borders::ALL)
                .border_style(if self.active_panel == CurrentPanel::Users {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                }),
        )
    }

    fn render_queues(&self, snapshot: &DashboardSnapshot, available_width: u16) -> Table<'static> {
        let total_queued = snapshot.queues_len.values().sum::<usize>()
            + snapshot.processing_counts.values().sum::<usize>();
        let bar_max_width = ((available_width as f32) * 0.45) as usize;

        let rows: Vec<Row> = snapshot
            .user_ids
            .iter()
            .map(|user| {
                let q_len = snapshot.queues_len.get(user).unwrap_or(&0)
                    + snapshot.processing_counts.get(user).unwrap_or(&0);
                let bar_len = if q_len > 0 {
                    ((q_len as f32 / 20.0).min(1.0) * bar_max_width as f32) as usize
                } else {
                    0
                };
                let color = if snapshot.vip_list.contains(user) {
                    Color::Magenta
                } else if *snapshot.processing_counts.get(user).unwrap_or(&0) > 0 {
                    Color::Cyan
                } else {
                    Color::Green
                };
                let bar = format!("{:<width$}", "⠿".repeat(bar_len), width = bar_max_width);
                let pct = if total_queued > 0 {
                    (q_len as f64 / total_queued as f64) * 100.0
                } else {
                    0.0
                };
                Row::new(vec![
                    Cell::from(user.clone()),
                    Cell::from(bar).style(Style::default().fg(color)),
                    Cell::from(format!("{} ({:.0}%)", q_len, pct))
                        .style(Style::default().fg(color).bold()),
                ])
            })
            .collect();

        Table::new(
            rows,
            [
                Constraint::Percentage(30),
                Constraint::Percentage(45),
                Constraint::Percentage(25),
            ],
        )
        .header(
            Row::new(vec!["User ID", "Progress", "Num"])
                .style(Style::default().fg(Color::Yellow).bold())
                .bottom_margin(1),
        )
        .row_highlight_style(
            Style::default()
                .bg(Color::Rgb(40, 40, 40))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ")
        .block(
            Block::default()
                .title(" Queue Status ")
                .borders(Borders::ALL),
        )
    }

    fn render_blocked(&self, snapshot: &DashboardSnapshot) -> Table<'static> {
        let mut items = Vec::new();
        for ip in snapshot.blocked_ips.iter() {
            items.push(("IP", ip.clone()));
        }
        for user in snapshot.blocked_users.iter() {
            items.push(("USER", user.clone()));
        }
        items.sort_by(|a, b| a.1.cmp(&b.1));

        let rows: Vec<Row> = items
            .iter()
            .map(|(kind, val)| {
                Row::new(vec![
                    Cell::from(kind.to_string()).style(if *kind == "IP" {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default().fg(Color::Magenta)
                    }),
                    Cell::from(val.clone()),
                ])
            })
            .collect();

        Table::new(
            rows,
            [Constraint::Percentage(30), Constraint::Percentage(70)],
        )
        .header(
            Row::new(vec!["Type", "Value"])
                .style(Style::default().fg(Color::Yellow).bold())
                .bottom_margin(1),
        )
        .row_highlight_style(
            Style::default()
                .bg(Color::Rgb(40, 40, 40))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ")
        .block(
            Block::default()
                .title(" Blocked Items ")
                .borders(Borders::ALL)
                .border_style(if self.active_panel == CurrentPanel::Blocked {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                }),
        )
    }

    fn render_logs(&self, snapshot: &DashboardSnapshot, area: Rect) -> Paragraph<'static> {
        let logs: Vec<Line> = snapshot
            .log_lines
            .iter()
            .map(|(level, ts, msg)| {
                let color = match level.as_str() {
                    "DEBUG" => Color::Blue,
                    "INFO" => Color::White,
                    "WARN" => Color::Yellow,
                    "ERROR" => Color::Red,
                    _ => Color::Gray,
                };
                let time_str = format!(
                    "{}",
                    chrono::DateTime::from_timestamp(*ts, 0)
                        .unwrap_or(chrono::DateTime::UNIX_EPOCH)
                        .format("%H:%M")
                );
                Line::from(vec![
                    Span::styled(
                        format!("{} ", time_str),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(format!("{:<5} ", level), Style::default().fg(color).bold()),
                    Span::styled(msg.clone(), Style::default().fg(color)),
                ])
            })
            .collect();
        let available_height = area.height.saturating_sub(2);
        let scroll = logs.len().saturating_sub(available_height as usize);
        Paragraph::new(logs)
            .block(
                Block::default()
                    .title(" Log Messages ")
                    .borders(Borders::ALL),
            )
            .style(Style::default().fg(Color::Gray))
            .scroll((scroll as u16, 0))
    }

    fn render_help(&self) -> Paragraph<'static> {
        Paragraph::new(" Tab: Switch | p: VIP | x: Block User | X: Block IP | u: Unblock | m: Models | q: Quit")
            .block(Block::default().borders(Borders::ALL).title_bottom(Line::from(format!(" v{} ", env!("CARGO_PKG_VERSION"))).alignment(Alignment::Right)))
    }

    fn render_detailed_help(&self) -> Paragraph<'static> {
        Paragraph::new("\n  VIP: 'p' | BLOCK: 'x' (User) / 'X' (IP) | UNBLOCK: 'u'\n  PANELS: 'Tab' | MODELS: 'm' | QUIT: 'q' or 'Esc'\n\n  ★ VIP | ✖ Blocked | ▶ Processing | ● Queued").block(Block::default().title(" Help ").borders(Borders::ALL)).style(Style::default().fg(Color::Gray))
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to Unix socket for dashboard
    #[arg(short, long, default_value = "/run/all-llama-proxy.sock")]
    socket: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let socket_path = args.socket;

    if !std::path::Path::new(&socket_path).exists() {
        eprintln!("Dashboard socket not found at '{}'.", socket_path);
        eprintln!("Make sure all-llama-proxy is running and the socket exists.");
        eprintln!("Run 'sudo systemctl status all-llama-proxy' to check the service.");
        return;
    }

    let stream = match UnixStream::connect(&socket_path).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "Failed to connect to dashboard socket at '{}': {}",
                socket_path, e
            );
            eprintln!("Is the proxy running? Check with: sudo systemctl status all-llama-proxy");
            eprintln!(
                "Is the socket owned by you or in the right group? Run 'ls -l {}'",
                socket_path
            );
            return;
        }
    };
    info!("Connected to dashboard at {}", socket_path);

    let (wire_reader, mut wire_writer) = stream.into_split();
    let (snap_tx, mut snap_rx) = mpsc::channel::<DashboardSnapshot>(64);
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<DashboardCmd>(64);
    // Separate channel just for key events — fed by a dedicated OS thread so
    // event::read() never blocks the async runtime.
    let (key_tx, mut key_rx) = mpsc::channel::<KeyEvent>(32);

    let reader_socket_path = socket_path.clone();
    tokio::spawn(async move {
        match run_reader(wire_reader, snap_tx).await {
            Ok(()) => {}
            Err(e) => eprintln!("Reader error: {}", e),
        }
        let _ = std::fs::remove_file(&reader_socket_path);
    });

    // Dedicated OS thread for blocking key reads.
    // Exits automatically when key_tx is dropped (main loop ended).
    std::thread::spawn(move || {
        loop {
            match event::read() {
                Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                    if key_tx.blocking_send(key).is_err() {
                        break; // main loop exited, receiver dropped
                    }
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    let mut dashboard = TuiDashboard::new();
    let mut state = DashboardSnapshot {
        queues_len: HashMap::new(),
        processing_counts: HashMap::new(),
        processed_counts: HashMap::new(),
        dropped_counts: HashMap::new(),
        user_ips: HashMap::new(),
        blocked_ips: HashSet::new(),
        blocked_users: HashSet::new(),
        vip_list: Vec::new(),
        user_ids: Vec::new(),
        backends: Vec::new(),
        model_public_names: HashMap::new(),
        log_lines: Vec::new(),
    };
    let mut needs_redraw = true;
    let mut tick = tokio::time::interval(Duration::from_millis(16));

    if let Err(e) = enable_raw_mode() {
        eprintln!("ERROR: Failed to enable raw mode: {}", e);
        return;
    }

    if let Err(e) = io::stdout().execute(EnterAlternateScreen) {
        eprintln!("ERROR: Failed to enter alternate screen: {}", e);
        disable_raw_mode().ok();
        return;
    }

    let mut terminal = match Terminal::new(CrosstermBackend::<io::Stdout>::new(io::stdout())) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize terminal: {}", e);
            disable_raw_mode().ok();
            return;
        }
    };

    loop {
        tokio::select! {
            // Arm 1: keypress — handle immediately, no poll delay
            key = key_rx.recv() => {
                let key = match key {
                    Some(k) => k,
                    None => break, // event thread exited
                };
                needs_redraw = true;
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') => {
                        io::stdout().execute(LeaveAlternateScreen).ok();
                        disable_raw_mode().ok();
                        terminal.show_cursor().ok();
                        return;
                    }
                    KeyCode::Char('?') => dashboard.show_help = !dashboard.show_help,
                    KeyCode::Char('m') => dashboard.show_models = !dashboard.show_models,
                    KeyCode::Tab | KeyCode::Char('l') | KeyCode::Char('h') => {
                        dashboard.active_panel = match dashboard.active_panel {
                            CurrentPanel::Users => CurrentPanel::Blocked,
                            CurrentPanel::Blocked => CurrentPanel::Users,
                        };
                    }
                    KeyCode::Char('p') => {
                        if dashboard.active_panel == CurrentPanel::Users {
                            if let Some(i) = dashboard.table_state.selected() {
                                if i < state.user_ids.len() {
                                    let user_id = state.user_ids[i].clone();
                                    let user_id_clone = user_id.clone();
                                    if cmd_tx.send(DashboardCmd::ToggleVip(user_id)).await.is_err() {
                                        break;
                                    }
                                    let mut vip_list = state.vip_list.clone();
                                    if vip_list.contains(&user_id_clone) {
                                        vip_list.retain(|u| u != &user_id_clone);
                                    } else {
                                        vip_list.push(user_id_clone);
                                    }
                                    state.vip_list = vip_list;
                                }
                            }
                        }
                    }
                    KeyCode::Char('x') => {
                        if dashboard.active_panel == CurrentPanel::Users {
                            if let Some(i) = dashboard.table_state.selected() {
                                if i < state.user_ids.len() {
                                    let user_id = state.user_ids[i].clone();
                                    let _ = cmd_tx.send(DashboardCmd::BlockUser(user_id)).await;
                                }
                            }
                        }
                    }
                    KeyCode::Char('X') => {
                        if dashboard.active_panel == CurrentPanel::Users {
                            if let Some(i) = dashboard.table_state.selected() {
                                if i < state.user_ids.len() {
                                    let user_id = &state.user_ids[i];
                                    if let Some(ip) = state.user_ips.get(user_id) {
                                        let _ = cmd_tx.send(DashboardCmd::BlockIp(ip.clone())).await;
                                    }
                                }
                            }
                        }
                    }
                    KeyCode::Char('u') => {
                        if dashboard.active_panel == CurrentPanel::Blocked {
                            if let Some(i) = dashboard.blocked_table_state.selected() {
                                let mut items = Vec::new();
                                for ip in state.blocked_ips.iter() {
                                    items.push(("IP", ip.to_string()));
                                }
                                for user in state.blocked_users.iter() {
                                    items.push(("USER", user.clone()));
                                }
                                items.sort_by(|a, b| a.1.cmp(&b.1));
                                if i < items.len() {
                                    let (kind, value) = &items[i];
                                    if *kind == "IP" {
                                        let _ = cmd_tx.send(DashboardCmd::UnblockIp(value.clone())).await;
                                    } else {
                                        let _ = cmd_tx.send(DashboardCmd::UnblockUser(value.clone())).await;
                                    }
                                }
                            }
                        } else if dashboard.active_panel == CurrentPanel::Users {
                            if let Some(i) = dashboard.table_state.selected() {
                                if i < state.user_ids.len() {
                                    let user_id = &state.user_ids[i];
                                    if let Some(ip) = state.user_ips.get(user_id) {
                                        let _ = cmd_tx.send(DashboardCmd::UnblockIp(ip.clone())).await;
                                    }
                                    let _ = cmd_tx.send(DashboardCmd::UnblockUser(user_id.clone())).await;
                                }
                            }
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if dashboard.active_panel == CurrentPanel::Users {
                            let i = dashboard.table_state.selected().unwrap_or(0).saturating_sub(1);
                            dashboard.table_state.select(Some(i));
                        } else {
                            let i = dashboard.blocked_table_state.selected().unwrap_or(0).saturating_sub(1);
                            dashboard.blocked_table_state.select(Some(i));
                        }
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if dashboard.active_panel == CurrentPanel::Users {
                            let len = state.user_ids.len();
                            if len > 0 {
                                let i = dashboard.table_state.selected()
                                    .map(|s| (s + 1).min(len.saturating_sub(1)))
                                    .unwrap_or(0);
                                dashboard.table_state.select(Some(i));
                            }
                        } else {
                            let len = state.blocked_ips.len() + state.blocked_users.len();
                            if len > 0 {
                                let i = dashboard.blocked_table_state.selected()
                                    .map(|s| (s + 1).min(len.saturating_sub(1)))
                                    .unwrap_or(0);
                                dashboard.blocked_table_state.select(Some(i));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Arm 2: new snapshot from server
            snap = snap_rx.recv() => {
                match snap {
                    Some(s) => { state = s; needs_redraw = true; }
                    None => break, // server disconnected
                }
            }

            // Arm 3: render tick (~60 fps) — only draw when something changed
            _ = tick.tick() => {
                if needs_redraw {
                    if terminal.draw(|f| dashboard.render(f, &state)).is_err() {
                        eprintln!("ERROR: Failed to render terminal UI");
                        break;
                    }
                    needs_redraw = false;
                }
            }

            // Arm 4: outgoing command to server
            Some(cmd) = cmd_rx.recv() => {
                let buf = match encode(&cmd) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("ERROR: Failed to encode command: {}", e);
                        break;
                    }
                };
                if wire_writer.write_all(&buf).await.is_err() {
                    eprintln!("Failed to send command");
                    break;
                }
            }
        }
    }

    cleanup_terminal();
}

fn cleanup_terminal() {
    io::stdout().execute(LeaveAlternateScreen).ok();
    crossterm::terminal::disable_raw_mode().ok();
    io::stdout().execute(crossterm::cursor::Show).ok();
}

async fn run_reader(
    mut socket: tokio::net::unix::OwnedReadHalf,
    tx: mpsc::Sender<DashboardSnapshot>,
) -> io::Result<()> {
    let mut buf = vec![0u8; 8192];
    let mut read_buf = Vec::new();

    loop {
        let n = socket.read(&mut buf[..]).await?;
        if n == 0 {
            return Ok(());
        }

        read_buf.extend_from_slice(&buf[..n]);

        loop {
            if read_buf.len() < 4 {
                break;
            }
            match decode::<DashboardSnapshot>(&read_buf) {
                Ok(Some(snap)) => {
                    // decode() succeeded, so the header is in bounds and the
                    // payload is fully buffered.
                    match consumed_len(&read_buf)? {
                        Some(consumed) => {
                            read_buf.drain(..consumed);
                            let _ = tx.send(snap).await;
                        }
                        None => break,
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    read_buf.clear();
                    break;
                }
            }
        }
    }
}
