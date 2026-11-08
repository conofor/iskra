// ============================================================================
//  iskra — a minimal AI coding agent for the terminal (TTY)
// ============================================================================
//
//  An OpenCode / Claude Code alternative built for weak and old hardware
//  (a 32-bit i686 CPU is enough) and for working from a plain Linux text
//  console without a GUI.
//
//  Key performance decisions:
//    * single-threaded async runtime tokio (flavor = "current_thread") —
//      no multi-threaded scheduler and no extra threads;
//    * minimal crates: reqwest (HTTP) + serde/serde_json (JSON) + tokio;
//      interactive input is plain std::io, no TUI libraries;
//    * pure-Rust TLS (rustls + webpki-roots) — no system OpenSSL needed,
//      root certificates are embedded in the binary;
//    * release build opt-level="s" + lto + strip (see Cargo.toml);
//    * project scanning with hard limits — the context is built in a single
//      pass and never exceeds a few dozen kilobytes.
//
//  How it works:
//    1. On startup the app asks for permission to read/modify files in the
//       current directory (y/n). A refusal exits immediately.
//    2. Configuration lives in ~/.config/iskra/config.json (the name comes
//       from Cargo.toml): MULTIPLE providers (any OpenAI-compatible APIs),
//       each with its own list of models. Editor: /config, quick model
//       switching: /models. An old single-provider config migrates
//       automatically.
//    3. Main loop: the user types a request → the app rescans the current
//       directory (text files only, no .git, node_modules or other heavy
//       folders) and adds file contents to the system prompt → the request
//       goes to the API and the reply is printed as a stream.
//    4. The model reply is parsed: `file:` blocks (overwrite a file) and
//       `run` blocks (run a command) are applied after y/n confirmation —
//       or immediately in autopilot mode (/auto on or --yolo; dangerous
//       commands are blocked there so the user can walk away safely).
//    5. /compact compresses the conversation history into a summary; when
//       the history overflows (> 24 KB), compaction runs automatically
//       before the next request.
//
//  Reply format of the model (protocol, also described in the system prompt):
//    ```file:path/to/file   → full new file content
//    ```run                 → a command for the terminal
//    ```any_language        → ordinary code, display only
//
// ============================================================================

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, Write};
use std::mem;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;

// ---------------------------------------------------------------------------
// Constants and limits
// ---------------------------------------------------------------------------

/// Version, taken from Cargo.toml at compile time.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// App name = package name from Cargo.toml: substituted into the banner,
/// help text, User-Agent and the config path (~/.config/<name>/). Renaming
/// the app is a single `name = "..."` line in Cargo.toml — the code picks
/// it up automatically.
const APP_NAME: &str = env!("CARGO_PKG_NAME");
/// The old app name: only needed to migrate the config once from
/// ~/.config/opencode/ to ~/.config/<APP_NAME>/ when upgrading.
const LEGACY_APP_NAME: &str = "opencode";

/// Overall HTTP timeout for API requests: on slow hardware local models
/// may take minutes to generate an answer.
const HTTP_TIMEOUT_SECS: u64 = 600;
/// Fail fast if the server is unreachable.
const HTTP_CONNECT_TIMEOUT_SECS: u64 = 30;

// --- Project context limits (tuned for small local models) ---
/// Maximum number of files in the project tree listing.
const MAX_TREE_FILES: usize = 300;
/// Maximum number of files whose contents go into the system prompt.
const MAX_FILES_WITH_CONTENT: usize = 100;
/// Files larger than this are not included at all (tree listing only).
const MAX_FILE_BYTES: u64 = 64 * 1024;
/// Files larger than this are included in a truncated form.
const MAX_FILE_SNIPPET: usize = 6 * 1024;
/// Total budget for file contents in the system prompt.
const MAX_TOTAL_CONTEXT: usize = 32 * 1024;
/// Maximum directory traversal depth.
const MAX_SCAN_DEPTH: usize = 10;
/// How many bytes we read to decide "text or binary".
const PROBE_BYTES: usize = 1024;

// --- Conversation history limits (keeps the request context small) ---------
const MAX_HISTORY_MESSAGES: usize = 12;
const MAX_HISTORY_ITEM_BYTES: usize = 3000;

/// Auto-compaction threshold for the conversation history (bytes): when
/// exceeded, the history is compressed into a summary before the next
/// request — important in autopilot mode where the agent may run long and
/// unattended.
const AUTO_COMPACT_BYTES: usize = 24 * 1024;

/// Instruction for the model to compress the history (/compact and
/// auto-compaction).
const COMPACT_PROMPT: &str = "Summarize the conversation history with the AI agent into a brief summary so work can continue. List: which files were changed (paths), which commands were run, the key decisions and the current unfinished task. Return only the summary, briefly.";

/// Heuristic list of dangerous commands: in autopilot mode such commands are
/// NOT executed (safer than asking — the user may be away from the computer).
/// The list is not applied in manual mode, where every action is confirmed.
const DANGEROUS_CMD: &[&str] = &[
    "rm -rf", "rm -fr", "rm -r /", "mkfs", "dd if=", "of=/dev/",
    "shutdown", "poweroff", "reboot", "halt", "init 0", "init 6",
    "fdisk", "parted", ":(){", "chmod -R 777 /", "history -c",
    "| sh", "|sh", "| bash", "|bash", "> /dev/sd", "eval $(curl", "eval $(wget",
];

/// Folders never scanned: heavy and free of sources.
const SKIP_DIRS: &[&str] = &[
    ".git", ".svn", ".hg", ".bzr",
    "node_modules", "vendor", "target", "build", "dist", "out", "obj",
    "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", ".cache", ".gradle", ".next", ".nuxt", "coverage",
];

/// Extensions we do not even try to read as text (saves IO).
const BINARY_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "pdf", "zip", "gz",
    "bz2", "xz", "7z", "rar", "tar", "exe", "dll", "so", "a", "o", "obj",
    "bin", "iso", "img", "mp3", "mp4", "avi", "mkv", "mov", "wav", "flac",
    "ogg", "woff", "woff2", "ttf", "otf", "eot", "class", "jar", "pyc",
    "wasm", "db", "sqlite", "sqlite3", "deb", "rpm", "snap",
];

/// Known language names in code block headers (```rust etc.).
/// Used to tell ```rust apart from ```src/main.rs (a file path).
const KNOWN_LANGS: &[&str] = &[
    "rust", "rs", "python", "py", "javascript", "js", "jsx", "typescript",
    "ts", "tsx", "go", "golang", "c", "cpp", "c++", "h", "hpp", "java",
    "kotlin", "swift", "ruby", "rb", "php", "bash", "sh", "shell", "zsh",
    "fish", "console", "terminal", "plaintext", "text", "txt", "json",
    "yaml", "yml", "toml", "xml", "html", "css", "scss", "less", "sql",
    "md", "markdown", "diff", "patch", "ini", "lua", "perl", "haskell",
    "scala", "dart", "r", "dockerfile", "makefile", "cmake", "mermaid",
];

// --- ANSI color codes (disabled when TERM is empty or NO_COLOR is set) -----
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD: &str = "\x1b[1m";
const ANSI_DIM: &str = "\x1b[2m";
const ANSI_RED: &str = "\x1b[31m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_CYAN: &str = "\x1b[36m";

// ---------------------------------------------------------------------------
// Terminal helpers: colors, input reading, confirmations
// ---------------------------------------------------------------------------

/// Whether ANSI colors are enabled. Determined once per run.
fn colors_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if env::var_os("NO_COLOR").is_some() {
            return false;
        }
        match env::var("TERM") {
            // Old TTYs (linux console) do support ANSI codes; "dumb" and a
            // missing TERM mean no support. In a pipe TERM is usually unset
            // → colors off.
            Ok(t) => t != "dumb",
            Err(_) => false,
        }
    })
}

/// Returns the ANSI code if colors are enabled, otherwise an empty string.
fn col(code: &'static str) -> &'static str {
    if colors_enabled() { code } else { "" }
}

/// Prints text in color and resets attributes (no newline).
fn paint(code: &'static str, text: &str) {
    print!("{}{}{}", col(code), text, col(ANSI_RESET));
    let _ = io::stdout().flush();
}

/// Same, but with a newline.
fn paintln(code: &'static str, text: &str) {
    println!("{}{}{}", col(code), text, col(ANSI_RESET));
}

/// Reads a line from stdin as raw bytes (up to '\n' or EOF) and decodes it.
/// Unlike Stdin::read_line it does not fail on non-UTF-8 input (old Russian
/// terminals send CP1251/KOI8-R) and handles Ctrl+D without a newline
/// correctly. `None` — the input stream is closed (EOF).
fn read_line() -> Option<String> {
    let mut buf: Vec<u8> = Vec::new();
    let n = match io::stdin().lock().read_until(b'\n', &mut buf) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Input read error: {e}");
            return None;
        }
    };
    if n == 0 {
        return None; // EOF
    }
    // Strip trailing '\n' and '\r' (including Windows-style CRLF).
    while let Some(&last) = buf.last() {
        if last == b'\n' || last == b'\r' {
            buf.pop();
        } else {
            break;
        }
    }
    Some(decode_input(&buf))
}

/// CP1251 → Unicode conversion table for bytes 0x80–0xFF (index = byte − 0x80).
/// Position 0x98 is undefined in CP1251 → U+FFFD.
const CP1251_HIGH: [char; 128] = [
    'Ђ', 'Ѓ', '‚', 'ѓ', '„', '…', '†', '‡', '€', '‰', 'Љ', '‹', 'Њ', 'Ќ', 'Ћ', 'Џ', 'ђ', '‘', 
    '’', '“', '”', '•', '–', '—', '\u{FFFD}', '™', 'љ', '›', 'њ', 'ќ', 'ћ', 'џ', '\u{00A0}', 'Ў', 'ў', 
    'Ј', '¤', 'Ґ', '¦', '§', 'Ё', '©', 'Є', '«', '¬', '\u{00AD}', '®', 'Ї', '°', '±', 'І', 'і', 
    'ґ', 'µ', '¶', '·', 'ё', '№', 'є', '»', 'ј', 'Ѕ', 'ѕ', 'ї', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 
    'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 
    'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 
    'ь', 'э', 'ю', 'я'
];

/// Decodes raw input bytes: strict UTF-8 first; if the input is not UTF-8,
/// decodes it as CP1251 (the common encoding of old Russian consoles) and
/// warns once per session. The app never crashes on input.
fn decode_input(bytes: &[u8]) -> String {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        paintln(ANSI_YELLOW, "Note: input is not valid UTF-8 — decoding as CP1251.");
        println!("  (if the terminal uses a different 8-bit encoding, e.g. KOI8-R, the text may look garbled)");
    }
    bytes
        .iter()
        .map(|&b| if b < 0x80 { b as char } else { CP1251_HIGH[(b - 0x80) as usize] })
        .collect()
}

/// Asks a y/n question. An empty answer and EOF are treated as "no"
/// (the safe default).
fn confirm(question: &str) -> bool {
    loop {
        paint(ANSI_YELLOW, question);
        print!(" (y/n): ");
        let _ = io::stdout().flush();
        let Some(answer) = read_line() else { return false };
        match answer.trim().to_lowercase().as_str() {
            "y" | "yes" => return true,
            "n" | "no" | "" => return false,
            _ => println!("Enter y (yes) or n (no)."),
        }
    }
}

/// Prints a label and reads the answer line.
fn prompt_line(label: &str) -> Option<String> {
    paint(ANSI_CYAN, label);
    print!(" ");
    let _ = io::stdout().flush();
    read_line()
}

/// Trims a string at a byte limit without breaking UTF-8 characters.
fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut cut = max_bytes;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    &s[..cut]
}

// ---------------------------------------------------------------------------
// Provider configuration: ~/.config/iskra/config.json
// ---------------------------------------------------------------------------

/// A provider: one OpenAI-compatible API and its list of models.
#[derive(Serialize, Deserialize)]
struct Provider {
    /// Short name for menus, e.g. nvidia, ollama, api-openai.
    name: String,
    /// API base, e.g. http://localhost:11434/v1 (Ollama)
    /// or https://api.openai.com/v1.
    base_url: String,
    /// API key; usually empty for local models.
    api_key: String,
    /// Models the user added for this provider.
    models: Vec<String>,
}

/// Configuration: several providers, each with several models.
#[derive(Serialize, Deserialize)]
struct Config {
    providers: Vec<Provider>,
    /// Index of the active provider.
    active_provider: usize,
    /// Active model (must be in the active provider's model list).
    active_model: String,
}

impl Config {
    /// Active provider and model (guarded against broken indices).
    fn active(&self) -> (&Provider, &str) {
        let idx = self.active_provider.min(self.providers.len().saturating_sub(1));
        let p = &self.providers[idx];
        (p, &self.active_model)
    }

    /// Fixes broken indices (e.g. after a manual JSON edit).
    fn normalize(&mut self) {
        if self.active_provider >= self.providers.len() {
            self.active_provider = 0;
        }
        let p = &self.providers[self.active_provider];
        if !p.models.contains(&self.active_model) {
            self.active_model = p.models[0].clone();
        }
    }
}

/// Host name from a URL — used as the default provider name.
fn host_of(url: &str) -> String {
    let s = url.trim();
    let s = s
        .strip_prefix("https://")
        .or_else(|| s.strip_prefix("http://"))
        .unwrap_or(s);
    let s = s.split('/').next().unwrap_or("");
    let s = s.split(':').next().unwrap_or(s); // strip the port
    s.to_string()
}

/// Config parsing: the new format (providers) or migration of the old
/// single-provider format {base_url, api_key, model} → one provider.
fn config_from_value(v: &serde_json::Value) -> Option<Config> {
    if v.get("providers").is_some() {
        let cfg: Config = serde_json::from_value(v.clone()).ok()?;
        if cfg.providers.is_empty() || cfg.providers.iter().any(|p| p.models.is_empty()) {
            return None;
        }
        return Some(cfg);
    }
    // --- old format ---
    let base_url = v.get("base_url")?.as_str()?.trim().to_string();
    let model = v.get("model")?.as_str()?.trim().to_string();
    if base_url.is_empty() || model.is_empty() {
        return None;
    }
    let api_key = v
        .get("api_key")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let host = host_of(&base_url);
    let name = if host.is_empty() { "provider-1".to_string() } else { host };
    Some(Config {
        active_provider: 0,
        active_model: model.clone(),
        providers: vec![Provider {
            name,
            base_url,
            api_key,
            models: vec![model],
        }],
    })
}

/// Config path: $XDG_CONFIG_HOME/<app-name>/config.json, by default
/// ~/.config/<app-name>/config.json. If HOME is unset the file goes into
/// the current directory (a last-resort fallback for exotic systems).
fn config_path() -> PathBuf {
    let base = env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| env::var("HOME").ok().map(|h| format!("{h}/.config")))
        .unwrap_or_else(|| ".".to_string());
    PathBuf::from(base).join(APP_NAME).join("config.json")
}

/// Config in the OLD app-name folder (~/.config/opencode/config.json).
/// Only used for a one-time automatic migration on upgrade.
fn legacy_config_path() -> PathBuf {
    let base = env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| env::var("HOME").ok().map(|h| format!("{h}/.config")))
        .unwrap_or_else(|| ".".to_string());
    PathBuf::from(base).join(LEGACY_APP_NAME).join("config.json")
}

/// Short key masking for menus (first 8 characters).
fn mask_key(key: &str) -> String {
    if key.is_empty() {
        "<empty>".to_string()
    } else {
        format!("{}…", key.chars().take(8).collect::<String>())
    }
}

/// Interactive provider creation: Base URL, key, name, list of models.
fn prompt_provider(default_name: &str) -> Provider {
    println!();
    println!("— New provider (any OpenAI-compatible API) —");
    println!("Base URL examples: http://localhost:11434/v1 (Ollama), http://localhost:8080/v1 (llama.cpp), https://api.openai.com/v1");

    let base_url = loop {
        let Some(s) = prompt_line("Base URL:") else {
            println!("Input interrupted (EOF). Exiting.");
            std::process::exit(1);
        };
        let s = s.trim().to_string();
        if s.is_empty() {
            println!("  Base URL cannot be empty.");
            continue;
        }
        // No scheme given — assume a local server and add http://.
        break if s.starts_with("http://") || s.starts_with("https://") {
            s
        } else {
            format!("http://{s}")
        };
    };

    let api_key = prompt_line("API key (Enter to leave empty, e.g. for local models):")
        .unwrap_or_default();

    let host = host_of(&base_url);
    let entered_name = prompt_line(&format!("Provider name [{host}]:")).unwrap_or_default();
    let entered_name = entered_name.trim().to_string();
    let name = if entered_name.is_empty() {
        if host.is_empty() { default_name.to_string() } else { host }
    } else {
        entered_name
    };

    // At least one model is required; more until an empty line.
    let mut models: Vec<String> = Vec::new();
    println!("Enter the models of this provider (an empty line finishes):");
    loop {
        let label = if models.is_empty() {
            "Model (e.g. llama3, qwen2.5:1.5b, gpt-4o):"
        } else {
            "Another model:"
        };
        let Some(s) = prompt_line(label) else {
            println!("Input interrupted (EOF). Exiting.");
            std::process::exit(1);
        };
        let m = s.trim().to_string();
        if m.is_empty() {
            if models.is_empty() {
                println!("  At least one model is required.");
                continue;
            }
            break;
        }
        if models.contains(&m) {
            println!("  This model is already added.");
            continue;
        }
        models.push(m);
    }

    Provider {
        name,
        base_url,
        api_key: api_key.trim().to_string(),
        models,
    }
}

/// First run: create the initial provider. Others can be added later via
/// /config, models switched via /models.
fn prompt_config() -> Config {
    println!();
    println!("Set up the connection to your AI backend (any OpenAI-compatible API).");
    println!("You can add more providers later (/config) and switch models (/models).");
    let provider = prompt_provider("provider-1");
    let active_model = provider.models[0].clone();
    Config {
        active_provider: 0,
        active_model,
        providers: vec![provider],
    }
}

/// Saves the config as pretty JSON (so the file can be edited by hand).
fn save_config(cfg: &Config) -> Result<(), String> {
    let path = config_path();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(cfg)
        .map_err(|e| format!("serialization: {e}"))?;
    fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Loads the config (with old-format migration); if it is missing or damaged,
/// runs the interactive setup. The app does not crash: without a save we
/// keep working with the in-memory settings.
fn load_config() -> Config {
    let mut path = config_path();
    if !path.exists() {
        // Upgrade from an old version of the app: the config lived in the
        // old-name folder (~/.config/opencode/) — move it once.
        let legacy = legacy_config_path();
        if legacy.exists() {
            // The new-name folder may not exist yet — without create_dir_all
            // the copy would fail and the config would stay at the old place.
            if let Some(dir) = path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            match fs::copy(&legacy, &path) {
                Ok(_) => paintln(
                    ANSI_DIM,
                    &format!("Config migrated: {} → {}", legacy.display(), path.display()),
                ),
                // No write permission for the new folder — read from the old place.
                Err(_) => path = legacy,
            }
        }
    }
    if let Ok(raw) = fs::read_to_string(&path) {
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(v) => match config_from_value(&v) {
                Some(mut cfg) => {
                    cfg.normalize();
                    paintln(
                        ANSI_DIM,
                        &format!(
                            "Configuration: {} provider(s); model {} @ {}",
                            cfg.providers.len(),
                            cfg.active_model,
                            cfg.providers[cfg.active_provider].base_url
                        ),
                    );
                    return cfg;
                }
                None => paintln(ANSI_YELLOW, "The config file is damaged or incomplete — let's set it up again."),
            },
            Err(_) => paintln(ANSI_YELLOW, "The config file is damaged or incomplete — let's set it up again."),
        }
    }
    let cfg = prompt_config();
    match save_config(&cfg) {
        Ok(()) => println!("Configuration saved: {}", path.display()),
        Err(e) => paintln(ANSI_RED, &format!("Failed to save the config: {e} (continuing with in-memory settings)")),
    }
    cfg
}

/// Asks for a number in the range 1..=max; None — cancelled (empty input/EOF).
fn prompt_index(label: &str, max: usize) -> Option<usize> {
    let Some(s) = prompt_line(label) else { return None };
    let s = s.trim().to_string();
    if s.is_empty() {
        return None;
    }
    match s.parse::<usize>() {
        Ok(n) if n >= 1 && n <= max => Some(n - 1),
        _ => {
            paintln(ANSI_YELLOW, &format!("Enter a number from 1 to {max}."));
            None
        }
    }
}

/// The /models menu: a flat list of all models of all providers; entering
/// a number switches the active model (and provider).
fn models_menu(cfg: &mut Config) {
    let mut all: Vec<(usize, String)> = Vec::new();
    for (i, p) in cfg.providers.iter().enumerate() {
        for m in &p.models {
            all.push((i, m.clone()));
        }
    }
    if all.is_empty() {
        paintln(ANSI_YELLOW, "No models — add them via /config.");
        return;
    }
    println!("Available models:");
    for (n, (pi, m)) in all.iter().enumerate() {
        let cur = *pi == cfg.active_provider && *m == cfg.active_model;
        let mark = if cur { "  ← current" } else { "" };
        println!(" {}. {} — {} ({}){}", n + 1, cfg.providers[*pi].name, m, cfg.providers[*pi].base_url, mark);
    }
    if let Some(idx) = prompt_index("Model number (Enter to cancel): ", all.len()) {
        let (pi, m) = all[idx].clone();
        cfg.active_provider = pi;
        cfg.active_model = m;
        let p = &cfg.providers[pi];
        paintln(
            ANSI_GREEN,
            &format!("Active model: {} @ {} ({})", cfg.active_model, p.name, p.base_url),
        );
        if let Err(e) = save_config(cfg) {
            paintln(ANSI_RED, &format!("Failed to save the config: {e}"));
        }
    }
}

/// The /config menu: editor for the provider list and their models.
fn config_menu(cfg: &mut Config) {
    loop {
        println!();
        println!("=== Providers and models ===");
        for (i, p) in cfg.providers.iter().enumerate() {
            let mark = if i == cfg.active_provider { "  ← active" } else { "" };
            println!(
                " {}. {} — {} | key: {} | models: {}{}",
                i + 1,
                p.name,
                p.base_url,
                mask_key(&p.api_key),
                p.models.len(),
                mark
            );
            for (j, m) in p.models.iter().enumerate() {
                let cur = i == cfg.active_provider && *m == cfg.active_model;
                let mark = if cur { "  ← current" } else { "" };
                println!("      {j}. {m}{mark}");
            }
        }
        println!();
        println!("[a] add provider      [e] edit provider      [d] remove provider");
        println!("[m] add model         [x] remove model       [q] done");
        let Some(cmd) = prompt_line("[config]:") else { break };
        match cmd.trim().to_lowercase().as_str() {
            "q" | "" => break,
            "a" => {
                let provider = prompt_provider(&format!("provider-{}", cfg.providers.len() + 1));
                cfg.providers.push(provider);
                paintln(ANSI_GREEN, "Provider added (switch via /models).");
            }
            "e" => {
                if let Some(i) = prompt_index("Provider number: ", cfg.providers.len()) {
                    let name = cfg.providers[i].name.clone();
                    println!("Editing provider \"{name}\" (Enter keeps the current value):");
                    let s = prompt_line(&format!("Base URL [{}]:", cfg.providers[i].base_url))
                        .unwrap_or_default();
                    let s = s.trim().to_string();
                    if !s.is_empty() {
                        cfg.providers[i].base_url = s;
                    }
                    let s = prompt_line(&format!(
                        "API key [{}]:",
                        mask_key(&cfg.providers[i].api_key)
                    ))
                    .unwrap_or_default();
                    if !s.trim().is_empty() {
                        cfg.providers[i].api_key = s.trim().to_string();
                    }
                    let s = prompt_line(&format!("Name [{}]:", cfg.providers[i].name))
                        .unwrap_or_default();
                    if !s.trim().is_empty() {
                        cfg.providers[i].name = s.trim().to_string();
                    }
                    paintln(ANSI_GREEN, "Provider updated.");
                }
            }
            "d" => {
                if cfg.providers.len() <= 1 {
                    paintln(ANSI_YELLOW, "Cannot remove the only provider.");
                    continue;
                }
                if let Some(i) = prompt_index("Provider number: ", cfg.providers.len()) {
                    let name = cfg.providers[i].name.clone();
                    if confirm(&format!("Remove provider \"{name}\" with all its models?")) {
                        cfg.providers.remove(i);
                        if cfg.active_provider >= cfg.providers.len() {
                            cfg.active_provider = 0;
                        }
                        cfg.normalize();
                        paintln(ANSI_GREEN, "Provider removed.");
                    }
                }
            }
            "m" => {
                if let Some(i) = prompt_index("Add a model to which provider (number): ", cfg.providers.len()) {
                    if let Some(s) = prompt_line("Model name:") {
                        let m = s.trim().to_string();
                        if m.is_empty() {
                            paintln(ANSI_YELLOW, "Empty name — cancelled.");
                        } else if cfg.providers[i].models.contains(&m) {
                            paintln(ANSI_YELLOW, "That model already exists.");
                        } else {
                            cfg.providers[i].models.push(m);
                            paintln(ANSI_GREEN, "Model added.");
                        }
                    }
                }
            }
            "x" => {
                if let Some(i) = prompt_index("Provider number: ", cfg.providers.len()) {
                    let models = cfg.providers[i].models.clone();
                    if models.len() <= 1 {
                        paintln(ANSI_YELLOW, "Cannot remove the provider's last model.");
                        continue;
                    }
                    for (j, m) in models.iter().enumerate() {
                        println!("  {}. {m}", j + 1);
                    }
                    if let Some(k) = prompt_index("Model number: ", models.len()) {
                        let removed = cfg.providers[i].models.remove(k);
                        if i == cfg.active_provider && cfg.active_model == removed {
                            cfg.active_model = cfg.providers[i].models[0].clone();
                        }
                        paintln(ANSI_GREEN, &format!("Model \"{removed}\" removed."));
                    }
                }
            }
            _ => paintln(ANSI_YELLOW, "Unknown menu command (a/e/d/m/x/q)."),
        }
    }
    match save_config(cfg) {
        Ok(()) => paintln(ANSI_GREEN, &format!("Configuration saved: {}", config_path().display())),
        Err(e) => paintln(ANSI_RED, &format!("Failed to save the config: {e}")),
    }
}

/// Builds the Chat Completions endpoint URL from the Base URL.
/// Standard: the base already contains /v1 (e.g. http://localhost:11434/v1),
/// we append /chat/completions. If the user gave only a host without a path,
/// /v1 is added automatically.
fn endpoint_url(base: &str) -> String {
    let mut b = base.trim().trim_end_matches('/').to_string();
    let after_scheme = b.find("://").map(|i| i + 3).unwrap_or(0);
    if !b[after_scheme..].contains('/') {
        b.push_str("/v1");
    }
    format!("{b}/chat/completions")
}

// ---------------------------------------------------------------------------
// Project scanning: builds the context for the system prompt
// ---------------------------------------------------------------------------

/// Result of scanning the current directory.
struct ProjectScan {
    /// Relative paths of all found files (the tree listing).
    tree: Vec<String>,
    /// Ready-made text with file contents (within the budget).
    contents: String,
    /// How many files made it into contents.
    scanned: usize,
    /// How many files were truncated by the limits.
    truncated: usize,
}

/// Decides by extension that the file is definitely not text — not even opened.
fn skipped_ext(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let e = ext.to_lowercase();
            BINARY_EXTS.iter().any(|b| *b == e)
        }
        None => false,
    }
}

/// Quick "is this text?" check: reads the first kilobyte and looks for
/// binary markers (NUL byte, invalid UTF-8).
fn is_probably_text(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut file) = fs::File::open(path) else { return false };
    let mut buf = [0u8; PROBE_BYTES];
    let Ok(n) = file.read(&mut buf) else { return false };
    let probe = &buf[..n];
    if probe.contains(&0u8) {
        return false; // NUL — almost certainly a binary
    }
    std::str::from_utf8(probe).is_ok() // not UTF-8 — treat as binary
}

/// Walks the current directory: iteratively (no recursion), with a depth
/// limit, skipping symlinks (loop protection) and heavy folders.
fn scan_project(root: &Path) -> ProjectScan {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut stack: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];

    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_symlink() {
                continue; // do not follow links — loop protection
            }
            let path = entry.path();
            if ft.is_dir() {
                if depth >= MAX_SCAN_DEPTH {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if SKIP_DIRS.iter().any(|s| name.as_str() == *s) {
                    continue;
                }
                stack.push((path, depth + 1));
            } else if ft.is_file() {
                found.push(path);
            }
        }
    }

    // Relative paths, sorted for a stable order in the context.
    let mut rel: Vec<(PathBuf, String)> = found
        .into_iter()
        .filter_map(|p| {
            // Compute the name before p is moved into the tuple.
            let rel = p.strip_prefix(root).ok()?;
            let name = rel.to_string_lossy().replace('\\', "/");
            Some((p, name))
        })
        .collect();
    rel.sort_by(|a, b| a.1.cmp(&b.1));

    // --- file contents within the budget ---
    let mut contents = String::new();
    let mut budget = MAX_TOTAL_CONTEXT;
    let mut scanned = 0usize;
    let mut truncated = 0usize;

    for (path, name) in &rel {
        if scanned >= MAX_FILES_WITH_CONTENT || budget < 1024 {
            break;
        }
        if skipped_ext(path) {
            continue;
        }
        let Ok(meta) = fs::metadata(path) else { continue };
        if meta.len() == 0 || meta.len() > MAX_FILE_BYTES {
            continue;
        }
        if !is_probably_text(path) {
            continue;
        }
        let Ok(raw) = fs::read_to_string(path) else { continue };

        let mut body = raw;
        // Trim overly long files (cut on a UTF-8 character boundary).
        if body.len() > MAX_FILE_SNIPPET {
            body = format!(
                "{}\n… (file truncated: showing {} of {} bytes)",
                truncate_utf8(&body, MAX_FILE_SNIPPET),
                MAX_FILE_SNIPPET,
                body.len()
            );
            truncated += 1;
        }
        // If the file does not fit the total budget, trim it further.
        let overhead = name.len() + 40;
        if overhead + body.len() > budget {
            let allowed = budget.saturating_sub(overhead + 96);
            if allowed < 256 {
                break;
            }
            body = format!(
                "{}\n… (truncated: context limit reached)",
                truncate_utf8(&body, allowed)
            );
            truncated += 1;
        }
        budget = budget.saturating_sub(overhead + body.len());
        scanned += 1;

        contents.push_str("--- ");
        contents.push_str(name);
        contents.push_str(" ---\n");
        contents.push_str(&body);
        if !body.ends_with('\n') {
            contents.push('\n');
        }
        contents.push('\n');
    }

    ProjectScan {
        tree: rel.into_iter().map(|(_, n)| n).collect(),
        contents,
        scanned,
        truncated,
    }
}

/// The system prompt: reply protocol rules + the tree and file contents.
fn build_system_prompt(root: &Path, scan: &ProjectScan) -> String {
    let mut s = String::with_capacity(4096 + scan.contents.len());

    // IMPORTANT: the text below contains no { or } characters except the
    // {root} and {APP_NAME} substitutions.
    let _ = write!(
        s,
        "You are an AI programming agent running inside {APP_NAME}, a minimal \
         CLI application in the user's terminal.\n\
         Working directory: {root}\n\
         \n\
         Reply rules (follow strictly):\n\
         1. Briefly explain what you are doing first.\n\
         2. To create or modify a file, return a code block whose first line \
         contains the word file: followed by a relative path after three \
         backticks:\n\
         ```file:src/main.rs\n\
         (full new file content)\n\
         ```\n\
         Provide the COMPLETE file content (not a diff and not a fragment): \
         the file will be fully overwritten after user confirmation. Never use \
         absolute paths or paths containing ..\n\
         3. To suggest a terminal command, return a block:\n\
         ```run\n\
         cargo build\n\
         ```\n\
         4. Format ordinary code examples as regular fenced blocks with the \
         language name (e.g. ```rust) — they are displayed only and change \
         nothing.\n\
         5. Do not suggest dangerous commands (rm -rf, disk formatting, etc.) \
         unless absolutely necessary.\n\
         6. Do not invent contents of files that are not in the context: if a \
         file appears in the tree but its contents are not shown, ask the \
         user.\n",
        root = root.display()
    );

    s.push_str("\n=== PROJECT STRUCTURE (all found files) ===\n");
    if scan.tree.is_empty() {
        s.push_str("(no files found in the directory)\n");
    } else {
        for (i, name) in scan.tree.iter().enumerate() {
            if i >= MAX_TREE_FILES {
                let _ = writeln!(s, "… and {} more files", scan.tree.len() - i);
                break;
            }
            let _ = writeln!(s, "- {name}");
        }
    }

    s.push_str("\n=== FILE CONTENTS ===\n");
    if scan.scanned == 0 {
        s.push_str("(no file contents fit into the context)\n");
    } else {
        s.push_str(&scan.contents);
    }
    s
}

// ---------------------------------------------------------------------------
// API request: POST {base}/chat/completions (OpenAI-compatible format)
// ---------------------------------------------------------------------------

/// Messages of all causes under the top-level error, joined with ": "
/// ("a: b: c"). The top-level message itself is skipped: the caller prints it
/// in its own wording (for reqwest errors we already show the URL, and the
/// generic "error sending request for url (…)" would just repeat it). reqwest
/// hides the real reason — DNS, TCP connect, TLS, timeout — below the top,
/// which is exactly what one needs to see on a console-only machine. Falls
/// back to the plain Display text when the error has no causes.
fn error_chain(e: &impl std::error::Error) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut src = e.source();
    while let Some(err) = src {
        let s = err.to_string();
        if !parts.contains(&s) {
            parts.push(s);
        }
        src = err.source();
    }
    if parts.is_empty() {
        e.to_string()
    } else {
        parts.join(": ")
    }
}

/// Short practical hint for the two most common network-level failures,
/// appended to the error line.
fn net_hint(e: &reqwest::Error) -> Option<&'static str> {
    if e.is_timeout() {
        Some("timed out — the connection is slow or the server is unreachable")
    } else if e.is_connect() {
        Some("no connection — check the network and DNS, and also the system date (`date`): TLS rejects certificates when the clock is wrong")
    } else {
        None
    }
}

/// Waiting phases for the progress indicator.
const PHASE_WAIT: u8 = 0; // the first token has not arrived yet
const PHASE_THINK: u8 = 1; // the model is streaming hidden reasoning

/// One model request with SSE streaming: the reply is printed as it is
/// generated (important on slow reasoning models like gpt-oss — otherwise
/// the user stares at an empty screen for a long time, here they see text
/// immediately). Returns (full text, whether the reply was printed as a
/// stream). All errors are returned as strings: the app must not panic.
async fn chat_completion(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[(String, String)],
) -> Result<(String, bool), String> {
    let url = endpoint_url(base_url);

    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    let payload = json!({
        "model": model,
        "stream": true,
        "messages": msgs,
    });

    let key = api_key.trim();
    let request = if key.is_empty() {
        client.post(url.as_str()).json(&payload)
    } else {
        client.post(url.as_str()).json(&payload).bearer_auth(key)
    };

    let mut response = request.send().await.map_err(|e| {
        let mut msg = format!("API unreachable ({url}): {}", error_chain(&e));
        if let Some(hint) = net_hint(&e) {
            msg.push_str(&format!(" — {hint}"));
        }
        msg
    })?;

    if !response.status().is_success() {
        let status = response.status();
        // Read the body as bytes and decode manually — no dependency on the
        // charset feature.
        let raw = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read the API response: {}", error_chain(&e)))?;
        let body = String::from_utf8_lossy(&raw);
        return Err(format!(
            "API error {status}: {}",
            truncate_utf8(body.trim(), 1500)
        ));
    }

    // --- waiting indicator: "waiting" / "thinking" + seconds ---
    // Works on the single-threaded runtime too: the task runs between
    // await points of the network request.
    let stop = Arc::new(AtomicBool::new(false));
    let phase = Arc::new(AtomicU8::new(PHASE_WAIT));
    let (stop2, phase2) = (stop.clone(), phase.clone());
    let spinner = tokio::spawn(async move {
        let mut ticks: u32 = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if stop2.load(Ordering::Relaxed) {
                break;
            }
            ticks += 1;
            if ticks % 5 == 0 {
                // print once per second
                let label = if phase2.load(Ordering::Relaxed) == PHASE_THINK {
                    "model is thinking"
                } else {
                    "waiting for the model"
                };
                print!("\r{}{}{label}… {}s{}  ", col(ANSI_DIM), col(ANSI_CYAN), ticks / 5, col(ANSI_RESET));
                let _ = io::stdout().flush();
            }
        }
    });

    // --- read the SSE stream (lines "data: {…}") and print text deltas ---
    let mut content = String::new();
    let mut buffer: Vec<u8> = Vec::new();
    let mut printed = false; // has stream printing started
    let mut stream_error: Option<String> = None;

    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                buffer.extend_from_slice(&chunk);
                // Parse only complete lines (up to '\n').
                while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                    let line_bytes: Vec<u8> = buffer.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line_bytes[..line_bytes.len() - 1]);
                    let line = line.trim();
                    if let Some(data) = line.strip_prefix("data:") {
                        let data = data.trim();
                        if data == "[DONE]" {
                            continue;
                        }
                        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                            continue;
                        };
                        let delta = v.pointer("/choices/0/delta");
                        // Visible reply text — print immediately.
                        if let Some(t) = delta
                            .and_then(|d| d.get("content"))
                            .and_then(serde_json::Value::as_str)
                        {
                            if !t.is_empty() {
                                if !printed {
                                    printed = true;
                                    stop.store(true, Ordering::Relaxed); // stop the indicator
                                    print!("\r\x1b[2K"); // erase the indicator line
                                }
                                print!("{t}");
                                let _ = io::stdout().flush();
                                content.push_str(t);
                            }
                        } else if delta.map_or(false, |d| {
                            d.get("reasoning_content")
                                .or_else(|| d.get("reasoning"))
                                .and_then(serde_json::Value::as_str)
                                .map_or(false, |s| !s.is_empty())
                        }) {
                            // A reasoning model "thinks" into a hidden field —
                            // do not print it, only change the indicator label.
                            phase.store(PHASE_THINK, Ordering::Relaxed);
                        }
                    }
                }
            }
            Ok(None) => break, // the stream is closed by the server
            Err(e) => {
                stream_error = Some(if content.is_empty() {
                    format!("Failed to read the response stream: {e}")
                } else {
                    format!("Stream interrupted: {e} (part of the reply was already shown)")
                });
                break;
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let _ = spinner.await;
    if printed {
        println!(); // newline after the streamed reply
    }

    // --- fallback: the server ignored stream=true and returned plain JSON ---
    if content.is_empty() && stream_error.is_none() {
        let body = String::from_utf8_lossy(&buffer);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body.trim()) {
            if let Some(t) = v
                .pointer("/choices/0/message/content")
                .and_then(serde_json::Value::as_str)
            {
                content = t.to_string();
            }
        }
    }

    if let Some(err) = stream_error {
        return Err(err);
    }
    if content.trim().is_empty() {
        return Err("The API returned an empty reply (check the model name and Base URL).".to_string());
    }
    Ok((content, printed))
}

// ---------------------------------------------------------------------------
// Autopilot and context compaction
// ---------------------------------------------------------------------------

/// Heuristic danger check for a command (autopilot mode only).
fn is_dangerous(cmd: &str) -> bool {
    let c = cmd.to_lowercase();
    DANGEROUS_CMD.iter().any(|p| c.contains(p))
}

/// Size of the conversation history in bytes (an estimate for auto-compaction).
fn history_bytes(history: &[(String, String)]) -> usize {
    history.iter().map(|(r, c)| r.len() + c.len() + 8).sum()
}

/// Asks the model to compress the conversation history into a brief summary
/// (/compact and automatic compaction on overflow).
async fn compact_history(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    history: &[(String, String)],
) -> Result<String, String> {
    let mut transcript = String::new();
    for (role, text) in history {
        let _ = writeln!(transcript, "=== {role} ===\n{text}");
    }
    let messages = vec![
        (
            "system".to_string(),
            "You are an assistant that compresses conversation history. Reply with the summary only, briefly."
                .to_string(),
        ),
        (
            "user".to_string(),
            format!("{COMPACT_PROMPT}\n\n=== CONVERSATION HISTORY ===\n{transcript}"),
        ),
    ];
    let (reply, _) = chat_completion(client, base_url, api_key, model, &messages).await?;
    Ok(reply)
}

// ---------------------------------------------------------------------------
// Parsing the model reply: text, file:/run blocks, ordinary code blocks
// ---------------------------------------------------------------------------

/// A block of the model reply.
#[derive(Debug)]
enum Block {
    /// Ordinary text outside code blocks.
    Text(String),
    /// An ordinary code block (```lang) — display only, does nothing.
    Code { lang: String, body: String },
    /// ```file:path — a proposal to fully overwrite a file.
    File { path: String, body: String },
    /// ```run — a proposal to run a command.
    Run { cmd: String },
}

/// Actions that require user confirmation.
#[derive(Debug)]
enum Action {
    WriteFile { path: String, body: String },
    RunCmd { cmd: String },
}

/// Flushes the text accumulated between blocks into the block list.
fn flush_plain(plain: &mut String, blocks: &mut Vec<Block>) {
    let taken = mem::take(plain);
    if !taken.trim().is_empty() {
        blocks.push(Block::Text(taken));
    }
}

/// Does the block header look like a file path: a single "token" containing
/// '/' or '.' and not being a known language name.
fn looks_like_path(info: &str) -> bool {
    let s = info.trim();
    if s.is_empty() || s.contains(char::is_whitespace) {
        return false;
    }
    let lower = s.to_lowercase();
    if KNOWN_LANGS.iter().any(|l| *l == lower) {
        return false;
    }
    s.contains('/') || (s.contains('.') && !lower.chars().all(|c| c.is_ascii_alphabetic() || c == '-'))
}

/// Classifies a block by its header (info — the text after ```).
fn classify_block(info: &str, body: &str) -> Block {
    let info = info.trim();
    if info.is_empty() {
        return Block::Code { lang: String::new(), body: body.to_string() };
    }
    let lower = info.to_lowercase();

    // Explicit protocol: ```file:path or ```file=path
    if lower.starts_with("file:") || lower.starts_with("file=") {
        let path = info[5..].trim().replace('\\', "/");
        return Block::File { path, body: body.to_string() };
    }
    // Explicit protocol: ```run (the command is in the block body)
    if lower == "run" {
        return Block::Run { cmd: body.trim().to_string() };
    }
    // Tolerance: ```run:command or ```run=command (command in the header)
    if lower.starts_with("run:") || lower.starts_with("run=") {
        let cmd = info[4..].trim();
        let cmd = if cmd.is_empty() { body.trim() } else { cmd };
        return Block::Run { cmd: cmd.to_string() };
    }
    // Many models simply write ```src/main.rs — treat it as a file.
    if looks_like_path(info) {
        return Block::File { path: info.replace('\\', "/"), body: body.to_string() };
    }
    // Everything else (rust, python, bash, …) — code for display.
    Block::Code {
        lang: info.split_whitespace().next().unwrap_or("").to_string(),
        body: body.to_string(),
    }
}

/// Parses the model reply into blocks. An unclosed block at the end is
/// treated as ordinary code (safe: it is never executed).
fn parse_blocks(reply: &str) -> Vec<Block> {
    let lines: Vec<&str> = reply.lines().collect();
    let mut blocks: Vec<Block> = Vec::new();
    let mut plain = String::new();
    let mut i = 0usize;

    while i < lines.len() {
        if let Some(info) = lines[i].trim().strip_prefix("```") {
            flush_plain(&mut plain, &mut blocks);
            let mut body = String::new();
            let mut closed = false;
            let mut j = i + 1;
            while j < lines.len() {
                if lines[j].trim() == "```" {
                    closed = true;
                    break;
                }
                body.push_str(lines[j]);
                body.push('\n');
                j += 1;
            }
            blocks.push(classify_block(info, &body));
            i = if closed { j + 1 } else { j };
            continue;
        }
        plain.push_str(lines[i]);
        plain.push('\n');
        i += 1;
    }
    flush_plain(&mut plain, &mut blocks);
    blocks
}

/// Validates a path proposed by the model: relative only, no escaping the
/// working directory (no .., no / and no absolute prefixes).
fn sanitize_rel_path(raw: &str) -> Option<PathBuf> {
    let p = raw.trim().trim_matches(|c| c == '"' || c == '\'');
    if p.is_empty() {
        return None;
    }
    let path = Path::new(p);
    if path.is_absolute() {
        return None;
    }
    for comp in path.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {} // ./x — allowed
            _ => return None,                              // '..', '/', prefixes — forbidden
        }
    }
    Some(PathBuf::from(p))
}

/// Displays the model reply: text as is, actions as compact notes.
/// Also collects the list of actions in order of appearance.
/// If the reply was already printed as a stream, text and ordinary blocks
/// are not printed again — otherwise they would appear twice on screen.
fn render_reply(blocks: &[Block], actions: &mut Vec<Action>, streamed: bool) {
    for block in blocks {
        match block {
            Block::Text(t) => {
                if !streamed {
                    print!("{t}");
                }
            }
            Block::Code { lang, body } => {
                if !streamed {
                    println!("{}```{}{}", col(ANSI_DIM), lang, col(ANSI_RESET));
                    print!("{body}");
                    println!("{}```{}", col(ANSI_DIM), col(ANSI_RESET));
                }
            }
            Block::File { path, body } => {
                println!(
                    "{}[✎ file change suggested: {} ({} lines)]{}",
                    col(ANSI_CYAN),
                    path,
                    body.lines().count(),
                    col(ANSI_RESET)
                );
                match sanitize_rel_path(path) {
                    Some(safe) => actions.push(Action::WriteFile {
                        path: safe.to_string_lossy().to_string(),
                        body: body.clone(),
                    }),
                    None => paintln(
                        ANSI_RED,
                        "  ⚠ unsafe path (absolute or contains ..) — the change will be skipped",
                    ),
                }
            }
            Block::Run { cmd } => {
                println!("{}[▸ command suggested: {cmd}]{}", col(ANSI_CYAN), col(ANSI_RESET));
                if !cmd.is_empty() {
                    actions.push(Action::RunCmd { cmd: cmd.clone() });
                }
            }
        }
    }
}

/// Applies a confirmed action: file write via std::fs or command execution
/// via sh -c (in the working directory).
fn apply_action(root: &Path, action: &Action) {
    match action {
        Action::WriteFile { path, body } => {
            let Some(rel) = sanitize_rel_path(path) else {
                paintln(ANSI_RED, "  ✗ unsafe path — skipped");
                return;
            };
            let full = root.join(&rel);
            if let Some(dir) = full.parent() {
                if let Err(e) = fs::create_dir_all(dir) {
                    paintln(ANSI_RED, &format!("  ✗ failed to create directory {}: {e}", dir.display()));
                    return;
                }
            }
            // Files conventionally end with a newline.
            let content = format!("{}\n", body.trim_end());
            match fs::write(&full, content.as_bytes()) {
                Ok(()) => paintln(
                    ANSI_GREEN,
                    &format!("  ✓ file {} written ({} bytes)", rel.display(), content.len()),
                ),
                Err(e) => paintln(ANSI_RED, &format!("  ✗ failed to write the file: {e}")),
            }
        }
        Action::RunCmd { cmd } => match Command::new("sh").arg("-c").arg(cmd).current_dir(root).output() {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                if !stdout.trim().is_empty() {
                    println!("{}", truncate_utf8(&stdout, 4000));
                }
                if !stderr.trim().is_empty() {
                    paintln(ANSI_YELLOW, &truncate_utf8(&stderr, 4000).to_string());
                }
                if let Some(code) = out.status.code() {
                    if code == 0 {
                        paintln(ANSI_GREEN, "  ✓ command finished (exit code 0)");
                    } else {
                        paintln(ANSI_RED, &format!("  ✗ command exited with code {code}"));
                    }
                } else {
                    paintln(ANSI_RED, "  ✗ process terminated by a signal");
                }
            }
            Err(e) => paintln(ANSI_RED, &format!("  ✗ failed to run 'sh': {e}")),
        },
    }
}

// ---------------------------------------------------------------------------
// Main application loop
// ---------------------------------------------------------------------------

fn print_help() {
    println!("{}Commands:{}", col(ANSI_BOLD), col(ANSI_RESET));
    println!("  /help    — this help");
    println!("  /models  — pick the active model from the added ones");
    println!("  /config  — editor for providers and models (add/edit/remove)");
    println!("  /auto    — confirmation-free autopilot: /auto on|off (no argument shows the status)");
    println!("  /compact — compress the conversation history into a brief summary");
    println!("  /clear   — clear the conversation history");
    println!("  /scan    — show what the agent will see in the project context");
    println!("  /exit    — quit (also /quit, /q, Ctrl+D; Ctrl+C exits immediately)");
    println!("File changes and commands are applied after confirmation (y/n), except in /auto mode.");
}

// ---------------------------------------------------------------------------
// Install and uninstall: make the command available from anywhere
// ---------------------------------------------------------------------------

/// Candidate directories for installation (by priority): the system-wide
/// /usr/local/bin — as root; the user's ~/.local/bin — without root.
fn install_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![PathBuf::from("/usr/local/bin")];
    if let Ok(home) = env::var("HOME") {
        dirs.push(PathBuf::from(&home).join(".local").join("bin"));
    }
    dirs
}

/// Installs the executable at the target path with 0755 permissions.
/// We copy to a temporary file next to the target (rename works only within
/// one filesystem) and atomically rename it over the target. This way
/// re-installation works even "over itself": a running binary cannot be
/// overwritten ("Text file busy"), but its directory entry can be replaced;
/// the running process keeps executing the old inode. Explicit 0755 is a
/// must: a FAT32 flash drive does not store the executable bit — a file
/// copied from it without chmod gives "Permission denied".
fn copy_executable(src: &Path, dest: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    // Unique name with the pid — so parallel installs do not interfere.
    let tmp = dest.with_file_name(format!(".{}.tmp.{}", APP_NAME, std::process::id()));
    if let Err(e) = fs::copy(src, &tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(format!("{}: {e}", tmp.display()));
    }
    let mut perm = match fs::metadata(&tmp) {
        Ok(m) => m.permissions(),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            return Err(format!("{}: {e}", tmp.display()));
        }
    };
    perm.set_mode(0o755);
    if let Err(e) = fs::set_permissions(&tmp, perm) {
        let _ = fs::remove_file(&tmp);
        return Err(format!("{}: {e}", tmp.display()));
    }
    if let Err(e) = fs::rename(&tmp, dest) {
        let _ = fs::remove_file(&tmp);
        return Err(format!("{}: {e}", dest.display()));
    }
    Ok(())
}

/// `--install`: copies the running binary to /usr/local/bin (as root) or,
/// if it is not writable, to ~/.local/bin. Afterwards the command can be
/// typed from anywhere. Installation requires no confirmations and touches
/// no files in the current directory: it only copies the executable itself.
fn cmd_install() -> i32 {
    let exe = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to resolve the running binary path: {e}");
            return 1;
        }
    };
    let mut last_err = String::new();
    for dir in install_dirs() {
        let dest = dir.join(APP_NAME);
        let existed = dest.exists();
        // ~/.local/bin may not exist yet — create it; /usr/local/bin usually
        // exists and create_dir_all there is simply a no-op.
        if fs::create_dir_all(&dir).is_err() {
            continue;
        }
        match copy_executable(&exe, &dest) {
            Ok(()) => {
                let verb = if existed { "updated" } else { "installed" };
                println!("{APP_NAME} {verb}: {}", dest.display());
                let path_var = env::var("PATH").unwrap_or_default();
                let dir_str = dir.to_string_lossy();
                if path_var.split(':').any(|p| Path::new(p) == dir) {
                    println!("The \"{APP_NAME}\" command is now available from anywhere — run it in any project directory.");
                } else {
                    println!("Add the directory to PATH to make the command available from anywhere:");
                    println!("  export PATH=\"{dir_str}:$PATH\"   # add to ~/.bashrc or /etc/profile");
                }
                return 0;
            }
            Err(e) => last_err = e,
        }
    }
    eprintln!("Installation failed: {last_err}");
    eprintln!("Run the installation as root: su -c '/path/to/{APP_NAME} --install'");
    1
}

/// `--uninstall`: removes the previously installed binary from all
/// install locations.
fn cmd_uninstall() -> i32 {
    let mut removed = false;
    for dir in install_dirs() {
        let dest = dir.join(APP_NAME);
        if dest.exists() {
            match fs::remove_file(&dest) {
                Ok(()) => {
                    println!("Removed: {}", dest.display());
                    removed = true;
                }
                Err(e) => eprintln!("Failed to remove {}: {e}", dest.display()),
            }
        }
    }
    if removed {
        0
    } else {
        println!("{APP_NAME}: not found in /usr/local/bin or ~/.local/bin — nothing to remove.");
        1
    }
}

/// Command-line help.
fn print_usage() {
    println!("{APP_NAME} v{VERSION} — a minimal AI coding agent for the terminal (TTY)");
    println!("Usage: {APP_NAME} [--yolo] [--install|--uninstall]");
    println!("  --yolo        — autopilot mode: files and commands without confirmations (same as /auto on)");
    println!("  --install     — install: copy itself to /usr/local/bin (as root) or to ~/.local/bin");
    println!("                    so that the \"{APP_NAME}\" command works from anywhere");
    println!("  --uninstall   — remove the installed command");
    println!("In-session commands: /help /models /config /auto /compact /clear /scan /exit");
}

fn banner(root: &Path, auto: bool) {
    println!(
        "{}{APP_NAME}{} v{} — a minimal AI coding agent for the terminal (TTY)",
        col(ANSI_BOLD),
        col(ANSI_RESET),
        VERSION
    );
    println!("Working directory: {}", root.display());
    if auto {
        paintln(
            ANSI_RED,
            "AUTOPILOT MODE (--yolo): file changes and commands are applied without confirmation.",
        );
    }
}

/// Returns the process exit code: 0 — normal, 1 — refusal/error.
async fn run() -> i32 {
    // --- command-line arguments ---
    let mut auto = false;
    for a in env::args().skip(1) {
        match a.as_str() {
            "--yolo" | "--auto" => auto = true,
            "--install" => return cmd_install(),
            "--uninstall" => return cmd_uninstall(),
            "--help" | "-h" => {
                print_usage();
                return 0;
            }
            other => {
                println!("Unknown argument \"{other}\". Available: --yolo, --install, --uninstall, --help");
                return 1;
            }
        }
    }

    let root = match env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to determine the current directory: {e}");
            return 1;
        }
    };

    banner(&root, auto);

    // --- Step 1: explicit permission to access files in the current directory ---
    paintln(ANSI_YELLOW, "Warning: this application requests permission to read and modify files");
    paintln(ANSI_YELLOW, &format!("in the current directory [{}].", root.display()));
    if !confirm("Allow?") {
        println!("Permission denied — exiting.");
        return 1;
    }

    // --- Step 2: configuration (several providers and models) ---
    let mut cfg = load_config();

    // --- HTTP client: created once for the whole session ---
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
        .connect_timeout(Duration::from_secs(HTTP_CONNECT_TIMEOUT_SECS))
        .user_agent(format!("{APP_NAME}/{VERSION}"))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            paintln(ANSI_RED, &format!("Failed to create the HTTP client: {e}"));
            return 1;
        }
    };

    // Conversation history (user/assistant) — the system prompt is rebuilt
    // before every request, so it is not stored in the history.
    let mut history: Vec<(String, String)> = Vec::new();

    println!();
    print_help();
    println!();

    // --- Step 3: the main loop ---
    loop {
        print!("{}{}>{} ", col(ANSI_BOLD), col(ANSI_CYAN), col(ANSI_RESET));
        let _ = io::stdout().flush();
        let Some(raw_input) = read_line() else {
            println!();
            break; // EOF (Ctrl+D) — a graceful exit
        };
        let input = raw_input.trim().to_string();
        if input.is_empty() {
            continue;
        }

        // --- built-in commands ---
        match input.as_str() {
            "/exit" | "/quit" | "/q" => break,
            "/help" => {
                print_help();
                continue;
            }
            "/clear" => {
                history.clear();
                paintln(ANSI_GREEN, "Conversation history cleared.");
                continue;
            }
            "/models" => {
                models_menu(&mut cfg);
                continue;
            }
            "/config" => {
                config_menu(&mut cfg);
                continue;
            }
            "/scan" => {
                let scan = scan_project(&root);
                println!(
                    "Files in project: {}. Included in context: {} ({} KB, truncated: {}).",
                    scan.tree.len(),
                    scan.scanned,
                    scan.contents.len() / 1024,
                    scan.truncated
                );
                continue;
            }
            "/compact" => {
                if history.is_empty() {
                    paintln(ANSI_YELLOW, "History is empty — nothing to compact.");
                    continue;
                }
                let (p, model) = cfg.active();
                let before = history_bytes(&history);
                paintln(ANSI_CYAN, "Compacting the conversation history…");
                match compact_history(&client, &p.base_url, &p.api_key, model, &history).await {
                    Ok(summary) => {
                        history =
                            vec![("user".to_string(), format!("[Compressed conversation history]\n{summary}"))];
                        paintln(
                            ANSI_GREEN,
                            &format!(
                                "History compacted: {} KB → {} KB.",
                                before / 1024,
                                history_bytes(&history) / 1024
                            ),
                        );
                    }
                    Err(e) => paintln(ANSI_RED, &format!("Failed to compact the history: {e}")),
                }
                continue;
            }
            s if s.starts_with("/auto") => {
                let arg = s[5..].trim().to_lowercase();
                match arg.as_str() {
                    "on" => auto = true,
                    "off" => auto = false,
                    "" => {}
                    _ => {
                        paintln(ANSI_YELLOW, "Usage: /auto on|off (no argument shows the status).");
                        continue;
                    }
                }
                if auto {
                    paintln(
                        ANSI_RED,
                        "AUTOPILOT is ON: files and commands are applied without confirmation (dangerous commands are blocked).",
                    );
                } else {
                    paintln(
                        ANSI_GREEN,
                        "AUTOPILOT is OFF: every action requires confirmation (y/n).",
                    );
                }
                continue;
            }
            _ if input.starts_with('/') => {
                paintln(
                    ANSI_YELLOW,
                    "Unknown command. Available: /help /models /config /auto /compact /clear /scan /exit",
                );
                continue;
            }
            _ => {}
        }

        // --- Step 4: auto-compaction if the history overflowed its budget ---
        if history_bytes(&history) > AUTO_COMPACT_BYTES {
            paintln(
                ANSI_YELLOW,
                &format!(
                    "The conversation history grew large ({} KB) — compacting…",
                    history_bytes(&history) / 1024
                ),
            );
            let (p, model) = cfg.active();
            match compact_history(&client, &p.base_url, &p.api_key, model, &history).await {
                Ok(summary) => {
                    history =
                        vec![("user".to_string(), format!("[Compressed conversation history]\n{summary}"))];
                    paintln(ANSI_GREEN, "History compacted.");
                }
                Err(e) => paintln(ANSI_YELLOW, &format!("Compaction failed ({e}) — continuing as is.")),
            }
        }

        // --- Step 5: fresh project scan → system prompt ---
        // The context is rebuilt before EVERY request so the model sees the
        // current state of the files (including agent-made changes).
        let scan = scan_project(&root);
        let mut messages: Vec<(String, String)> = Vec::with_capacity(history.len() + 2);
        messages.push(("system".to_string(), build_system_prompt(&root, &scan)));
        messages.extend(history.iter().cloned());
        messages.push(("user".to_string(), input.clone()));

        // --- Step 6: request to the active provider/model ---
        let (provider, model) = cfg.active();
        let (reply, streamed) = match chat_completion(
            &client,
            &provider.base_url,
            &provider.api_key,
            model,
            &messages,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                paintln(ANSI_RED, &format!("Request error: {e}"));
                continue; // no crash: let the user retry the request
            }
        };

        // --- Step 7: parse the reply and run the proposed actions ---
        let blocks = parse_blocks(&reply);
        let mut actions: Vec<Action> = Vec::new();
        render_reply(&blocks, &mut actions, streamed);

        for action in &actions {
            match action {
                Action::WriteFile { path, .. } => {
                    let apply = if auto {
                        true // autopilot: no questions
                    } else {
                        confirm(&format!(
                            "The AI wants to modify the file [{path}]. Apply the changes?"
                        ))
                    };
                    if apply {
                        if auto {
                            paintln(ANSI_CYAN, &format!("[auto] writing file {path}"));
                        }
                        apply_action(&root, action);
                    } else {
                        paintln(ANSI_YELLOW, "  — change rejected.");
                    }
                }
                Action::RunCmd { cmd } => {
                    if auto {
                        // In autopilot, dangerous commands are not asked about
                        // but blocked: the user may be away from the computer.
                        if is_dangerous(cmd) {
                            paintln(
                                ANSI_RED,
                                &format!("[auto] ✗ command blocked as dangerous: {cmd}"),
                            );
                        } else {
                            paintln(ANSI_CYAN, &format!("[auto] running: {cmd}"));
                            apply_action(&root, action);
                        }
                    } else if confirm(&format!(
                        "The AI wants to run the command [{cmd}]. Run it?"
                    )) {
                        apply_action(&root, action);
                    } else {
                        paintln(ANSI_YELLOW, "  — command rejected.");
                    }
                }
            }
        }

        // --- Step 8: remember the dialog (long replies are truncated) ---
        history.push(("user".to_string(), input));
        history.push((
            "assistant".to_string(),
            truncate_utf8(&reply, MAX_HISTORY_ITEM_BYTES).to_string(),
        ));
        while history.len() > MAX_HISTORY_MESSAGES {
            history.remove(0);
        }
        println!();
    }

    println!("Goodbye!");
    0
}

/// Entry point. Single-threaded runtime (current_thread): all of run() and
/// the network requests execute on the only thread — minimal CPU and RAM.
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let code = run().await;
    std::process::exit(code);
}

// ---------------------------------------------------------------------------
// Unit tests of the key functions (cargo test)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_url() {
        assert_eq!(
            endpoint_url("http://localhost:11434/v1/"),
            "http://localhost:11434/v1/chat/completions"
        );
        // A host without a path — /v1 is added automatically
        assert_eq!(
            endpoint_url("https://api.openai.com"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            endpoint_url(" http://127.0.0.1:8080/v1 "),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
    }

    #[test]
    fn test_error_chain_joins_sources() {
        #[derive(Debug)]
        struct Root;
        impl std::fmt::Display for Root {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "name resolution failed")
            }
        }
        impl std::error::Error for Root {}
        #[derive(Debug)]
        struct Mid;
        impl std::fmt::Display for Mid {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "dns error")
            }
        }
        impl std::error::Error for Mid {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&Root)
            }
        }
        assert_eq!(error_chain(&Mid), "name resolution failed");
        // No source chain — fall back to the Display text.
        assert_eq!(error_chain(&Root), "name resolution failed");
    }

    #[test]
    fn test_parse_blocks() {
        let reply = "Text before\n\
                     ```file:src/a.rs\nfn main() {}\n```\n\
                     Middle\n\
                     ```run\ncargo build\n```\n\
                     ```rust\nlet x = 1;\n```\n\
                     Text after";
        let blocks = parse_blocks(reply);
        assert_eq!(blocks.len(), 6);
        assert!(matches!(&blocks[0], Block::Text(_)));
        if let Block::File { path, body } = &blocks[1] {
            assert_eq!(path, "src/a.rs");
            assert_eq!(body, "fn main() {}\n");
        } else {
            panic!("block 1 must be File");
        }
        if let Block::Run { cmd } = &blocks[3] {
            assert_eq!(cmd, "cargo build");
        } else {
            panic!("block 3 must be Run");
        }
        if let Block::Code { lang, .. } = &blocks[4] {
            assert_eq!(lang, "rust");
        } else {
            panic!("block 4 must be Code");
        }
        assert!(matches!(&blocks[5], Block::Text(_)));
    }

    #[test]
    fn test_parse_file_with_space_and_unclosed() {
        // a space after "file:" and an unclosed block at the end
        let blocks = parse_blocks("```file: x.txt\nhello\n");
        assert_eq!(blocks.len(), 1);
        if let Block::File { path, body } = &blocks[0] {
            assert_eq!(path, "x.txt");
            assert_eq!(body, "hello\n");
        } else {
            panic!("File expected");
        }
    }

    #[test]
    fn test_sanitize_rel_path() {
        assert!(sanitize_rel_path("src/main.rs").is_some());
        assert!(sanitize_rel_path("./a/b.txt").is_some());
        assert!(sanitize_rel_path(" README.md ").is_some());
        assert!(sanitize_rel_path("../evil").is_none());
        assert!(sanitize_rel_path("a/../../b").is_none());
        assert!(sanitize_rel_path("/etc/passwd").is_none());
        assert!(sanitize_rel_path("").is_none());
    }

    #[test]
    fn test_looks_like_path() {
        assert!(looks_like_path("src/main.rs"));
        assert!(looks_like_path("README.md"));
        assert!(!looks_like_path("rust"));
        assert!(!looks_like_path("mermaid"));
        assert!(!looks_like_path("some text here"));
    }

    #[test]
    fn test_truncate_utf8() {
        let s = "привет"; // Cyrillic: 2 bytes per character — multibyte check
        assert_eq!(truncate_utf8(s, 5), "пр");
        assert_eq!(truncate_utf8(s, 100), s);
        assert_eq!(truncate_utf8("", 10), "");
    }

    #[test]
    fn test_decode_input() {
        // valid UTF-8 passes through unchanged
        assert_eq!(decode_input("привет мир".as_bytes()), "привет мир");
        // "привет" in CP1251 (what old Russian terminals send)
        assert_eq!(decode_input(&[0xEF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2]), "привет");
        // Ё and ё in CP1251
        assert_eq!(decode_input(&[0xA8, 0xB8]), "Ёё");
        // undefined byte 0x98 → U+FFFD, no panic
        assert_eq!(decode_input(&[0x98]), "\u{FFFD}");
        // mixed input: ASCII + CP1251
        assert_eq!(decode_input(b"file: " as &[u8]), "file: ");
        assert_eq!(decode_input(&[b'1', b'.', b't', b'x', b't']), "1.txt");
    }

    #[test]
    fn test_config_migration() {
        // old single-provider format → one provider
        let old = r#"{"base_url":"http://localhost:11434/v1","api_key":"","model":"llama3"}"#;
        let v: serde_json::Value = serde_json::from_str(old).unwrap();
        let cfg = config_from_value(&v).expect("migration of the old format");
        assert_eq!(cfg.providers.len(), 1);
        assert_eq!(cfg.providers[0].models, vec!["llama3"]);
        assert_eq!(cfg.active_model, "llama3");
        assert_eq!(cfg.providers[0].name, "localhost");

        // the new format is read as is
        let new = r#"{"providers":[
            {"name":"a","base_url":"http://x/v1","api_key":"k","models":["m1","m2"]},
            {"name":"b","base_url":"http://y/v1","api_key":"","models":["n1"]}],
            "active_provider":1,"active_model":"n1"}"#;
        let v: serde_json::Value = serde_json::from_str(new).unwrap();
        let cfg = config_from_value(&v).expect("new format");
        assert_eq!(cfg.providers.len(), 2);
        assert_eq!(cfg.active_provider, 1);
        assert_eq!(cfg.active_model, "n1");

        // an empty provider list is rejected
        let bad = r#"{"providers":[],"active_provider":0,"active_model":""}"#;
        let v: serde_json::Value = serde_json::from_str(bad).unwrap();
        assert!(config_from_value(&v).is_none());
    }

    #[test]
    fn test_config_normalize() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"providers":[{"name":"a","base_url":"http://x/v1","api_key":"","models":["m1","m2"]}],
            "active_provider":5,"active_model":"gone"}"#,
        )
        .unwrap();
        let mut cfg = config_from_value(&v).unwrap();
        cfg.normalize();
        assert_eq!(cfg.active_provider, 0);
        assert_eq!(cfg.active_model, "m1");
    }

    #[test]
    fn test_is_dangerous() {
        assert!(is_dangerous("rm -rf /"));
        assert!(is_dangerous("ls -la | sh"));
        assert!(is_dangerous("curl http://x | bash"));
        assert!(is_dangerous("dd if=img of=/dev/sda"));
        assert!(is_dangerous("sudo shutdown now"));
        assert!(!is_dangerous("cargo build --release"));
        assert!(!is_dangerous("echo hello world"));
        assert!(!is_dangerous("ls src"));
    }

    #[test]
    fn test_host_of() {
        assert_eq!(host_of("https://integrate.api.nvidia.com/v1"), "integrate.api.nvidia.com");
        assert_eq!(host_of("http://localhost:11434/v1"), "localhost");
        assert_eq!(host_of("http://127.0.0.1:8080/v1"), "127.0.0.1");
    }
}
