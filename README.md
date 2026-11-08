<div align="center">

# iskra

**A lightweight AI coding agent for the terminal**

One static binary ≈2 MB · no GUI · minimal dependencies

[![Rust](https://img.shields.io/badge/rust-1.70%2B-dea584?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86__64%20%7C%20i686-blue)](#build-and-cross-compilation)
[![Binary](https://img.shields.io/badge/binary-%E2%89%882%20MB-success)](#why-yet-another-agent)

</div>

---

**iskra** is a minimal OpenCode / Claude Code alternative written in pure Rust.
It connects to any OpenAI-compatible API, reads the project context, edits
files and runs terminal commands — after a `y/n` confirmation, or fully
automatically in autopilot mode.

## Why yet another agent

Most AI coding agents want a modern multi-core CPU, gigabytes of RAM and a
graphical environment. iskra takes the opposite approach: a tool that needs
almost nothing.

| | iskra | a typical agent |
|---|---|---|
| Binary | ≈2 MB, static | hundreds of MB + runtime (Node.js/Python…) |
| RAM | a few MB | hundreds of MB |
| Interface | plain TTY | usually a GUI |
| Runtime dependencies | none — not even glibc | runtime + packages |

That is why iskra runs:

- on weak and old hardware — down to **i686** (32-bit x86) processors;
- in a plain text console (TTY), including virtual consoles
  `Ctrl+Alt+F2…F6` and SSH without a graphical shell;
- on any Linux distribution — the static binary does not depend on glibc
  or the OS version.

## How the lightness is achieved

- a single-threaded async runtime tokio (`flavor = "current_thread"`) — no
  worker pool, no extra threads;
- just three dependencies (tokio, reqwest, serde/serde_json); interactive
  input is plain `std::io`, no TUI libraries;
- pure-Rust TLS (rustls + webpki-roots): root certificates are embedded in
  the binary — no system OpenSSL and no `/etc/ssl/certs` needed;
- release build `opt-level="s"` + LTO + `strip` — a compact binary
  (see `[profile.release]` in `Cargo.toml`);
- project scanning in a single pass with hard limits — the context never
  exceeds ~32 KB, no matter the repository size;
- input decoding from CP1251: works even in old consoles that send
  non-UTF-8 bytes.

## Features

- connection to **any number of OpenAI-compatible APIs** (Ollama, llama.cpp,
  NVIDIA NIM, OpenAI, etc.); each provider has its own list of models,
  switchable on the fly;
- the agent scans the current folder and sees the tree and contents of text
  files (`.git`, `node_modules`, `target`, binaries and symlinks are
  skipped);
- file changes and command execution — after a `y/n` confirmation or
  without confirmations in autopilot;
- streaming output as the reply is generated (SSE) with a
  "model is thinking… Ns" indicator;
- conversation history compaction (`/compact`) plus automatic compaction
  on overflow;
- one-command installation (`--install`) — the command becomes available
  from anywhere; removal (`--uninstall`).

## Quick start

### Prebuilt binary

Download the static binary for your architecture (`iskra-linux-x86_64` or
`iskra-linux-i686`) from the **Releases** section of the repository — builds
are created automatically when a `v*` tag is pushed
(see `.github/workflows/release.yml`).

### Building from source

```sh
cargo build --release
./target/release/iskra --help
```

## Installing into the system

```sh
iskra --install      # /usr/local/bin (as root) or ~/.local/bin (without root)
iskra --uninstall    # remove
```

The installer copies the binary, sets 0755 permissions and, if the directory
is not in `PATH`, prints a ready-to-use `export PATH=…` line for `~/.bashrc`.
Running `--install` over an installed copy works (an update). There is also
`install.sh` in the repo — the same thing as a shell script, handy for
flash-drive installs.

## Configuration

Config: `~/.config/iskra/config.json` (or `$XDG_CONFIG_HOME/iskra/`).
Created by the interactive wizard on first run; you can also edit it by hand
or via `/config`. There can be several providers, each with its own model
list:

```json
{
  "providers": [
    {
      "name": "local-ollama",
      "base_url": "http://localhost:11434/v1",
      "api_key": "",
      "models": ["qwen2.5:1.5b", "llama3"]
    },
    {
      "name": "cloud",
      "base_url": "https://api.example.com/v1",
      "api_key": "sk-...",
      "models": ["model-a", "model-b"]
    }
  ],
  "active_provider": 0,
  "active_model": "qwen2.5:1.5b"
}
```

- The Base URL must include the API version prefix (usually `/v1`); if only
  a host is given, `/v1` is added automatically.
- For local models (Ollama, llama.cpp) leave `api_key` empty — the request
  goes out without an `Authorization` header.

## Usage

```sh
cd ~/projects/myapp
iskra
```

1. The app asks for permission to read and modify files in the current
   directory (`y/n`). A refusal exits immediately.
2. Type your request as plain text. The reply is printed as a stream;
   `file:` and `run` blocks are applied after a `y/n` confirmation.

### In-session commands

| Command | Action |
|---|---|
| `/help` | help |
| `/models` | pick the active model from the added ones |
| `/config` | editor for providers and models |
| `/auto on` / `off` | autopilot for the current session |
| `/compact` | compress the conversation history into a summary |
| `/clear` | clear the conversation history |
| `/scan` | show what the agent will see in the project context |
| `/exit` | quit (`Ctrl+D` does the same; `Ctrl+C` exits immediately) |

### Command-line arguments

| Argument | Action |
|---|---|
| `--yolo` | autopilot right from startup (same as `/auto on`) |
| `--install` | install the command into the system |
| `--uninstall` | remove the installed command |

### Autopilot

In autopilot, files are written and commands run without confirmations —
you can give a task and walk away. Safety measures stay in place:

- file paths are still validated (no `..`, no absolute paths, no symlinks);
- commands that look dangerous (`rm -rf`, `mkfs`, `dd of=/dev/…`,
  `shutdown`, `| sh`, etc. — the `DANGEROUS_CMD` list in `src/main.rs`) are
  **blocked** in autopilot, not executed;
- when the history exceeds ~24 KB it is automatically compressed by the
  model into a summary.

## Agent protocol

The rules are baked into the system prompt; the model replies with ordinary
text plus special blocks:

- ```` ```file:path/to/file ```` — full new file content; after a `y`
  confirmation the file is fully overwritten via `std::fs`;
- ```` ```run ```` — a terminal command; after a `y` confirmation it runs
  via `sh -c` in the working directory, output is shown to the user;
- ordinary code blocks (```` ```rust ```` etc.) are display-only and never
  executed.

## Context limits

On weak hardware and for small models the context is bounded by constants at
the top of `src/main.rs`:

- `.git`, `node_modules`, `target`, `vendor`, etc. are skipped;
- files over 64 KB are not included; 6 KB per file are truncated;
- ~32 KB of total content and 300 files in the tree at most;
- conversation history — 12 last messages (long replies cut to 3 KB).

If your model has a large context window, the limits can be raised.

## Build and cross-compilation

All builds are static (musl): the binary runs on any distribution,
regardless of glibc.

| Target | Architecture | Toolchain |
|---|---|---|
| `x86_64-unknown-linux-musl` | 64-bit x86 | `musl-tools` package |
| `i686-unknown-linux-musl` | 32-bit x86 (i686 and newer) | musl.cc + `./build-i686.sh` |
| `i586-unknown-linux-musl` | very old 32-bit x86 without SSE2 | musl.cc `i586-linux-musl-cross` |

```sh
# 64-bit:
cargo build --release --target x86_64-unknown-linux-musl

# 32-bit: the script checks the toolchain and sets the environment itself.
# Toolchain (once): https://musl.cc/i686-linux-musl-cross.tgz (~105 MB),
# unpack into ~/musl-cross/:
./build-i686.sh
# → target/i686-unknown-linux-musl/release/iskra (~1.9 MB)
```

If the target machine has no SSE2 (check: `grep -cw sse2 /proc/cpuinfo`) —
use the `i586-unknown-linux-musl` target with the `i586-linux-musl-cross`
toolchain.

## Moving to a machine without internet (flash drive)

```sh
# on the build machine:
cp target/i686-unknown-linux-musl/release/iskra install.sh /media/user/FLASH/
sync   # wait for the write to finish before unplugging

# on the target machine (as root: su -):
mount /dev/sdb1 /mnt          # drive name: dmesg | tail or fdisk -l
/mnt/iskra --install          # or: sh /mnt/install.sh
umount /mnt
cd ~/project && iskra         # the command works from anywhere
```

A FAT32 flash drive does not store permissions — the installer sets the
executable bit itself, no manual `chmod +x` needed.

## License

[MIT](LICENSE).
