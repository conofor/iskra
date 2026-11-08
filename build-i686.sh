#!/bin/sh
# ============================================================================
# Cross-compilation of a static 32-bit iskra binary (i686 architecture):
# for old/weak hardware and any 32-bit Linux with no glibc dependency.
#
# One-time prerequisites:
#   1. rustup (https://rustup.rs) with any recent Rust;
#   2. the musl.cc toolchain: download https://musl.cc/i686-linux-musl-cross.tgz
#      (~105 MB) and unpack it into ~/musl-cross/ — you get the folder
#      ~/musl-cross/i686-linux-musl-cross with i686-linux-musl-gcc inside.
#
# Result: target/i686-unknown-linux-musl/release/iskra — a fully static
# ELF32 (~1.9 MB) that runs on any distribution without glibc.
# If the target machine has no SSE2 (check: grep -cw sse2 /proc/cpuinfo),
# switch the toolchain to i586-linux-musl-cross and the target to
# i586-unknown-linux-musl.
# ============================================================================
set -e

TC="${MUSL_CROSS:-$HOME/musl-cross/i686-linux-musl-cross}"
if [ ! -x "$TC/bin/i686-linux-musl-gcc" ]; then
    echo "Toolchain not found in $TC"
    echo "Download https://musl.cc/i686-linux-musl-cross.tgz and run:"
    echo "  mkdir -p ~/musl-cross && tar xzf i686-linux-musl-cross.tgz -C ~/musl-cross"
    exit 1
fi

export PATH="$HOME/.cargo/bin:$PATH"
rustup target add i686-unknown-linux-musl

export CC_i686_unknown_linux_musl="$TC/bin/i686-linux-musl-gcc"
export AR_i686_unknown_linux_musl="$TC/bin/i686-linux-musl-ar"
export CARGO_TARGET_I686_UNKNOWN_LINUX_MUSL_LINKER="$TC/bin/i686-linux-musl-gcc"

cargo build --release --target i686-unknown-linux-musl

echo
echo "Done: target/i686-unknown-linux-musl/release/iskra"
echo "Copy it to a flash drive (together with install.sh), then on the target machine:"
echo "  sh /mnt/install.sh     # or: /mnt/iskra --install"
