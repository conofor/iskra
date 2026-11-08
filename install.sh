#!/bin/sh
# ============================================================================
# iskra installer: copies the binary from the folder with this script to
# /usr/local/bin (as root) or to ~/.local/bin, after which the command can be
# typed from anywhere. Intended for the target machine:
#
#   dmesg | tail            # find the flash drive name, e.g. /dev/sdb1
#   mount /dev/sdb1 /mnt
#   sh /mnt/install.sh      # or, instead of this: /mnt/iskra --install
#   umount /mnt
#
# The equivalent is the built-in installation by the binary itself:
#   ./iskra --install
# Uninstall: ./iskra --uninstall  or  rm /usr/local/bin/iskra
# ============================================================================
set -e

# The command name must match the binary name (name in Cargo.toml).
APP="iskra"

DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/$APP"

if [ ! -f "$SRC" ]; then
    echo "Error: the binary is not next to the script: $SRC"
    echo "Put install.sh and the $APP binary into one folder (on the flash drive)."
    exit 1
fi

# Try the system directory (needs root) first, then the user's one.
# FAT32 does not store permissions — that is why chmod +x is mandatory.
for DEST in /usr/local/bin "$HOME/.local/bin"; do
    if mkdir -p "$DEST" 2>/dev/null \
       && cp "$SRC" "$DEST/" 2>/dev/null \
       && chmod 755 "$DEST/$APP" 2>/dev/null; then
        echo "$APP installed: $DEST/$APP"
        case ":$PATH:" in
            *":$DEST:"*)
                echo "The \"$APP\" command is now available from anywhere — run it in any project directory."
                ;;
            *)
                echo "Add the directory to PATH to make the command available from anywhere:"
                echo "  export PATH=\"$DEST:\$PATH\"   # add to ~/.bashrc or /etc/profile"
                ;;
        esac
        exit 0
    fi
done

echo "Installation failed: no permission for /usr/local/bin or ~/.local/bin."
echo "Run as root:  su -c 'sh /path/to/install.sh'"
exit 1
