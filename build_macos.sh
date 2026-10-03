#!/usr/bin/env bash
#
# Build a native Apple Silicon (arm64) "MSR Viewer.app" and pack it into a .dmg.
#
# Usage (on the Mac, in Terminal):
#   ./build_macos.sh                   # version from __version__ in msr_viewer.py
#   ./build_macos.sh 0.4.0             # override the version label of the .dmg
#   ./build_macos.sh --skip-install    # reuse .venv-build-macos as it is
#   PYTHON=/opt/homebrew/bin/python3 ./build_macos.sh   # pick the Python
#
# Requirements:
#   - A native arm64 Python 3.10+ (NOT x86_64 under Rosetta).
#   - Xcode command line tools (iconutil, hdiutil, codesign): xcode-select --install
#   - Optional: `brew install create-dmg` for a drag-to-Applications window layout.
#
# The app is signed ad hoc, not with an Apple Developer ID: on first start macOS
# asks to confirm it (System Settings > Privacy & Security > Open Anyway).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SKIP_INSTALL=0
VERSION=""
for arg in "$@"; do
    case "$arg" in
        --skip-install) SKIP_INSTALL=1 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) VERSION="$arg" ;;
    esac
done
if [[ -z "$VERSION" ]]; then
    VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' msr_viewer.py | head -1)"
    VERSION="${VERSION:-0.0.0}"
fi
PYTHON="${PYTHON:-python3}"
echo ">> Building MSR Viewer v${VERSION} for Apple Silicon"

# --- 0) Refuse to build under Rosetta / x86_64 ----------------------------------
"$PYTHON" - <<'PY'
import platform, sys
if platform.machine() != "arm64":
    sys.exit(
        "ERROR: this Python is '%s', not arm64.\n"
        "Use a native Apple Silicon Python (arm64 Homebrew python3, the python.org\n"
        "installer or an arm64 conda env); under Rosetta the app would be x86_64."
        % platform.machine()
    )
if sys.version_info < (3, 10):
    sys.exit("ERROR: Python 3.10 or newer is needed, this is " + platform.python_version())
print(">> OK: native arm64 Python", platform.python_version())
PY

# --- 1) Build environment with only the viewer's packages -----------------------
VENV="$ROOT/.venv-build-macos"
if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
if [[ "$SKIP_INSTALL" == 0 ]]; then
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt pyinstaller pillow
fi

# --- 2) App icon (.icns) ----------------------------------------------------------
# From msr_viewer.png if present (the logo at 1024 px or more, sharpest on Retina
# screens), else from the 256 px image in msr_viewer.ico.  The logo is placed on
# the macOS icon grid (artwork ~824 of 1024 px) so it matches the other Dock icons.
python - <<'PY' || echo ">> (icon generation skipped; the app gets the default icon)"
import subprocess, tempfile
from pathlib import Path
from PIL import Image

src = next((p for p in (Path("msr_viewer.png"), Path("msr_viewer.ico")) if p.exists()), None)
if src is None:
    raise SystemExit(1)
img = Image.open(src).convert("RGBA")   # an .ico opens at its largest size
if img.width != img.height:
    side = max(img.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2), img)
    img = square
inner = round(824 * 1.04)               # the logo carries a 2 % margin around its rounded square
art = img.convert("RGBa").resize((inner, inner), Image.LANCZOS).convert("RGBA")
base = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
base.alpha_composite(art, ((1024 - inner) // 2, (1024 - inner) // 2))
with tempfile.TemporaryDirectory() as d:
    iconset = Path(d) / "msr_viewer.iconset"
    iconset.mkdir()
    for s in (16, 32, 128, 256, 512):
        for scale, suffix in ((1, ""), (2, "@2x")):
            px = s * scale
            frame = base.convert("RGBa").resize((px, px), Image.LANCZOS).convert("RGBA")
            frame.save(iconset / f"icon_{s}x{s}{suffix}.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", "msr_viewer.icns"], check=True)
print(">> Icon msr_viewer.icns from", src, "" if src.suffix == ".png" else
      "(256 px; put the logo as msr_viewer.png next to this script for sharper large icons)")
PY

# --- 3) Build the .app ---------------------------------------------------------------
python -m PyInstaller --noconfirm --clean msr_viewer_macos.spec

APP="$ROOT/dist/MSR Viewer.app"
if [[ ! -d "$APP" ]]; then
    echo "ERROR: expected $APP was not produced." >&2
    exit 1
fi
if [[ -z "$(find "$APP/Contents" -name 'ffmpeg-*' -perm -u+x -print -quit)" ]]; then
    echo ">> WARNING: no executable ffmpeg in the app; video export will need ffmpeg on PATH." >&2
fi

# --- 4) Pack into a .dmg -------------------------------------------------------------
DMG="$ROOT/dist/MSR_Viewer_AppleSilicon_v${VERSION}.dmg"
rm -f "$DMG"
if command -v create-dmg >/dev/null 2>&1; then
    create-dmg \
        --volname "MSR Viewer ${VERSION}" \
        --window-size 640 320 \
        --icon "MSR Viewer.app" 150 160 \
        --app-drop-link 450 160 \
        "$DMG" "$APP"
else
    echo ">> create-dmg not found (brew install create-dmg for a nicer window); using hdiutil."
    STAGE="$(mktemp -d)"
    ditto "$APP" "$STAGE/MSR Viewer.app"
    ln -s /Applications "$STAGE/Applications"
    hdiutil create -volname "MSR Viewer ${VERSION}" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
    rm -rf "$STAGE"
fi

echo ""
echo ">> Done: $DMG ($(du -h "$DMG" | cut -f1))"
echo ">> Signed ad hoc only: on first start confirm it under System Settings > Privacy & Security"
echo ">> (Open Anyway), or: xattr -dr com.apple.quarantine \"/Applications/MSR Viewer.app\""
