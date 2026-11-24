#!/usr/bin/env bash
# Build ON THE ROBOT COMPUTER (aarch64, offline, no root, never contacts the robot): private Python 3.10 + wheels + CycloneDDS (compiled only if missing) + quick self-check.
#   ./build.sh [--skip-check] [--cdds-home DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
CHECK=1; CDDS_ARG=""
while [ $# -gt 0 ]; do case "$1" in --skip-check) CHECK=0;; --cdds-home) CDDS_ARG="$2"; shift;; -h|--help) sed -n 2,4p "$0"; exit 0;; *) echo "unknown argument $1"; exit 2;; esac; shift; done
step() { echo; echo "=== [$1/6] $2"; }

step 1 "system"
[ "$(uname -m)" = aarch64 ] || { echo "ERROR: this package carries aarch64 python/wheels; run it on the G1 computer (this is $(uname -m))"; exit 1; }
gl="$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0.0)"
[ "$(printf '%s\n2.28\n' "$gl" | sort -V | head -1)" = 2.28 ] || { echo "ERROR: glibc $gl < 2.28 (wheels need manylinux_2_28)"; exit 1; }
for t in tar sha256sum; do command -v $t >/dev/null || { echo "ERROR: missing $t"; exit 1; }; done
echo "aarch64, glibc $gl, $(nproc) cores, $(df -Pk . | awk 'NR==2{printf "%.1f", $4/1048576}') GiB free"
[ -f MANIFEST.sha256 ] && { sha256sum -c --quiet MANIFEST.sha256 && echo "package integrity OK (MANIFEST.sha256)"; }

step 2 "private Python 3.10"
PYTGZ="$(ls vendor/python/cpython-3.10*aarch64*.tar.gz | head -1)"
[ -x python/bin/python3.10 ] || tar xzf "$PYTGZ" -C "$HERE"
[ -x .venv/bin/python ] || python/bin/python3.10 -m venv .venv

step 3 "python dependencies (offline wheelhouse)"
.venv/bin/python -m pip --disable-pip-version-check install --no-index --find-links wheelhouse -r requirements-robot.txt

step 4 "CycloneDDS 0.10.2"
find_cdds() { for c in "$CDDS_ARG" "${CYCLONEDDS_HOME:-}" "$HERE/cdds" /opt/cyclonedds /usr/local /usr "$HOME/cyclonedds/install" "$HOME/unitree_sdk2/thirdparty"; do
  [ -n "$c" ] || continue; if ls "$c"/lib/libddsc.so* "$c"/lib/aarch64*/libddsc.so* >/dev/null 2>&1 && [ -f "$c/include/dds/dds.h" ]; then echo "$c"; return; fi; done; }
CDDS="$(find_cdds || true)"
if [ -z "$CDDS" ]; then
  for t in gcc g++ make cmake; do command -v $t >/dev/null || { echo "ERROR: need $t to compile CycloneDDS:  sudo apt install -y build-essential cmake  (then re-run ./build.sh)"; exit 1; }; done
  bash scripts/build_cyclonedds.sh "$HERE/cdds"; CDDS="$HERE/cdds"
fi
LIBDIR="$(dirname "$(ls "$CDDS"/lib/libddsc.so* "$CDDS"/lib/aarch64*/libddsc.so* 2>/dev/null | head -1)")"
{ echo "export CYCLONEDDS_HOME=\"$CDDS\""; echo "export LD_LIBRARY_PATH=\"$LIBDIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""; } > scripts/ft_env.local.sh
. scripts/ft_env.sh
.venv/bin/python -c "import cyclonedds" 2>/dev/null || CYCLONEDDS_HOME="$CDDS" .venv/bin/python -m pip --disable-pip-version-check install --no-index --find-links wheelhouse --no-build-isolation vendor/cyclonedds-0.10.2.tar.gz
echo "CycloneDDS: $CDDS"

step 5 "import check"
PYTHONPATH=".sdk_target:vendor/pylibs" .venv/bin/python -c "
import platform, numpy, onnxruntime, mujoco, torch, tensordict, cyclonedds, unitree_sdk2py
print('python', platform.python_version(), platform.machine(), '| numpy', numpy.__version__, '| onnxruntime', onnxruntime.__version__, '| torch', torch.__version__, '| cyclonedds ok | unitree_sdk2py ok')"

step 6 "self-check (policy stack parity + inference time; DDS loopback only, no robot)"
if [ $CHECK -eq 1 ]; then .venv/bin/python -B tools/selfcheck.py || { echo "SELFCHECK FAILED: do not run on the robot; send logs."; exit 1; }; else echo "skipped (--skip-check)"; fi
mkdir -p logs; chmod +x start.sh stop.sh ack.sh scripts/*.sh 2>/dev/null || true
echo; echo "BUILD OK (nothing was sent to any robot).  Next: ./start.sh"
