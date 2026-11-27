#!/usr/bin/env bash
# Build ON THE ROBOT COMPUTER (aarch64, offline, no root, never contacts the robot):  ./build.sh [--jobs N] [--ort-root DIR] [--skip-check] [--clean]
#   compiles the C++17 deployment against what the robot already has (Unitree SDK2 + CycloneDDS in /usr/local, Eigen in /usr/include/eigen3),
#   uses a system ONNX Runtime if one is installed, else the bundled aarch64 one (third_party/onnxruntime),
#   then verifies the frozen artifacts, the dynamic-library dependencies and runs `partymimic selfcheck` (offline, no robot connection).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
JOBS="$(nproc 2>/dev/null || echo 2)"; ORT_ROOT=""; CHECK=1; CLEAN=0
while [ $# -gt 0 ]; do case "$1" in
  --jobs) JOBS="$2"; shift;; --ort-root) ORT_ROOT="$2"; shift;; --skip-check) CHECK=0;; --clean) CLEAN=1;;
  -h|--help) sed -n 2,5p "$0"; exit 0;; *) echo "unknown argument $1"; exit 2;; esac; shift; done
step() { echo; echo "=== [$1/6] $2"; }
die() { echo "ERROR: $*" >&2; exit 1; }

step 1 "toolchain"
ARCH="$(uname -m)"; echo "machine: $ARCH, $(nproc) cores, $(df -Pk . | awk 'NR==2{printf "%.1f", $4/1048576}') GiB free"
for t in g++ cmake make sha256sum; do command -v $t >/dev/null || die "missing $t  (sudo apt install -y build-essential cmake)"; done
gccv="$(g++ -dumpversion | cut -d. -f1)"; [ "${gccv:-0}" -ge 7 ] 2>/dev/null || die "g++ $gccv is too old for C++17 (need >= 7)"
cmv="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)"
[ "$(printf '%s\n3.16\n' "$cmv" | sort -V | head -1)" = 3.16 ] || die "cmake $cmv < 3.16"
echo "g++ $(g++ -dumpversion), cmake $cmv, glibc $(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [ "$ARCH" != aarch64 ] && [ -z "$ORT_ROOT" ]; then die "this package bundles the aarch64 ONNX Runtime: run it on the G1 computer (this is $ARCH), or pass --ort-root DIR with a matching ONNX Runtime"; fi

step 2 "frozen artifacts"
( cd assets && sha256sum -c --quiet SHA256SUMS ) || die "assets do not match SHA256SUMS (corrupted unzip?)"
echo "assets/policy.onnx, policy_params.txt, motion.bin, fk_chain.txt: sha256 OK"
( cd config && [ -f modes.conf ] ) || die "config/modes.conf missing"

step 3 "dependencies already on the robot"
[ -d /usr/include/eigen3/Eigen ] || [ -d /usr/local/include/eigen3/Eigen ] || [ -d third_party/eigen3/Eigen ] || die "Eigen3 headers not found (sudo apt install -y libeigen3-dev; wbc_nx/remote_start uses /usr/include/eigen3 too)"
echo "Eigen3: found"
SDKLIB="$(ls /usr/local/lib/libunitree_sdk2.a /opt/unitree_robotics/lib/libunitree_sdk2.a "$HOME"/unitree_sdk2/lib/aarch64/libunitree_sdk2.a 2>/dev/null | head -1 || true)"
[ -n "$SDKLIB" ] || die "Unitree SDK2 (libunitree_sdk2.a + CycloneDDS libddsc/libddscxx in /usr/local) not found - it is what wbc_nx/remote_start is built with"
echo "Unitree SDK2: $SDKLIB"
if [ -z "$ORT_ROOT" ]; then
  for c in /usr/local /usr /opt/onnxruntime "$HOME/onnxruntime"; do
    if [ -f "$c/include/onnxruntime_cxx_api.h" ] || [ -f "$c/include/onnxruntime/onnxruntime_cxx_api.h" ]; then
      ls "$c"/lib/libonnxruntime.so* "$c"/lib/aarch64-linux-gnu/libonnxruntime.so* >/dev/null 2>&1 && { ORT_ROOT="$c"; echo "system ONNX Runtime: $c"; break; }
    fi
  done
fi
if [ -z "$ORT_ROOT" ]; then
  ORT_ROOT="$HERE/third_party/onnxruntime"; L="$ORT_ROOT/lib"
  [ -f "$L/libonnxruntime.so.1.22.0" ] || die "bundled ONNX Runtime missing"
  ln -sf libonnxruntime.so.1.22.0 "$L/libonnxruntime.so.1"; ln -sf libonnxruntime.so.1 "$L/libonnxruntime.so"
  echo "bundled ONNX Runtime 1.22.0 (aarch64): $ORT_ROOT"
fi

step 4 "compile"
[ $CLEAN -eq 1 ] && rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPM_ORT_ROOT="$ORT_ROOT" > build.cmake.log 2>&1 || { tail -30 build.cmake.log; die "cmake configure failed (see build.cmake.log)"; }
cmake --build build -j"$JOBS" --target partymimic > build.make.log 2>&1 || { tail -40 build.make.log; die "compile failed (see build.make.log)"; }
[ -x build/partymimic ] || die "build/partymimic missing"
ls -la build/partymimic | awk '{print $5" bytes  "$9}'

step 5 "dynamic-library dependencies"
export LD_LIBRARY_PATH="/usr/local/lib:$ORT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd build/partymimic | sed 's/^/  /'
if ldd build/partymimic | grep -q "not found"; then
  echo "ERROR: unresolved libraries above. Typical fix: they are in the robot's wbc_install/libs (libddsc.so.0, libddscxx.so.0): export LD_LIBRARY_PATH=/home/unitree/wbc_install/libs:\$LD_LIBRARY_PATH" >&2; exit 1; fi
echo "all shared libraries resolved"
{ echo "# written by build.sh: runtime library path used by start.sh"; echo "export LD_LIBRARY_PATH=\"/usr/local/lib:$ORT_ROOT/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""; } > build/env.sh

step 6 "self-check (offline: policy numerics, timing, FK, CRC, control logic; DDS/robot are NOT used)"
if [ $CHECK -eq 1 ]; then build/partymimic selfcheck || die "SELFCHECK FAILED: do not run on the robot; send build.make.log and this output"; else echo "skipped (--skip-check)"; fi
mkdir -p logs; chmod +x start.sh stop.sh ack.sh status.sh export_logs.sh build.sh 2>/dev/null || true
echo; echo "BUILD OK (nothing was sent to any robot).  Next: ./start.sh   (nothing else: no authorisation, no confirmation; then F1+F3 -> READY -> L1+A.  Other modes: ./start.sh --list; logs: ./export_logs.sh)"
