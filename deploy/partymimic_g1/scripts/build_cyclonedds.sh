#!/usr/bin/env bash
# Build the CycloneDDS C library 0.10.2 (the version unitree_sdk2py/cyclonedds-python 0.10.2 expects) from the bundled source.
# usage: scripts/build_cyclonedds.sh [PREFIX]     (default PREFIX = <package>/cdds).   Needs gcc g++ make cmake>=3.16.  No network.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PREFIX="${1:-$HERE/cdds}"
SRC_TGZ="${CDDS_SRC:-$HERE/vendor/cyclonedds-0.10.2-src.tar.gz}"
[ -f "$SRC_TGZ" ] || { echo "missing $SRC_TGZ"; exit 1; }
for t in gcc g++ make cmake; do command -v $t >/dev/null || { echo "missing $t: sudo apt install -y build-essential cmake"; exit 1; }; done
W="$(mktemp -d "${TMPDIR:-/tmp}/cdds_build.XXXXXX")"; trap 'rm -rf "$W"' EXIT
tar xzf "$SRC_TGZ" -C "$W"; S="$(ls -d "$W"/cyclonedds-*)"
cmake -S "$S" -B "$W/b" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_TESTING=OFF -DBUILD_DDSPERF=OFF -DENABLE_SSL=NO -DENABLE_SHM=NO -DBUILD_DOCS=OFF >"$W/cmake.log" 2>&1 || { tail -30 "$W/cmake.log"; exit 1; }
cmake --build "$W/b" --target install -j"$(nproc)" >"$W/build.log" 2>&1 || { tail -30 "$W/build.log"; exit 1; }
ls "$PREFIX"/lib*/libddsc.so* >/dev/null && [ -f "$PREFIX/include/dds/dds.h" ] && echo "CycloneDDS installed in $PREFIX"
