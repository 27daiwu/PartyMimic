#!/usr/bin/env bash
# DEV PC ONLY: assemble deploy/dist/partymimic_cpp_p4.1.zip from an explicit FILE WHITELIST (nothing else can enter the package):
#   C++ deployment sources + needed headers, bundled ONNX Runtime aarch64 (headers + .so), frozen ONNX, motion data, configs, build/start scripts, README.
# Never packed: python, torch, mujoco, robot XML/STL models, training code, tests/dev tooling, evidence/logs, caches.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; OUT="$HERE/../dist"; ST="$OUT/stage_cpp_p4.1/partymimic_cpp"
cd "$HERE"
WL=(README.md EXIT_POLICY.md SAFETY_AUDIT.md CMakeLists.txt build.sh start.sh stop.sh ack.sh config/site.conf
    assets/policy.onnx assets/policy_params.txt assets/motion.bin assets/fk_chain.txt assets/SHA256SUMS
    include/gamepad.hpp
    third_party/onnxruntime/LICENSE third_party/onnxruntime/README.txt third_party/onnxruntime/lib/libonnxruntime.so.1.22.0
    third_party/onnxruntime/include/onnxruntime_c_api.h third_party/onnxruntime/include/onnxruntime_cxx_api.h
    third_party/onnxruntime/include/onnxruntime_cxx_inline.h third_party/onnxruntime/include/onnxruntime_float16.h)
for f in src/*.cpp include/pm/*.hpp; do WL+=("$f"); done
for f in "${WL[@]}"; do [ -f "$f" ] || { echo "whitelist file missing: $f"; exit 1; }; done
# hard denylist: even a whitelisted path must not look like one of these
BAD='(\.py|\.pyc|\.pt|\.pth|\.npz|\.npy|\.xml|\.stl|\.urdf|\.mjcf|\.ipynb|\.csv|\.log|\.zip|\.tar\.gz)$|(^|/)(__pycache__|tests?|dev|results|\.git|logs|build|wheelhouse|python|vendor)(/|$)'
for f in "${WL[@]}"; do echo "$f" | grep -qiE "$BAD" && { echo "DENIED by the package denylist: $f"; exit 1; }; done
bash -n build.sh start.sh stop.sh ack.sh
rm -rf "$OUT/stage_cpp_p4.1" "$OUT/partymimic_cpp_p4.1.zip"; mkdir -p "$ST"
for f in "${WL[@]}"; do mkdir -p "$ST/$(dirname "$f")"; cp -p "$f" "$ST/$f"; done
chmod +x "$ST"/{build.sh,start.sh,stop.sh,ack.sh}
( cd "$ST/.." && zip -q -r -9 -X "$OUT/partymimic_cpp_p4.1.zip" partymimic_cpp )
( cd "$OUT" && sha256sum partymimic_cpp_p4.1.zip > partymimic_cpp_p4.1.zip.sha256 )
echo "== package contents (uncompressed bytes)"; ( cd "$ST/.." && find partymimic_cpp -type f -printf '%s %p\n' | sort -k2 | awk '{printf "%10d  %s\n",$1,$2}' )
echo "== zip"; ls -l "$OUT/partymimic_cpp_p4.1.zip" | awk '{printf "%d bytes = %.2f MiB\n",$5,$5/1048576}'; cat "$OUT/partymimic_cpp_p4.1.zip.sha256"
echo "== compressed size per component"; unzip -v "$OUT/partymimic_cpp_p4.1.zip" | awk 'NR>3 && $1+0>0 {print $3, $8}' | sort -rn | head -6
n_bad=$(unzip -Z1 "$OUT/partymimic_cpp_p4.1.zip" | grep -ciE "$BAD" || true); [ "$n_bad" = 0 ] && echo "denylist scan of the zip: clean" || { echo "denylist hit in zip!"; exit 1; }
