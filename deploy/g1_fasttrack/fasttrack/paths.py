"""Locations of the frozen P2 artifacts and the pinned EGalahad/sim2real checkout. Override with env vars."""
from __future__ import annotations
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent                       # deploy/g1_fasttrack
REPO = PKG_ROOT.parent.parent                # PartyMimic
P2 = REPO / "results/p2/run_20261119"
ART = P2 / "artifacts"
POLICY_YAML = Path(os.environ.get("FT_POLICY_YAML", ART / "ssb_seed0_it1500.yaml"))
POLICY_ONNX = Path(os.environ.get("FT_POLICY_ONNX", ART / "ssb_seed0_it1500.onnx"))
MOTION_NPZ = Path(os.environ.get("FT_MOTION", P2 / "sim2real_adapter/legacy_ssb/motion.npz"))
SIM2REAL = Path(os.environ.get("FT_SIM2REAL", REPO.parent / "sim2real_party_p0_patched"))
ADAPTER_CACHE = P2 / "sim2real_adapter/cache"   # offline HF snapshot of the stock G1 MJCF (sim host only)
SDK_TARGET = PKG_ROOT / ".sdk_target"            # isolated unitree_sdk2py (official, pinned commit)

# frozen hashes (P2_DANCE_REPORT.md section 2)
EXPECTED_SHA256 = {
    "onnx": "0c51d5f26de2698ec6922cb3005410f714a1ac573b665b22b134be389fb1b9a6",
    "ssb_motion": "d4d4d936a670d54e1cca5bf0f059f1ee9fc79f17733a64a859b1e929d6084cc6",
}


def setup_sys_path(sdk: bool = True) -> None:
    for p in (str(SIM2REAL), str(PKG_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if sdk and SDK_TARGET.is_dir() and str(SDK_TARGET) not in sys.path:
        sys.path.append(str(SDK_TARGET))


def setup_offline_env() -> None:
    """Make EGalahad's hf:// MJCF resolution work offline from the P2 private snapshot."""
    for k, v in {
        "HF_HOME": str(ADAPTER_CACHE / "huggingface"),
        "HF_HUB_CACHE": str(ADAPTER_CACHE / "huggingface/hub"),
        "HUGGINGFACE_HUB_CACHE": str(ADAPTER_CACHE / "huggingface/hub"),
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }.items():
        os.environ[k] = v
    for k in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(k, None)


def sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
