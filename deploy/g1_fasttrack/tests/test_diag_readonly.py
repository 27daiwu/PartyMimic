"""The diagnostic must not be able to move the robot: static scan + whitelist behaviour."""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = ["enable_write", "switch_to_user", "switch_to_internal", ".send(", "SwitchToUserCtrl", "SwitchToInternalCtrl", "SetFsmId", "SetVelocity",
             "ReleaseMode", "SelectMode", "ChannelPublisher(", "read_only=False", "take_over", "estop"]

def code_lines(p):
    return [l for l in Path(p).read_text().splitlines() if not l.strip().startswith(("#", '"""')) ]

def test_diag_has_no_write_path():
    txt = "\n".join(code_lines(ROOT / "tools/diag.py"))
    # strip docstring + string literals that merely NAME the forbidden calls in the guarantees text
    body = txt.split("def main(", 1)[1]
    hits = [f for f in FORBIDDEN if f in body]
    assert not hits, hits

def test_diag_only_checkmode_on_motion_switcher():
    body = (ROOT / "tools/diag.py").read_text()
    calls = set(re.findall(r"msc\.(\w+)\(", body)); assert calls <= {"SetTimeout", "Init", "CheckMode"}, calls

def test_whitelist_blocks_state_changing_api_ids():
    from fasttrack import paths; paths.setup_sys_path(); paths.setup_offline_env()
    from fasttrack.link import DdsUserCtrlLink
    ids = DdsUserCtrlLink.READ_ONLY_API_IDS
    for bad in (7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7110, 7111): assert bad not in ids
    assert set(ids) == {7001, 7002, 7003, 7004, 7005}

def test_read_only_link_refuses_writes():
    from fasttrack import paths; paths.setup_sys_path(); paths.setup_offline_env()
    import fasttrack.link as L, inspect
    src = inspect.getsource(L.DdsUserCtrlLink)
    assert src.count("raise RuntimeError(\"read-only link cannot") >= 3
