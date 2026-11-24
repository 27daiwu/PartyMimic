import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(); paths.setup_offline_env()
from fasttrack.crc import lowcmd_crc
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

def test_pure_python_crc_matches_official_so():
    c, m = CRC(), unitree_hg_msg_dds__LowCmd_(); rnd = random.Random(1)
    for _ in range(50):
        m.mode_machine = rnd.choice([0, 5, 7]); m.mode_pr = rnd.choice([0, 1])
        for x in m.motor_cmd:
            x.mode = rnd.choice([0, 1]); x.q = rnd.uniform(-3, 3); x.dq = rnd.uniform(-2, 2); x.tau = rnd.uniform(-9, 9); x.kp = rnd.uniform(0, 120); x.kd = rnd.uniform(0, 6)
        assert lowcmd_crc(m) == c.Crc(m)

def test_crc_excludes_its_own_field():
    m = unitree_hg_msg_dds__LowCmd_(); a = lowcmd_crc(m); m.crc = 12345; assert lowcmd_crc(m) == a

def test_crc_sensitive_to_every_motor_field():
    m = unitree_hg_msg_dds__LowCmd_(); base = lowcmd_crc(m)
    for f in ("mode", "q", "dq", "tau", "kp", "kd"):
        m2 = unitree_hg_msg_dds__LowCmd_(); setattr(m2.motor_cmd[28], f, 1); assert lowcmd_crc(m2) != base, f
