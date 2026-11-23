#!/usr/bin/env python3
"""TEST ONLY: a second process that publishes rt/user_lowcmd on the loopback sim domain (simulates a competing controller)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path()
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
ChannelFactoryInitialize(77, "lo")
pub = ChannelPublisher("rt/user_lowcmd", LowCmd_); pub.Init(); m = unitree_hg_msg_dds__LowCmd_()
t_end = time.time() + 15
while time.time() < t_end:
    pub.Write(m); time.sleep(0.01)
