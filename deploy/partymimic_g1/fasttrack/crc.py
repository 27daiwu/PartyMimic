"""CRC32 of a unitree_hg LowCmd_ (Unitree's non-reflected CRC32, poly 0x04c11db7, init 0xffffffff, no final xor), pure Python.

Wire layout = unitree_sdk2py/utils/crc.py __packFmtHGLowCmd ('<2B2x' + 'B3x5fI'*35 + '5I', 1004 bytes): mode_pr, mode_machine,
35 x {mode, q, dq, tau, kp, kd, reserve}, reserve[4], crc(placeholder, excluded: last word is not hashed).
The pip install of unitree_sdk2py (from git) does NOT ship utils/lib/crc_*.so; this module removes that dependency and is
cross-checked against the official .so in tests/test_crc.py.
"""
from __future__ import annotations
import struct

_POLY = 0x04C11DB7
_FMT = "<2B2x" + "B3x5fI" * 35 + "5I"


def _crc32_words(words) -> int:
    crc = 0xFFFFFFFF
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ _POLY
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= _POLY
            xbit >>= 1
    return crc


# table-free but vectorised enough: 251 words * 32 iterations ~ 8k ops (~3 ms in CPython) -> acceptable at 50-100 Hz.
def lowcmd_crc(msg) -> int:
    raw = [msg.mode_pr, msg.mode_machine]
    for c in msg.motor_cmd:
        raw += [c.mode, c.q, c.dq, c.tau, c.kp, c.kd, c.reserve]
    raw += list(msg.reserve) + [0]
    packed = struct.pack(_FMT, *raw)
    words = struct.unpack("<%dI" % (len(packed) // 4), packed)
    return _crc32_words(words[:-1])
