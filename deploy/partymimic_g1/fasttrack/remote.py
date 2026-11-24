"""Remote-controller key decoding, chords and START double-click (pure logic, no I/O).

KEY LAYOUT = the one already used and verified on the robot by wbc_nx/remote_start (include/gamepad.hpp, src/remote.cpp):
`rt/lowstate.wireless_remote` (40 bytes) -> bytes 2..3 are a little-endian uint16 `xKeySwitchUnion`:
    byte 2 : bit0 R1, bit1 L1, bit2 START, bit3 SELECT, bit4 R2, bit5 L2, bit6 F1, bit7 F2 (called F3 on our remote: same bit)
    byte 3 : bit0 A,  bit1 B,  bit2 X,      bit3 Y,      bit4 UP, bit5 RIGHT, bit6 DOWN, bit7 LEFT
Rules taken over from remote_start: edge triggered (no auto-repeat while held); a function needs EXACTLY its keys down, any extra key
cancels it; every key change is logged (logs/keys.log) with the raw bytes, so an unexpected key shows up as `unmapped`.
"""
from __future__ import annotations

import json
from pathlib import Path

START_BYTE, START_MASK = 2, 0x04

DEFAULT_KEYS = {
    "R1": (2, 0), "L1": (2, 1), "START": (2, 2), "SELECT": (2, 3), "R2": (2, 4), "L2": (2, 5), "F1": (2, 6), "F3": (2, 7),
    "A": (3, 0), "B": (3, 1), "X": (3, 2), "Y": (3, 3), "UP": (3, 4), "RIGHT": (3, 5), "DOWN": (3, 6), "LEFT": (3, 7)}
DEFAULT_COMBOS = {"load": ["F1", "F3"], "go": ["L1", "A"]}


def start_pressed(wireless_remote) -> bool:
    try:
        return bool(wireless_remote[START_BYTE] & START_MASK)
    except Exception:
        return False


class StartDoubleClick:
    """feed(t, pressed) -> True exactly when the 2nd rising edge arrives within `window_s` of the 1st (the SDK's own double-click rule)."""

    def __init__(self, window_s: float = 0.3, min_gap_s: float = 0.03):
        self.window_s, self.min_gap_s = window_s, min_gap_s
        self._prev = False
        self._last_edge: float | None = None
        self._last_release: float | None = None
        self.edges = 0

    def feed(self, t: float, pressed: bool) -> bool:
        fired = False
        if pressed and not self._prev:
            self.edges += 1
            if (self._last_edge is not None and t - self._last_edge <= self.window_s
                    and (self._last_release is not None and t - self._last_release >= 0.0) and t - self._last_edge >= self.min_gap_s):
                fired = True
                self._last_edge = None
            else:
                self._last_edge = t
        if (not pressed) and self._prev:
            self._last_release = t
        self._prev = pressed
        return fired


class KeyMap:
    def __init__(self, keys=None, verified=True, source="remote_start/gamepad.hpp", combos=None):
        self.keys = {k: tuple(v) for k, v in (keys or DEFAULT_KEYS).items()}
        self.verified, self.source = bool(verified), source
        self.combos = combos or dict(DEFAULT_COMBOS)

    @classmethod
    def load(cls, path) -> "KeyMap":
        try:
            d = json.loads(Path(path).read_text())
            return cls(d.get("keys"), d.get("verified", True), d.get("source", str(path)), d.get("combos"))
        except Exception:
            return cls(source="built-in default (config/keymap.json unreadable)")

    def pressed(self, raw) -> frozenset:
        try:
            return frozenset(n for n, (b, bit) in self.keys.items() if raw[b] & (1 << bit))
        except Exception:
            return frozenset()

    def unmapped_bits(self, raw) -> list:
        known = {(b, bit) for b, bit in self.keys.values()}
        try:
            return [(b, bit) for b in (2, 3) for bit in range(8) if raw[b] & (1 << bit) and (b, bit) not in known]
        except Exception:
            return []


class ChordDetector:
    """Fires ONCE when the pressed set equals exactly `combo` for >= hold_s; re-arms only after ALL keys were released (>= release_s)."""

    def __init__(self, combo, hold_s: float = 0.15, release_s: float = 0.2):
        self.combo, self.hold_s, self.release_s = frozenset(combo), hold_s, release_s
        self._since = None; self._fired = False; self._free_since = None; self._armed = True

    def feed(self, t: float, pressed) -> bool:
        pressed = frozenset(pressed)
        if not pressed:
            if self._free_since is None: self._free_since = t
            if t - self._free_since >= self.release_s: self._armed = True; self._fired = False
            self._since = None
            return False
        self._free_since = None
        if pressed == self.combo and self._armed and not self._fired:
            if self._since is None: self._since = t
            if t - self._since >= self.hold_s:
                self._fired = True; self._armed = False
                return True
        else:
            self._since = None
        return False
