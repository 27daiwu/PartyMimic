"""Remote key logic: the layout is checked against wbc_nx/remote_start/include/gamepad.hpp (xKeySwitchUnion), chords, START double-click."""
import struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
from fasttrack.remote import KeyMap, ChordDetector, StartDoubleClick, start_pressed, DEFAULT_KEYS

# component order of the bitfield in gamepad.hpp (LSB first); F2 there == F3 here (same bit)
HPP_ORDER = ["R1", "L1", "START", "SELECT", "R2", "L2", "F1", "F3", "A", "B", "X", "Y", "UP", "RIGHT", "DOWN", "LEFT"]


def wireless(*names):
    """40-byte wireless_remote as the robot sends it: head[2], uint16 btn (little endian), floats, idle."""
    v = sum(1 << HPP_ORDER.index(n) for n in names)
    return bytes(struct.pack("<2sH", b"\x00\x00", v)) + bytes(36)


def test_layout_equals_gamepad_hpp_bit_order():
    km = KeyMap.load(paths.CONFIG / "keymap.json")
    assert km.verified and km.combos == {"load": ["F1", "F3"], "go": ["L1", "A"]}
    for i, n in enumerate(HPP_ORDER):
        assert km.keys[n] == (2 + i // 8, i % 8), n
        assert km.pressed(wireless(n)) == {n}, n
    assert dict(DEFAULT_KEYS) == dict(km.keys)

def test_chords_decode_exactly():
    km = KeyMap(); assert km.pressed(wireless("F1", "F3")) == {"F1", "F3"} and km.pressed(wireless("L1", "A")) == {"L1", "A"}
    assert km.pressed(wireless()) == frozenset() and km.pressed(b"") == frozenset()

def test_start_bit():
    assert start_pressed(wireless("START")) and not start_pressed(wireless("F1", "A")) and not start_pressed(b"")

def test_unmapped_bits_reported():
    km = KeyMap({"A": (3, 0)}); w = bytearray(40); w[3] = 0x03; assert km.unmapped_bits(bytes(w)) == [(3, 1)]

def test_unreadable_keymap_falls_back_to_the_default_layout(tmp_path):
    km = KeyMap.load(tmp_path / "nope.json"); assert km.keys["START"] == (2, 2) and km.combos["go"] == ["L1", "A"]

def test_chord_fires_once_needs_exact_set_and_hold():
    c = ChordDetector(["F1", "F3"], hold_s=0.15)
    out = [c.feed(t / 100, {"F1", "F3"}) for t in range(0, 40)]
    assert out.count(True) == 1 and out.index(True) >= 15
    assert not any(c.feed(0.5 + t / 100, {"F1", "F3"}) for t in range(30))             # still held: no re-fire
    assert not c.feed(1.0, set()) and not c.feed(1.1, set()) and not c.feed(1.25, set())
    assert [c.feed(1.3 + t / 100, {"F1", "F3"}) for t in range(30)].count(True) == 1   # re-armed after release

def test_chord_rejects_extra_partial_keys_and_short_taps():
    c = ChordDetector(["L1", "A"], hold_s=0.15)
    assert not any(c.feed(t / 100, {"L1", "A", "B"}) for t in range(50))
    c = ChordDetector(["L1", "A"], hold_s=0.15); assert not any(c.feed(t / 100, {"L1"}) for t in range(50))
    c = ChordDetector(["L1", "A"], hold_s=0.15); assert not any(c.feed(t / 100, {"L1", "A"} if t < 8 else set()) for t in range(50))
    c = ChordDetector(["F1", "F3"], hold_s=0.15)
    assert any([c.feed(t / 100, {"F1"} if t < 5 else {"F1", "F3"}) for t in range(40)])                   # staggered press is fine

def feed(det, seq): return [det.feed(t, p) for t, p in seq]

def test_double_click_fires_once():
    assert feed(StartDoubleClick(), [(0.00, 1), (0.05, 0), (0.14, 1), (0.20, 0)]) == [False, False, True, False]

def test_single_press_slow_pair_held_and_bounce_do_not_fire():
    assert not any(feed(StartDoubleClick(), [(0.0, 1), (0.1, 0), (0.3, 0)]))
    assert not any(feed(StartDoubleClick(), [(0.0, 1), (0.1, 0), (1.3, 1), (1.4, 0)]))
    assert not any(feed(StartDoubleClick(), [(t / 100, 1) for t in range(100)]))
    assert not any(feed(StartDoubleClick(), [(0.0, 1), (0.005, 0), (0.01, 1), (0.02, 0)]))
