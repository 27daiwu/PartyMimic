"""prepare / preview: robot_motion.pkl -> 50 Hz SuperMimic NPZ/CSV (existing converter), kinematic QA, leading-glitch trim, padded training clip.

Reuses: unitree_rl_mjlab/extensions/supermimic/scripts/data/gmr_pkl_to_npz.py (the converter used for SSB / YTHN, unchanged) and the YTHN
padded-clip recipe (1 s hold of the first frame + dance + 1 s hold of the last frame, the clip robust_d trains on). The source PKL is never modified.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import sys

import numpy as np

from .core import DATASETS, PY, REPO, SUPERMIMIC, Dance, Logger, die, now, py_env, say, sha256, sh, write_json, read_json

PAD = 50  # hold frames before / after the dance (matches evaluator PAD and the C++ lead-in of 1 s)
# a leading frame is a tracker-init glitch when ANY of these is exceeded by the motion from that frame to the next (source fps units)
GLITCH = dict(joint_vel_rad_s=30.0, root_ang_vel_rad_s=20.0, root_lin_vel_m_s=6.0)


def _load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    miss = {"fps", "root_pos", "root_rot", "dof_pos"} - set(d)
    if miss:
        die(f"{path}: missing PKL fields {sorted(miss)} (expected a video2robot/GMR G1 robot_motion.pkl)")
    return d


def _quat_ang_speed(q_xyzw, fps):
    q = np.asarray(q_xyzw, np.float64)
    dot = np.abs(np.sum(q[1:] * q[:-1], -1)).clip(0, 1)
    return 2 * np.arccos(dot) * fps


def detect_leading_glitch(d, max_trim: int = 3) -> tuple[int, list[dict]]:
    """Number of leading source frames to drop (tracker-init glitch, e.g. YTHN frame 0: 53 rad/s joint, 7.2 m/s root) + per-frame evidence."""
    fps = float(d["fps"])
    jv = np.abs(np.diff(np.asarray(d["dof_pos"], np.float64), axis=0)).max(-1) * fps
    lv = np.linalg.norm(np.diff(np.asarray(d["root_pos"], np.float64), axis=0), axis=-1) * fps
    av = _quat_ang_speed(d["root_rot"], fps)
    ev, n = [], 0
    for k in range(min(max_trim, len(jv))):
        row = dict(frame=k, joint_vel=float(jv[k]), root_lin_vel=float(lv[k]), root_ang_vel=float(av[k]))
        bad = jv[k] > GLITCH["joint_vel_rad_s"] or av[k] > GLITCH["root_ang_vel_rad_s"] or lv[k] > GLITCH["root_lin_vel_m_s"]
        row["glitch"] = bool(bad)
        ev.append(row)
        if bad and n == k:
            n = k + 1
        else:
            break
    return n, ev


def _converter():
    spec = importlib.util.spec_from_file_location("gmr_pkl_to_npz", SUPERMIMIC / "scripts/data/gmr_pkl_to_npz.py")
    sys.path.insert(0, str(SUPERMIMIC / "src"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_padded(src: str, dst: str, n: int = PAD):
    z = np.load(src, allow_pickle=True)
    out = {k: z[k] for k in z.files}

    def pad(a, zero=False):
        h, t = np.repeat(a[:1], n, 0), np.repeat(a[-1:], n, 0)
        if zero:
            h, t = np.zeros_like(h), np.zeros_like(t)
        return np.concatenate([h, a, t], 0)
    for k in ("joint_pos", "body_pos_w", "body_quat_w"):
        out[k] = pad(z[k])
    for k in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        out[k] = pad(z[k], zero=True)
    np.savez(dst, **out)
    return out["joint_pos"].shape


def qa(dance: Dance, src_pkl, src_sha: str, trimmed: int, glitch_ev: list) -> dict:
    """Kinematic QA of the converted 50 Hz clip. verdict FAIL blocks training; WARN is reported."""
    conv = _converter()
    from supermimic.data.motion_convert import G1_JOINT_NAMES, load_g1_model
    z = np.load(dance.npz, allow_pickle=True)
    jp, jv = z["joint_pos"], z["joint_vel"]
    fps = float(z["fps"][0])
    T = len(jp)
    bi = list(z["body_names"]).index("pelvis")
    rp, rq = z["body_pos_w"][:, bi], z["body_quat_w"][:, bi]
    model = load_g1_model()

    class _M:  # tiny adapter for the converter's ground_clearance
        num_frames = T
        root_pos, root_quat, joint_pos = rp, rq, jp
    clr, worst = conv.ground_clearance(model, _M)
    dots = np.abs(np.sum(rq[1:] * rq[:-1], -1))
    flips = int(np.sum(np.sum(rq[1:] * rq[:-1], -1) < 0))
    lim = model.jnt_range[1:]
    viol = float(np.max(np.maximum(lim[:, 0] - jp, jp - lim[:, 1])))
    vel_lim = np.array([32, 32, 32, 20, 37, 37, 32, 32, 32, 20, 37, 37, 32, 0, 0, 37, 37, 37, 37, 37, 22, 22, 37, 37, 37, 37, 37, 22, 22], float)  # G1 hardware (deploy policy_params)
    known = vel_lim > 0
    over = float(np.max(np.abs(jv)[:, known] / vel_lim[known]))
    root_lin = np.linalg.norm(z["body_lin_vel_w"][:, bi], axis=-1)
    root_ang = np.linalg.norm(z["body_ang_vel_w"][:, bi], axis=-1)
    checks = {
        "finite": bool(all(np.isfinite(z[k]).all() for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"))),
        "joint_order_exact": list(z["joint_names"]) == list(G1_JOINT_NAMES),
        "pelvis_first": list(z["body_names"])[0] == "pelvis",
        "fps_50": abs(fps - 50.0) < 1e-6,
        "duration_ok(2s..60s)": 2.0 <= T / fps <= 60.0,
        "ground_clearance_ok": clr >= -1e-6,
        "joint_limit_violation_ok(<=0.05rad)": viol <= 0.05,
    }
    warns = []
    if over > 1.0:
        warns.append(f"joint velocity reaches {over:.2f}x the hardware limit (feasibility risk)")
    if T / fps > 12.0:
        warns.append(f"dance is {T / fps:.1f}s > 12 s (default site.conf max_run_s): `pm package` writes an explicit per-package max_run_s override (recorded in ADMISSION.json run_caps); dances > 60 s (compile-time ABS_MAX_RUN_S) are refused. Long dances are outside the validated envelope")
    if root_lin.max() > 5.0:
        warns.append(f"root linear speed up to {root_lin.max():.1f} m/s")
    if flips:
        warns.append(f"{flips} quaternion sign flips (harmless for matrix-based metrics)")
    if trimmed:
        warns.append(f"dropped {trimmed} leading source frame(s): tracker-init glitch")
    verdict = "FAIL" if not all(checks.values()) else ("WARN" if warns else "PASS")
    out = dict(dance=dance.name, time=now(), verdict=verdict, checks=checks, warnings=warns,
               source=dict(path=str(src_pkl), sha256=src_sha, trimmed_leading_frames=trimmed, glitch_evidence=glitch_ev),
               clip=dict(frames=T, fps=fps, duration_s=T / fps, root_z_offset_m=float(z["root_z_offset_m"][0]) if "root_z_offset_m" in z.files else None,
                         min_ground_clearance_m=float(clr), worst_geom=str(worst), joint_limit_violation_rad=viol,
                         max_joint_vel_rad_s=float(np.abs(jv).max()), max_joint_vel_over_hw_limit=over, max_root_lin_vel=float(root_lin.max()),
                         max_root_ang_vel=float(root_ang.max()), quat_sign_flips=flips, quat_adjacent_min_abs_dot=float(dots.min()),
                         root_xy_travel_m=float(np.linalg.norm(rp[-1, :2] - rp[0, :2])), root_height_min=float(rp[:, 2].min()), root_height_max=float(rp[:, 2].max())),
               npz_sha256=sha256(dance.npz))
    return out


def prepare(dance: Dance, source: str | None = None, trim: str = "auto", force: bool = False, log=None) -> dict:
    log = log or Logger(dance.dir / "logs" / "prepare.log")
    src = dance.source_pkl(source)
    if not src.exists():
        avail = sorted(p.name for p in dance.src_dir.glob("*.pkl")) if dance.src_dir.exists() else []
        die(f"source not found: {src}" + (f"  (same folder has: {', '.join(avail)}; pick one with --source)" if avail else f"\n  dataset root: {DATASETS} ({'exists' if DATASETS.exists() else 'MISSING'})"))
    meta_path = dance.metrics / "prepare.json"
    sha_src = sha256(src)
    prev = read_json(meta_path)
    if prev and not force and prev.get("source_sha256") == sha_src and dance.npz.exists() and dance.padded_npz.exists() and prev.get("trim") in (trim, "adopted"):
        log("prepare: up to date (source sha256 unchanged); use --force to redo")
        return prev
    legacy_sha = (dance.metrics / "source_sha256.txt")
    if not prev and not force and dance.npz.exists() and dance.padded_npz.exists() and legacy_sha.exists() and legacy_sha.read_text().split()[0] == sha_src:
        # pre-pipeline products of the SAME source (e.g. YTHN): adopt in place, never regenerate (a trainer / evaluator may be reading them)
        meta = dict(time=now(), source=str(src), source_sha256=sha_src, trim="adopted", trimmed=None, npz=str(dance.npz), padded=str(dance.padded_npz),
                    npz_sha256=sha256(dance.npz), qa_verdict="ADOPTED", note="pre-pipeline conversion adopted unchanged")
        write_json(meta_path, meta)
        log("prepare: adopted existing conversion of the same source (sha256 matches metrics/source_sha256.txt)")
        return meta
    d = _load_pkl(src)
    n_trim, ev = (detect_leading_glitch(d) if trim == "auto" else (int(trim), []))
    log(f"source {src} frames={len(d['root_pos'])} fps={d['fps']} sha256={sha_src[:12]}  leading-glitch trim={n_trim}")
    conv_src = src
    if n_trim:
        for k in ("root_pos", "root_rot", "dof_pos", "local_body_pos"):
            if k in d:
                d[k] = d[k][n_trim:]
        d["num_frames"] = len(d["root_pos"])
        dp = dance.motions / "derived_pkl" / f"robot_motion_trim{n_trim}.pkl"
        dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, "wb") as f:
            pickle.dump(d, f)
        (dp.parent / "README.txt").write_text(f"derived: dropped {n_trim} leading source frame(s) (tracker-init glitch, evidence: {json.dumps(ev)}). Source pkl untouched: {src} sha256 {sha_src}\n")
        conv_src = dp
    out_dir = dance.npz.parent
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    r = sh([PY, SUPERMIMIC / "scripts/data/gmr_pkl_to_npz.py", "--input", conv_src, "--output", dance.npz, "--overwrite"],
           env=py_env({"PYTHONPATH": f"{SUPERMIMIC / 'src'}:{REPO / 'src'}"}), log=dance.dir / "logs" / "00_convert.log")
    if r.returncode:
        die(f"converter failed (see {dance.dir / 'logs/00_convert.log'})", 3)
    if dance.padded_npz.exists():
        dance.padded_npz.unlink()
    dance.padded_dir.mkdir(parents=True, exist_ok=True)
    shp = make_padded(str(dance.npz), str(dance.padded_npz))
    log("padded clip", dance.padded_npz, shp)
    rep = qa(dance, src, sha_src, n_trim, ev)
    rep["source_sha256_after"] = sha256(src)
    rep["source_unchanged"] = rep["source_sha256_after"] == sha_src
    rep["padded"] = dict(path=str(dance.padded_npz), frames=int(shp[0]), pad_frames=PAD)
    write_json(dance.metrics / "qa_kinematic.json", rep)
    meta = dict(time=now(), source=str(src), source_sha256=sha_src, trim=trim, trimmed=n_trim, npz=str(dance.npz), padded=str(dance.padded_npz), qa_verdict=rep["verdict"])
    write_json(meta_path, meta)
    (dance.metrics / "source_sha256.txt").write_text(f"{sha_src}  {src}\n")
    log("QA", rep["verdict"], rep["warnings"], f"{rep['clip']['frames']} frames, {rep['clip']['duration_s']:.2f}s")
    if rep["verdict"] == "FAIL":
        die(f"kinematic QA FAILED: {[k for k, v in rep['checks'].items() if not v]} (see {dance.metrics / 'qa_kinematic.json'})", 4)
    return meta


def preview(dance: Dance, video: bool = False, log=None) -> dict:
    """QA text summary + overview figure (+ optional MuJoCo EGL video). Needs `prepare` to have run."""
    if not dance.npz.exists():
        die(f"{dance.npz} missing: run `./pm prepare {dance.name}` first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(dance.npz, allow_pickle=True)
    jp, jv = z["joint_pos"], z["joint_vel"]
    T, fps = len(jp), float(z["fps"][0])
    t = np.arange(T) / fps
    bi = list(z["body_names"]).index("pelvis")
    rp = z["body_pos_w"][:, bi]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].plot(rp[:, 0], rp[:, 1]); ax[0, 0].set_title("root XY (m)"); ax[0, 0].axis("equal"); ax[0, 0].plot(*rp[0, :2], "go"); ax[0, 0].plot(*rp[-1, :2], "rs")
    ax[0, 1].plot(t, rp[:, 2]); ax[0, 1].set_title("root height (m)")
    ax[0, 2].plot(t, np.linalg.norm(z["body_lin_vel_w"][:, bi], axis=-1), label="lin m/s"); ax[0, 2].plot(t, np.linalg.norm(z["body_ang_vel_w"][:, bi], axis=-1), label="ang rad/s"); ax[0, 2].legend(); ax[0, 2].set_title("root speed")
    ax[1, 0].plot(t, jp); ax[1, 0].set_title("joint pos (29)")
    ax[1, 1].plot(t, np.abs(jv).max(-1)); ax[1, 1].set_title("max |joint vel| (rad/s)")
    names = [str(n) for n in z["joint_names"]]
    top = np.argsort(-np.abs(jv).max(0))[:6]
    for i in top:
        ax[1, 2].plot(t, jv[:, i], label=names[i].replace("_joint", ""))
    ax[1, 2].legend(fontsize=6); ax[1, 2].set_title("fastest joints (rad/s)")
    for a in ax.flat:
        a.set_xlabel("t (s)") if a is not ax[0, 0] else None; a.grid(alpha=.3)
    fig.suptitle(f"{dance.name}: {T} frames @ {fps:.0f} Hz = {T / fps:.2f} s"); fig.tight_layout()
    png = dance.dir / "reports" / f"{dance.name}_preview.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110); plt.close(fig)
    rep = read_json(dance.metrics / "qa_kinematic.json", {})
    print(f"== {dance.name}  QA {rep.get('verdict')}  {T} frames, {T / fps:.2f}s, root travel {rep.get('clip', {}).get('root_xy_travel_m', 0):.2f} m")
    for w in rep.get("warnings", []):
        print("  WARN:", w)
    print("  preview figure:", png)
    out = {"png": str(png)}
    if video:
        out["video"] = _render_video(dance, z)
    return out


def _render_video(dance: Dance, z) -> str | None:
    """Best-effort MuJoCo EGL render of the reference clip (G1 MJCF of the converter). Returns the mp4 path or None."""
    try:
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.path.insert(0, str(SUPERMIMIC / "src"))
        import mujoco
        from supermimic.data.motion_convert import load_g1_model
        import imageio.v2 as imageio
        m = load_g1_model()
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 140
        bi = list(z["body_names"]).index("pelvis")
        out = dance.dir / "reports" / f"{dance.name}_preview.mp4"
        w = imageio.get_writer(str(out), fps=25)
        for k in range(0, len(z["joint_pos"]), 2):
            d.qpos[:3], d.qpos[3:7], d.qpos[7:] = z["body_pos_w"][k, bi], z["body_quat_w"][k, bi], z["joint_pos"][k]
            mujoco.mj_forward(m, d); cam.lookat[:] = d.qpos[:3]
            r.update_scene(d, cam); w.append_data(r.render())
        w.close()
        return str(out)
    except Exception as e:  # headless / missing imageio: not fatal
        print(f"  (video skipped: {type(e).__name__}: {e})")
        return None
