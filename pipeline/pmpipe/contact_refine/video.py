"""contact_refine.video - ORIGINAL vs REFINED reference comparison video (MuJoCo kinematic replay, EGL offscreen; same scene helpers as pmpipe.replay).

Layout 1280x920:  row 1 = oblique tracking view  [ORIGINAL | REFINED];  row 2 = side close-up of the feet [ORIGINAL | REFINED];  row 3 = metric strip chart with a moving cursor.
Overlays: frame numbers (dance frame / training frame incl. the 50-frame lead-in), per foot: lowest-sole height over the real floor z=0 (mm) and contact state, support-point
spheres (green = stance, red = swing, amber = uncertain; large = pinned), a drop line from the lowest sole point to the floor (green <5 mm, amber 5-20, red >20 mm hover; magenta = penetration).
"""
from __future__ import annotations

import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from .replay_compat import FONT, PH, PW, marker, scene_model
from . import geometry as Gm

GREEN, RED, AMBER = (90, 220, 110), (240, 80, 80), (250, 190, 60)


def _chart(S, lo, hi, fps, W, H, title_extra=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); L, R_, B, Tp = 60, 20, 34, 12; axw, axh = W - L - R_, (H - B - Tp - 24) / 2
    t = np.arange(lo, hi)
    a1 = fig.add_axes([L / W, (B + axh + 24) / H, axw / W, axh / H]); a2 = fig.add_axes([L / W, B / H, axw / W, axh / H])
    a1.plot(t, S["low0"][lo:hi].min(1) * 1000, c="0.35", lw=1.2, label="original: lowest sole pt over z=0"); a1.plot(t, S["e1"][lo:hi].min(1) * 1000, c="tab:green", lw=1.2, label="refined: lowest sole pt over z=0")
    a1.axhline(0, c="k", lw=0.5); a1.set_ylabel("mm", fontsize=8); a1.legend(fontsize=7, loc="upper right", ncol=2); a1.tick_params(labelsize=7); a1.set_xlim(lo, hi); a1.set_xticklabels([]); a1.set_ylim(min(-5, S["e1"][lo:hi].min() * 1000 - 2), min(120, max(30, S["low0"][lo:hi].min(1).max() * 1000 + 5)))
    for key, c, lab in (("0", "0.35", "original"), ("1", "tab:green", "refined")):
        s = np.where(S["o" + key], S["slip" + key], 0).max(axis=(1, 2)); a2.plot(t, np.minimum(s[lo - 1:hi - 1], 1.2), c=c, lw=0.9, label=f"{lab}: contact-point slip (stance)")
    a2.set_ylim(0, 1.2); a2.set_ylabel("m/s", fontsize=8); a2.legend(fontsize=7, loc="upper right"); a2.tick_params(labelsize=7); a2.set_xlim(lo, hi); a2.set_xlabel("dance frame", fontsize=8, labelpad=1)
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return img, (L, W - R_)


def render(S, root_q, q0, q1, out_mp4, lo, hi, step, play_fps, title, lead=50, tilt=None, label_note=""):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    m = scene_model(); d = [mujoco.MjData(m) for _ in range(2)]; rend = mujoco.Renderer(m, PH, PW)
    W, H = 2 * PW, 2 * PH + 200
    chart, (cx0, cx1) = _chart(S, lo, hi, 50, W, 200); f_t = ImageFont.truetype(FONT, 17); f_s = ImageFont.truetype(FONT, 13); f_b = ImageFont.truetype(FONT, 20)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(play_fps), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(out_mp4)], stdin=subprocess.PIPE)
    roots = [S["root0"], S["root1"]]; qs = [q0, q1]; solesx = [S["sole0"], S["sole1"]]; states = [S["state0"], S["state1"]]; pins = [S["pin"], S["pin"]]
    heights = [S["low0"] * 0 + 0, None]
    tit = ["ORIGINAL (as delivered)", "REFINED (contact_refine)"]
    for k in range(lo, hi, step):
        canvas = Image.new("RGB", (W, H))
        for i in range(2):
            rp = roots[i][k]; dd = d[i]; dd.qpos[:3] = rp; dd.qpos[3:7] = root_q[k]; dd.qpos[7:] = qs[i][k]; mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
            sole = solesx[i][k]; foot_mean = sole.reshape(-1, 3).mean(0); yaw = float(Gm.yaw_of(root_q[k:k + 1])[0])
            for row in range(2):
                cam = mujoco.MjvCamera()
                if row == 0: cam.distance = 2.1; cam.elevation = -12; cam.azimuth = 135; cam.lookat[:] = [rp[0], rp[1], 0.42]
                else: cam.distance = 0.95; cam.elevation = -3; cam.azimuth = np.degrees(yaw) + 90; cam.lookat[:] = [foot_mean[0], foot_mean[1], 0.08]
                rend.update_scene(dd, cam); scn = rend.scene
                for f in (0, 1):
                    stt = states[i][k, f]; col = {1: [0.2, 0.9, 0.3, 1], 0: [0.95, 0.25, 0.25, 1], 2: [0.98, 0.75, 0.2, 1]}[int(stt)]
                    for n in range(14):
                        p = sole[f, n]; marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [p[0], p[1], p[2] - 0.01], [0.007, 0, 0], col)
                    j = int(np.argmin(sole[f, :, 2])); p = sole[f, j]; h = p[2] - 0.01
                    c2 = [0.9, 0.2, 0.9, 1] if h < -0.001 else ([0.2, 0.9, 0.3, 1] if h < 0.005 else ([0.98, 0.75, 0.2, 1] if h < 0.02 else [0.95, 0.2, 0.2, 1]))
                    marker(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [p[0], p[1], 0], [0.003, 0.003, 0.003], c2, connector=([p[0], p[1], h], [p[0], p[1], 0.0]))
                img = Image.fromarray(rend.render()); dr = ImageDraw.Draw(img)
                if row == 0:
                    dr.rectangle([0, 0, PW, 24], fill=(18, 18, 22)); dr.text((6, 3), tit[i], font=f_t, fill=(255, 255, 255))
                    dr.text((PW - 330, 5), f"dance fr {k:4d} | train fr {k + lead:4d} | {k*0.02:6.2f}s", font=f_s, fill=(255, 255, 0))
                else:
                    dr.rectangle([0, PH - 60, PW, PH], fill=(18, 18, 22))
                    for f, x0 in ((0, 6), (1, 215)):
                        hh = (sole[f, :, 2].min() - 0.01) * 1000; stt = int(states[i][k, f]); nm = {1: "STANCE", 0: "swing", 2: "uncertain"}[stt]
                        dr.text((x0, PH - 57), f"{'LR'[f]} low pt {hh:6.1f} mm", font=f_s, fill=(RED if (stt == 1 and hh > 10) else (255, 255, 255) if hh >= -0.5 else (240, 120, 240)))
                        dr.text((x0, PH - 38), f"{nm}", font=f_s, fill={1: GREEN, 0: RED, 2: AMBER}[stt])
                    dr.text((6, PH - 19), "side view of the feet; drop line = height of the lowest sole point (red>20mm hover, magenta=penetration)", font=ImageFont.truetype(FONT, 10), fill=(180, 180, 180))
                    if tilt is not None: dr.text((PW - 140, PH - 57), f"trunk tilt {tilt[k]:.2f} rad", font=f_s, fill=(255, 255, 0) if tilt[k] > 0.35 else (200, 200, 200))
                canvas.paste(img, (i * PW, row * PH))
        ch = Image.fromarray(chart); cd = ImageDraw.Draw(ch); x = cx0 + (k - lo) / max(hi - lo, 1) * (cx1 - cx0); cd.line([x, 4, x, 196], fill=(220, 30, 30), width=2)
        canvas.paste(ch, (0, 2 * PH)); dd_ = ImageDraw.Draw(canvas); dd_.text((8, 2 * PH + 2), title + label_note, font=ImageFont.truetype(FONT, 11), fill=(0, 0, 0))
        ff.stdin.write(np.asarray(canvas).tobytes())
    ff.stdin.close(); ff.wait(); return out_mp4
