"""PartyMimic Pipeline V1.1 - synchronized quad-panel replay video: reference + up to 3 closed-loop MuJoCo policy rollouts (main venv, EGL offscreen).

Panels : top-left = reference (kinematic replay), the other three = real physics rollouts recorded by pmpipe.rollout (same sample index = same reference frame).
Overlay: frame number / time, per-foot contact state (sim: MuJoCo contact force as % body weight; reference: calibrated foot height tier), tilt,
         CoM ground projection (red) + centre of pressure (green) + load-bearing contact points (blue = left, orange = right), peak torque ratio
         and a SAT flag when any joint touches its effort limit, guard-active flag, global root-XY offset to the reference.
usage (library): render_quad(dance, panels, out_mp4, sheet_png=None)   panels = [(title, npz_path)...] (<= 3)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from . import quality as Q

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PW, PH = 640, 360
GREEN, RED, AMBER, BLUE, ORANGE = (90, 220, 110), (240, 80, 80), (250, 190, 60), (90, 160, 255), (255, 150, 60)


def _scene_model(robot="g1mj"):
    import mujoco
    from partymimic.robot import get_robot
    spec = mujoco.MjSpec.from_file(get_robot(robot).mjcf_path)
    g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [30, 30, 0.1]; g.rgba = [0.80, 0.82, 0.86, 1]; g.contype = 0; g.conaffinity = 0
    spec.visual.headlight.ambient = [0.55, 0.55, 0.55]; spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]; spec.visual.rgba.haze = [0.78, 0.84, 0.92, 1]; spec.visual.map.haze = 0.0
    spec.worldbody.add_light(pos=[2, -2, 4], dir=[-0.4, 0.4, -1]); spec.visual.global_.offwidth = PW; spec.visual.global_.offheight = PH
    return spec.compile()


def _marker(scn, kind, pos, size, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array(size, float), np.array(pos, float), np.eye(3).flatten(), np.array(rgba, np.float32)); scn.ngeom += 1


def _panel_data(dance, panels, robot):
    ref = Q.load_ref(dance.padded_npz); K = Q.fk_series(ref["root_p"], ref["root_q"], ref["q"], robot)
    cal = Q.calibrate_floor(K["clr"], ref["jv"], ref["root_p"], ref["fps"]); tier, h = Q.contact_tiers(K["clr"], cal["offset"])
    com_ref = K["com"]; out = []
    for title, p in panels:
        ro = Q.load_rollout(p); out.append((title, ro))
    return ref, h, tier, com_ref, cal, out


def render_quad(dance, panels, out_mp4, sheet_png=None, robot="g1mj", seconds=None, nframes_sheet=8):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    panels = list(panels)[:3]
    ref, h, tier, com_ref, cal, sims = _panel_data(dance, panels, robot)
    T = min(len(s[1]["t"]) for s in sims) if sims else len(ref["q"]) - 1
    if seconds: T = min(T, int(seconds * 50))
    m = _scene_model(robot); datas = [mujoco.MjData(m) for _ in range(1 + len(sims))]; rend = mujoco.Renderer(m, PH, PW)
    cam = mujoco.MjvCamera(); cam.distance = 2.9; cam.elevation = -14; cam.azimuth = 135
    f_t = ImageFont.truetype(FONT, 17); f_s = ImageFont.truetype(FONT, 13); f_b = ImageFont.truetype(FONT, 22)
    BW = [s[1]["meta"]["mass_kg"] * 9.81 for s in sims]
    lim = [np.array(s[1]["meta"]["effort_lim"]) for s in sims]
    exy = [np.linalg.norm(s[1]["root_p"][:T, :2] - ref["root_p"][:T, :2], axis=1) for s in sims]
    W, H = 2 * PW, 2 * PH
    ref_dr = Q.dead_reckon(ref["root_p"][:, :2], Q.fk_series(ref["root_p"], ref["root_q"], ref["q"], robot)["ank"][:, :, :2], h < 0.025)
    allxy = np.concatenate([ref["root_p"][:T, :2]] + [s_[1]["root_p"][:T, :2] for s_ in sims] + [ref_dr[:T]]); lo, hi = allxy.min(0) - 0.15, allxy.max(0) + 0.15
    MW, MH = 150, 110; sc = min(MW / (hi[0] - lo[0]), MH / (hi[1] - lo[1]))
    def mp(xy, ox, oy): return (ox + (xy[0] - lo[0]) * sc, oy + MH - (xy[1] - lo[1]) * sc)
    def minimap(dr, ox, oy, k, sim_xy, col):
        dr.rectangle([ox - 2, oy - 2, ox + MW + 2, oy + MH + 2], fill=(245, 245, 245), outline=(60, 60, 60))
        dr.line([mp(p_, ox, oy) for p_ in ref["root_p"][:T:2, :2]], fill=(60, 60, 60), width=1)
        dr.line([mp(p_, ox, oy) for p_ in ref_dr[:T:2]], fill=(200, 60, 200), width=1)
        if sim_xy is not None: dr.line([mp(p_, ox, oy) for p_ in sim_xy[:k + 1:2]], fill=col, width=2); x_, y_ = mp(sim_xy[k], ox, oy); dr.ellipse([x_ - 4, y_ - 4, x_ + 4, y_ + 4], fill=col)
        x_, y_ = mp(ref["root_p"][k, :2], ox, oy); dr.ellipse([x_ - 3, y_ - 3, x_ + 3, y_ + 3], outline=(0, 0, 0), width=2)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "50", "-i", "-",
                           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "21", "-preset", "medium", str(out_mp4)], stdin=subprocess.PIPE)
    sheet_at = set(np.linspace(PAD_ := 0, T - 1, nframes_sheet).astype(int).tolist()); sheet = []
    for k in range(T):
        canvas = Image.new("RGB", (W, H))
        for i in range(1 + len(sims)):
            d = datas[i]
            if i == 0:
                title, rp, rq, q = f"REFERENCE ({getattr(dance, 'name', '')}, kinematic)", ref["root_p"][k], ref["root_q"][k], ref["q"][k]
            else:
                title, ro = sims[i - 1]; rp, rq, q = ro["root_p"][k], ro["root_q"][k], ro["q"][k]
            d.qpos[:3] = rp; d.qpos[3:7] = rq; d.qpos[7:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            com = d.subtree_com[1].copy(); cam.lookat[:] = [rp[0], rp[1], 0.38]
            rend.update_scene(d, cam); scn = rend.scene
            _marker(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [com[0], com[1], 0.004], [0.035, 0.002, 0], [0.95, 0.15, 0.15, 1.0])  # CoM ground projection
            if i > 0:
                ro = sims[i - 1][1]
                if np.isfinite(ro["cop"][k, 0]): _marker(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [ro["cop"][k, 0], ro["cop"][k, 1], 0.006], [0.03, 0.002, 0], [0.1, 0.85, 0.2, 1.0])  # CoP
                for x, y, z, fz, ft in ro["pts"][k]:
                    if np.isfinite(x) and fz > 1.0 and ft < 2:
                        _marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [x, y, 0.008], [0.009, 0, 0], [0.35, 0.6, 1, 0.9] if ft == 0 else [1, 0.6, 0.2, 0.9])
            img = Image.fromarray(rend.render()); dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, PW, 24], fill=(18, 18, 22)); dr.text((6, 3), title, font=f_t, fill=(255, 255, 255))
            tilt = np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1, 1)))
            y0 = PH - 78; dr.rectangle([0, y0, PW, PH], fill=(18, 18, 22))
            if i == 0:
                cl = [h[k, f] * 1000 for f in (0, 1)]; st = ["CONTACT" if tier[k, f] == 2 else "SWING" if tier[k, f] == 0 else "ambig." for f in (0, 1)]
                col = [GREEN if tier[k, f] == 2 else RED if tier[k, f] == 0 else AMBER for f in (0, 1)]
                dr.text((6, y0 + 3), f"L foot {cl[0]:5.1f} mm  {st[0]}", font=f_s, fill=col[0]); dr.text((250, y0 + 3), f"R foot {cl[1]:5.1f} mm  {st[1]}", font=f_s, fill=col[1])
                dr.text((6, y0 + 22), f"tilt {tilt:4.1f} deg   (height above calibrated floor; floor offset {cal['offset'][0]*1000:.0f}/{cal['offset'][1]*1000:.0f} mm)", font=f_s, fill=(220, 220, 220))
                dr.text((6, y0 + 41), "red dot = CoM ground projection", font=f_s, fill=(200, 200, 200))
            else:
                fz = ro["foot_fz"][k] / BW[i - 1] * 100; ld = [fz[f] > 10 for f in (0, 1)]
                for f, x0 in ((0, 6), (1, 250)):
                    c = GREEN if ld[f] else RED
                    dr.text((x0, y0 + 3), f"{'LR'[f]} foot {fz[f]:5.0f}% BW  {'LOADED' if ld[f] else 'unloaded'}", font=f_s, fill=c)
                    dr.rectangle([x0, y0 + 21, x0 + min(fz[f], 150) * 1.4, y0 + 27], fill=BLUE if f == 0 else ORANGE)
                tr = ro["tau_ratio"][k] ; sat = bool((tr >= 0.98).any()); jn = int(np.argmax(tr))
                dev = bool((np.abs(ro["qcmd"][k] - ro["qpre"][k]) > 1e-6).any())
                dr.text((6, y0 + 33), f"tilt {tilt:4.1f} deg   max torque {tr.max()*100:3.0f}% ({ro['meta']['joint_names'][jn].replace('_joint','')})", font=f_s, fill=RED if sat else (220, 220, 220))
                dr.text((6, y0 + 52), f"TORQUE SAT" if sat else "torque ok", font=f_s, fill=RED if sat else GREEN); dr.text((110, y0 + 52), "guard CLIP" if dev else "guard idle", font=f_s, fill=AMBER if dev else (160, 160, 160))
                dr.text((250, y0 + 52), f"root XY offset to ref {exy[i - 1][k]:.2f} m", font=f_s, fill=(220, 220, 220))
                if ro["meta"]["fell_t"] is not None and k * 0.02 >= ro["meta"]["fell_t"]: dr.text((PW - 90, 28), "FALLEN", font=f_b, fill=RED)
            dr.text((PW - 150, 3), f"frame {k:4d}  t={k*0.02:5.2f}s", font=f_s, fill=(255, 255, 0))
            minimap(dr, PW - MW - 8, 32, k, None if i == 0 else sims[i - 1][1]["root_p"][:T, :2], [(0, 0, 0), (30, 110, 230), (20, 150, 60), (200, 90, 0)][i])
            fs9 = ImageFont.truetype(FONT, 10)
            dr.text((PW - MW - 8, 32 + MH + 4), "top view  black: reference root", font=fs9, fill=(255, 255, 255)); dr.text((PW - MW - 8, 32 + MH + 17), "magenta: root implied by ref's own feet", font=fs9, fill=(255, 120, 255))
            canvas.paste(img, ((i % 2) * PW, (i // 2) * PH))
        ff.stdin.write(np.asarray(canvas).tobytes())
        if k in sheet_at: sheet.append(canvas.resize((W // 2, H // 2)))
    ff.stdin.close(); ff.wait()
    if sheet_png and sheet:
        cols = 2; rows = (len(sheet) + cols - 1) // cols; sh = Image.new("RGB", (cols * W // 2, rows * H // 2))
        for j, im in enumerate(sheet): sh.paste(im, ((j % cols) * W // 2, (j // cols) * H // 2))
        sh.save(sheet_png)
    return dict(video=str(out_mp4), frames=T, sheet=str(sheet_png) if sheet_png else None)


def render_repair_compare(dance, rdance, sim_orig, sim_rep, out_mp4, sheet_png=None, robot="g1mj", nframes_sheet=8):
    """V1.2: 4 panels = [original reference | repaired reference | frozen policy on original | frozen policy on repaired].
    Reference panels: foot height above z=0 (floating), contact-patch markers (blue L / orange R, larger = planted) and the horizontal speed of the planted patch (red = skating);
    sim panels: MuJoCo force contact (% body weight), CoM/CoP markers, root offset to the SAME panel's reference."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    from . import repair as RP
    refs = []
    for dn in (dance, rdance):
        R = Q.load_ref(dn.padded_npz); K = Q.fk_series(R["root_p"], R["root_q"], R["q"], robot, want_pts=True); patch, clr = RP.contact_patch(K)
        cal = Q.calibrate_floor(K["clr"], R["jv"], R["root_p"], R["fps"]); hcal = K["clr"] - cal["offset"]; v = np.linalg.norm(np.gradient(patch, 1 / R["fps"], axis=0), axis=-1)
        imp = Q.dead_reckon(R["root_p"][:, :2], K["ank"][:, :, :2], hcal < 0.025); refs.append(dict(R=R, K=K, patch=patch, clr=clr, hcal=hcal, v=v, imp=imp))
    sims = [Q.load_rollout(sim_orig), Q.load_rollout(sim_rep)]; T = min(len(sims[0]["t"]), len(sims[1]["t"]))
    m = _scene_model(robot); datas = [mujoco.MjData(m) for _ in range(4)]; rend = mujoco.Renderer(m, PH, PW); cam = mujoco.MjvCamera(); cam.distance = 2.9; cam.elevation = -14; cam.azimuth = 135
    f_t = ImageFont.truetype(FONT, 17); f_s = ImageFont.truetype(FONT, 13); f_9 = ImageFont.truetype(FONT, 10)
    W, H = 2 * PW, 2 * PH
    allxy = np.concatenate([r["R"]["root_p"][:T, :2] for r in refs] + [s["root_p"][:T, :2] for s in sims]); lo, hi = allxy.min(0) - 0.15, allxy.max(0) + 0.15; MW, MH = 150, 110; sc = min(MW / (hi[0] - lo[0]), MH / (hi[1] - lo[1]))
    mp = lambda xy, ox, oy: (ox + (xy[0] - lo[0]) * sc, oy + MH - (xy[1] - lo[1]) * sc)
    def minimap(dr, ox, oy, k, line_xy, col, ref_xy, imp_xy):
        dr.rectangle([ox - 2, oy - 2, ox + MW + 2, oy + MH + 2], fill=(245, 245, 245), outline=(60, 60, 60))
        dr.line([mp(p_, ox, oy) for p_ in ref_xy[:T:2]], fill=(120, 120, 120), width=1); dr.line([mp(p_, ox, oy) for p_ in imp_xy[:T:2]], fill=(200, 60, 200), width=1)
        dr.line([mp(p_, ox, oy) for p_ in line_xy[:k + 1:2]], fill=col, width=2); x_, y_ = mp(line_xy[k], ox, oy); dr.ellipse([x_ - 4, y_ - 4, x_ + 4, y_ + 4], fill=col)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "50", "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "21", str(out_mp4)], stdin=subprocess.PIPE)
    titles = ["ORIGINAL reference", "REPAIRED reference", "frozen policy on ORIGINAL", "frozen policy on REPAIRED"]; cols = [(0, 0, 0), (20, 140, 60), (200, 90, 0), (30, 110, 230)]
    sheet_at = set(np.linspace(0, T - 1, nframes_sheet).astype(int).tolist()); sheet = []
    for k in range(T):
        canvas = Image.new("RGB", (W, H))
        for i in range(4):
            d = datas[i]; isref = i < 2; rf = refs[i if isref else i - 2]
            if isref: rp, rq, q = rf["R"]["root_p"][k], rf["R"]["root_q"][k], rf["R"]["q"][k]
            else: ro = sims[i - 2]; rp, rq, q = ro["root_p"][k], ro["root_q"][k], ro["q"][k]
            d.qpos[:3] = rp; d.qpos[3:7] = rq; d.qpos[7:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); com = d.subtree_com[1].copy(); cam.lookat[:] = [rp[0], rp[1], 0.38]
            rend.update_scene(d, cam); scn = rend.scene
            _marker(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [com[0], com[1], 0.004], [0.035, 0.002, 0], [0.95, 0.15, 0.15, 1.0])
            if isref:
                for f in (0, 1):
                    planted = rf["hcal"][k, f] < 0.010; p = rf["patch"][k, f]
                    _marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [p[0], p[1], 0.012], [0.02 if planted else 0.01, 0, 0], [0.35, 0.6, 1, 0.95] if f == 0 else [1, 0.6, 0.2, 0.95])
            else:
                if np.isfinite(ro["cop"][k, 0]): _marker(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [ro["cop"][k, 0], ro["cop"][k, 1], 0.006], [0.03, 0.002, 0], [0.1, 0.85, 0.2, 1.0])
                for x, y, z, fz, ft in ro["pts"][k]:
                    if np.isfinite(x) and fz > 1.0 and ft < 2: _marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [x, y, 0.008], [0.009, 0, 0], [0.35, 0.6, 1, 0.9] if ft == 0 else [1, 0.6, 0.2, 0.9])
            img = Image.fromarray(rend.render()); dr = ImageDraw.Draw(img); dr.rectangle([0, 0, PW, 24], fill=(18, 18, 22)); dr.text((6, 3), titles[i], font=f_t, fill=(255, 255, 255)); dr.text((PW - 150, 3), f"frame {k:4d}  t={k*0.02:5.2f}s", font=f_s, fill=(255, 255, 0))
            y0 = PH - 78; dr.rectangle([0, y0, PW, PH], fill=(18, 18, 22))
            if isref:
                for f, x0 in ((0, 6), (1, 250)):
                    pl = rf["hcal"][k, f] < 0.010; v = rf["v"][k, f]; skating = pl and v > 0.3
                    dr.text((x0, y0 + 3), f"{'LR'[f]} foot {rf['clr'][k, f]*1000:5.1f} mm above z=0", font=f_s, fill=(255, 170, 80) if rf["clr"][k, f] > 0.012 else (120, 230, 130))
                    dr.text((x0, y0 + 22), f"{'planted' if pl else 'swing'} v={v:4.2f} m/s{' SKATE' if skating else ''}", font=f_s, fill=(240, 80, 80) if skating else (200, 200, 200))
                dr.text((6, y0 + 41), f"root XY path so far {Q.path_len(rf['R']['root_p'][:k + 1, :2]):4.2f} m (planted feet imply {Q.path_len(rf['imp'][:k + 1]):4.2f} m)", font=f_s, fill=(220, 220, 220))
                dr.text((6, y0 + 58), "blue/orange balls = foot contact patch (big = planted)", font=f_9, fill=(170, 170, 170))
                line, ref_xy, imp_xy = rf["R"]["root_p"][:T, :2], rf["R"]["root_p"][:T, :2], rf["imp"][:T]
            else:
                BW = ro["meta"]["mass_kg"] * 9.81; fz = ro["foot_fz"][k] / BW * 100
                for f, x0 in ((0, 6), (1, 250)):
                    ld = fz[f] > 10; dr.text((x0, y0 + 3), f"{'LR'[f]} foot {fz[f]:5.0f}% BW  {'LOADED' if ld else 'unloaded'}", font=f_s, fill=(90, 220, 110) if ld else (240, 80, 80)); dr.rectangle([x0, y0 + 21, x0 + min(fz[f], 150) * 1.4, y0 + 27], fill=BLUE if f == 0 else ORANGE)
                tr = ro["tau_ratio"][k]; exy = np.linalg.norm(ro["root_p"][k, :2] - rf["R"]["root_p"][k, :2])
                dr.text((6, y0 + 33), f"max torque {tr.max()*100:3.0f}%   {'TORQUE SAT' if (tr >= 0.98).any() else 'torque ok'}", font=f_s, fill=(240, 80, 80) if (tr >= 0.98).any() else (90, 220, 110))
                dr.text((6, y0 + 52), f"root XY offset to its OWN reference {exy:.2f} m", font=f_s, fill=(220, 220, 220))
                line, ref_xy, imp_xy = ro["root_p"][:T, :2], rf["R"]["root_p"][:T, :2], rf["imp"][:T]
            minimap(dr, PW - MW - 8, 32, k, line, cols[i], ref_xy, imp_xy)
            dr.text((PW - MW - 8, 32 + MH + 4), "top view  grey: its reference root", font=f_9, fill=(255, 255, 255)); dr.text((PW - MW - 8, 32 + MH + 17), "magenta: root implied by planted feet", font=f_9, fill=(255, 120, 255))
            canvas.paste(img, ((i % 2) * PW, (i // 2) * PH))
        ff.stdin.write(np.asarray(canvas).tobytes())
        if k in sheet_at: sheet.append(canvas.resize((W // 2, H // 2)))
    ff.stdin.close(); ff.wait()
    if sheet_png and sheet:
        rows = (len(sheet) + 1) // 2; sh = Image.new("RGB", (W, rows * H // 2))
        for j, im in enumerate(sheet): sh.paste(im, ((j % 2) * W // 2, (j // 2) * H // 2))
        sh.save(sheet_png)
    return dict(video=str(out_mp4), sheet=str(sheet_png) if sheet_png else None, frames=T)
