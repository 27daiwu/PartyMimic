"""contact_refine.config - algorithm parameters and the PRE-REGISTERED data-QA gate (frozen BEFORE any refinement is run; never edited to chase a verdict).

QA_GATES is the single source of truth used by qa.py, the unit tests, the report and the CLI.  Changing a number here changes GATES_VERSION; a verdict
always records the version and the sha256 of the gate table it was judged with.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

VERSION = "contact_refine v1.0"
GATES_VERSION = "qa-gates v1.0 (2026-11-27, frozen before the first DIY refinement run)"


@dataclass
class Params:
    # --- contact identification
    stance_h_mm: float = 12.0         # lowest-sole-point height (above the calibrated ground) at which a foot is "certainly touching"
    stance_tau_mm: float = 4.0        # logistic width of the height feature
    infer_max_h_mm: float = 16.0      # support inference only for a foot at most this high over the ground curve
    swing_h_mm: float = 30.0          # above this height a foot is a certain swing foot
    v_h_ref: float = 0.25             # horizontal contact-point speed at which the speed feature drops to e^-0.5 [m/s]
    v_z_ref: float = 0.20             # vertical speed likewise
    p_hi: float = 0.60                # Schmitt trigger thresholds on the stance probability
    p_lo: float = 0.35
    min_run: int = 3                  # minimum duration (frames) of a stance / swing run (hysteresis)
    pinz_band_mm: float = 4.0         # z pin only for samples within this band above the foot's lowest sample
    pin_h_mm: float = 10.0            # sole samples lower than this (above ground) enter the pinned set of a stance foot
    # --- flight (true airborne) validation
    flight_h_mm: float = 25.0         # both feet higher than this over the ground for a flight candidate
    flight_min_frames: int = 4
    flight_max_s: float = 1.2
    flight_az_range: tuple = (-13.5, -3.5)  # accepted fitted CoM vertical acceleration [m/s^2]: ballistic = -9.81, but retargeted (scaled) mocap jumps measure -4..-10; hover artefacts measure |az| < 3.3
    flight_fit_rms_mm: float = 12.0   # RMS of the parabola fit of the CoM height
    # --- ground
    als_p: float = 0.05               # asymmetric least squares: weight of points ABOVE the ground curve (the foot is allowed to be in the air)
    ground_fc_hz: float = 2.5         # equivalent cut-off of the ground curve (curvature limit)
    ground_max_rate: float = 0.20     # [m/s] reported / enforced drift-rate bound
    static_jv_rms: float = 0.20
    static_root_v: float = 0.05
    static_min_s: float = 0.30
    static_boost: float = 20.0
    # --- solver weights (relative; all residuals in metres)
    w_pin_xy: float = 150.0
    w_pin_z: float = 300.0
    pin_z_target_mm: float = 1.0      # lowest stance samples are driven to +1 mm (the hinge keeps everything >= 0.5 mm)
    w_d1: float = 6.0                 # root-correction velocity weight (units of 0.3 m/s)
    w_flat: float = 6.0               # weak flattening of stance soles that are only slightly tilted
    flat_h_mm: float = 25.0
    w_pen: float = 1000.0
    pen_tol_mm: float = 0.5
    w_d0: float = 0.02                # root deviation magnitude
    w_d2: float = 0.3                 # root correction smoothness: weight of the correction's acceleration in units of 10 m/s^2 (the DELTA, not the trajectory)
    w_d2_flight: float = 400.0        # inside a validated flight the correction must be (almost) a rigid shift: ballistic shape preserved
    w_u0_stance: float = 0.3          # joint-delta magnitude penalty (per rad, scaled by 1/sigma_j) in stance
    w_u0_swing: float = 40.0          # swing legs: change only as needed for transitions
    w_u2: float = 0.3                 # joint-correction acceleration (units of 50 rad/s^2)
    w_u1: float = 0.5                 # joint-correction velocity (units of 2 rad/s)
    huber_mm: float = 12.0            # robust loss threshold of the pin residuals (pivot turns / video noise are not forced out)
    lm_d: float = 3.0                 # LM damping of the root increments
    lm_u: float = 0.5                 # LM damping of the joint increments (per sigma)
    ik: bool = True
    gn_iters: int = 4
    leg_bound_deg: float = 6.0        # hip/knee delta bound
    ankle_bound_deg: float = 10.0     # ankle delta bound (sole flattening)
    waist_ik: bool = False            # waist/torso/arms/head are NEVER changed by default
    ik_joint_sigma_deg: float = 2.0
    root_z_bound_m: tuple = (-0.15, 0.06)

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# pre-registered data QA gate.  every entry: (id, human description, comparator, threshold, unit).  evaluated by qa.evaluate
QA_GATES = [
    dict(id="slip_p95_reduction", desc="确定支撑帧接触点脚滑 p95（V1.4 口径）较原始下降", op=">=", value=50.0, unit="%", key="slip.p95_reduction_pct"),
    dict(id="slip_mean_reduction", desc="确定支撑帧接触点脚滑均值较原始下降", op=">=", value=50.0, unit="%", key="slip.mean_reduction_pct"),
    dict(id="stance_height_p95", desc="支撑脚最低足底点离地 |高度| p95", op="<=", value=5.0, unit="mm", key="ground.stance_abs_height_p95_mm"),
    dict(id="penetration_max", desc="全帧最大足底穿透深度（两脚全部采样点）", op="<=", value=5.0, unit="mm", key="ground.penetration_max_mm"),
    dict(id="penetration_p95_frames", desc="穿透 >2 mm 的帧占比", op="<=", value=1.0, unit="%", key="ground.penetration_gt2mm_frac_pct"),
    dict(id="upper_body_delta", desc="上肢/腰/头相关关节改动最大值", op="<=", value=0.5, unit="deg", key="joints.upper_max_deg"),
    dict(id="leg_delta_max", desc="腿部关节改动最大值（踝以外）", op="<=", value=6.5, unit="deg", key="joints.leg_noankle_max_deg"),
    dict(id="ankle_delta_max", desc="踝关节改动最大值", op="<=", value=10.5, unit="deg", key="joints.ankle_max_deg"),
    dict(id="joint_delta_rms", desc="全部关节改动 RMS（含不变关节）", op="<=", value=2.0, unit="deg", key="joints.rms_all_deg"),
    dict(id="joint_vel_limit", desc="腿部关节速度：超硬件限速的帧占比不高于原始，且峰值不超过 max(硬件限速, 1.05×原始峰值)", op="==", value=True, unit="bool", key="dynamics.joint_vel_ok"),
    dict(id="joint_acc_p999", desc="关节加速度 p99.9（腿）不超过原始的 1.25 倍", op="<=", value=1.25, unit="x", key="dynamics.joint_acc_p999_ratio"),
    dict(id="root_acc_ratio", desc="根线加速度 p99.5 不超过原始的 1.25 倍", op="<=", value=1.25, unit="x", key="dynamics.root_acc_p995_ratio"),
    dict(id="root_xy_dev_max", desc="根 XY 与原始最大偏离（累积重建的落脚一致性）", op="<=", value=1.5, unit="m", key="root.dev_xy_max_m"),
    dict(id="root_xy_rate_p99", desc="根速度被改动量 |v_new-v_old| p99", op="<=", value=0.30, unit="m/s", key="root.dv_p99_mps"),
    dict(id="root_z_dev", desc="根高度偏离范围（下限 -0.15 / 上限 +0.06 m）", op="==", value=True, unit="bool", key="root.dz_in_bounds"),
    dict(id="style_rootrel_mpjpe", desc="风格偏离：骨盆相对 30 body 平均位置差", op="<=", value=10.0, unit="mm", key="style.pelvis_rel_mpjpe_mm"),
    dict(id="style_leg_mpjpe", desc="风格偏离：骨盆相对 腿部 body 平均位置差", op="<=", value=25.0, unit="mm", key="style.leg_pelvis_rel_mpjpe_mm"),
    dict(id="style_upper_mpjpe", desc="风格偏离：骨盆相对 上肢 body 平均位置差（应≈0）", op="<=", value=1.0, unit="mm", key="style.upper_pelvis_rel_mpjpe_mm"),
    dict(id="flight_preserved", desc="真实腾空段：数量/起止不变，腾空段根弹道形状改变与离地高度改变 ≤ 20 mm（无腾空则恒真）", op="==", value=True, unit="bool", key="flight.preserved"),
    dict(id="no_new_flight", desc="修正后不得新增'腾空'帧（相对原始真实腾空）", op="==", value=True, unit="bool", key="flight.no_new_flight"),
    dict(id="no_new_anomalies", desc="无新增异常帧：新增穿透帧 / 新增关节超限帧 / 新增接触毛刺数 均 ≤ 0", op="==", value=True, unit="bool", key="anomalies.none_new"),
    dict(id="zmp_margin_not_worse", desc="可靠帧 ZMP 支撑域余量中位数不低于原始 -10 mm", op=">=", value=-10.0, unit="mm", key="dynamics.zmp_margin_median_delta_mm"),
    dict(id="orientation_unchanged", desc="根朝向/帧数/fps/名称不变，且 npz 字段齐全有限", op="==", value=True, unit="bool", key="integrity.ok"),
]


def gates_digest() -> str:
    return hashlib.sha256(json.dumps(QA_GATES, sort_keys=True, default=str).encode()).hexdigest()
