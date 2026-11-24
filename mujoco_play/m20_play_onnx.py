#!/usr/bin/env python3
"""MuJoCo viewer for DeepRobotics M20 RSL-RL ONNX (57→16).

Obs (policy, no noise): ang_vel*0.25 | gravity | cmd | q_rel(wheels=0) | qd*0.05 | last_a
Actions: 12 leg position (+default, scale hipx=0.125 else 0.25) + 4 wheel velocity (*5)
PD: legs Kp=80 Kd=2; wheels Kp=0 Kd=0.6

Keys: 8/2 vx± · 4/6 vy± · 7/9 yaw± · 5/0/Space stop · R reset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# Policy / Isaac joint order: 12 legs then 4 wheels
LEG_JOINTS = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
WHEEL_JOINTS = [
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]
POLICY_JOINTS = LEG_JOINTS + WHEEL_JOINTS

# Match DEEPROBOTICS_M20_CFG init_state
DEFAULT_POS = np.array(
    [
        0.0, -0.3, 0.6,
        0.0, -0.3, 0.6,
        0.0, 0.3, -0.6,
        0.0, 0.3, -0.6,
        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4 + [5.0] * 4, dtype=np.float64)
KP = np.array([80.0] * 12 + [0.0] * 4, dtype=np.float64)
KD = np.array([2.0] * 12 + [0.6] * 4, dtype=np.float64)
EFFORT_LIMIT = np.array([76.4] * 12 + [21.6] * 4, dtype=np.float64)

ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
OBS_DIM = 57
ACT_DIM = 16
BASE_Z = 0.58

_K = {
    "2": 50, "4": 52, "5": 53, "6": 54, "7": 55, "8": 56, "9": 57, "0": 48,
    "SPACE": 32, "R": 82,
    "UP": 265, "DOWN": 264, "LEFT": 263, "RIGHT": 262,
    "KP_2": 322, "KP_4": 324, "KP_5": 325, "KP_6": 326,
    "KP_7": 327, "KP_8": 328, "KP_9": 329, "KP_0": 320,
}


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    q_vec = np.array([x, y, z], dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return a - b + c


class DigitStepTeleop:
    def __init__(
        self,
        vx_max: float = 1.0,
        vy_max: float = 0.5,
        yaw_max: float = 1.0,
        vx_step: float = 0.2,
        vy_step: float = 0.15,
        yaw_step: float = 0.2,
    ):
        self.vx_max, self.vy_max, self.yaw_max = vx_max, vy_max, yaw_max
        self.cmd = np.zeros(3, dtype=np.float64)
        self.reset_requested = False
        self._last_step_t = 0.0
        self._min_step_dt = 0.08
        self._steps: dict[int, np.ndarray] = {}
        for keys, delta in (
            ((_K["8"], _K["KP_8"], _K["UP"]), np.array([vx_step, 0.0, 0.0])),
            ((_K["2"], _K["KP_2"], _K["DOWN"]), np.array([-vx_step, 0.0, 0.0])),
            ((_K["4"], _K["KP_4"], _K["LEFT"]), np.array([0.0, vy_step, 0.0])),
            ((_K["6"], _K["KP_6"], _K["RIGHT"]), np.array([0.0, -vy_step, 0.0])),
            ((_K["7"], _K["KP_7"]), np.array([0.0, 0.0, yaw_step])),
            ((_K["9"], _K["KP_9"]), np.array([0.0, 0.0, -yaw_step])),
        ):
            for k in keys:
                self._steps[k] = delta
        self._stop = {_K["5"], _K["KP_5"], _K["0"], _K["KP_0"], _K["SPACE"]}
        self._reset = {_K["R"]}

    def _clip(self) -> None:
        self.cmd[0] = float(np.clip(self.cmd[0], -self.vx_max, self.vx_max))
        self.cmd[1] = float(np.clip(self.cmd[1], -self.vy_max, self.vy_max))
        self.cmd[2] = float(np.clip(self.cmd[2], -self.yaw_max, self.yaw_max))

    def on_key(self, keycode: int) -> None:
        now = time.time()
        if keycode in self._stop:
            self.cmd[:] = 0.0
            print("[m20] cmd=0")
            return
        if keycode in self._reset:
            self.reset_requested = True
            self.cmd[:] = 0.0
            print("[m20] reset")
            return
        delta = self._steps.get(keycode)
        if delta is None or (now - self._last_step_t) < self._min_step_dt:
            return
        self._last_step_t = now
        self.cmd = self.cmd + delta
        self._clip()
        print(f"[m20] cmd=({self.cmd[0]:+.2f}, {self.cmd[1]:+.2f}, {self.cmd[2]:+.2f})")


class M20Player:
    def __init__(self, xml_path: Path, onnx_path: Path, cmd0: np.ndarray):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self._in_name = inp.name
        self._out_name = out.name

        # map policy joint index → (qposadr, dofadr, actuator_id)
        self.qpos_adr = np.zeros(16, dtype=np.int32)
        self.dof_adr = np.zeros(16, dtype=np.int32)
        self.act_id = np.zeros(16, dtype=np.int32)
        for i, name in enumerate(POLICY_JOINTS):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid < 0 or aid < 0:
                raise RuntimeError(f"missing joint/actuator: {name}")
            self.qpos_adr[i] = self.model.jnt_qposadr[jid]
            self.dof_adr[i] = self.model.jnt_dofadr[jid]
            self.act_id[i] = aid

        self.last_action = np.zeros(ACT_DIM, dtype=np.float64)
        self.teleop = DigitStepTeleop()
        self.teleop.cmd[:] = cmd0
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = max(1, int(round(0.02 / self.sim_dt)))  # 50 Hz policy
        self._reset()

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, BASE_Z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # wxyz
        for i in range(16):
            self.data.qpos[self.qpos_adr[i]] = DEFAULT_POS[i]
            self.data.qvel[self.dof_adr[i]] = 0.0
        self.data.qvel[0:6] = 0.0
        self.last_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _joint_state_policy(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.data.qpos[a] for a in self.qpos_adr], dtype=np.float64)
        qd = np.array([self.data.qvel[a] for a in self.dof_adr], dtype=np.float64)
        return q, qd

    def _build_obs(self, cmd: np.ndarray) -> np.ndarray:
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()  # body frame
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        q, qd = self._joint_state_policy()
        q_rel = q - DEFAULT_POS
        q_rel[12:] = 0.0  # wheels zeroed in obs
        obs = np.concatenate(
            [
                ang_vel * ANG_VEL_SCALE,
                grav,
                cmd.astype(np.float64),
                q_rel,
                qd * JOINT_VEL_SCALE,
                self.last_action,
            ]
        )
        assert obs.shape == (OBS_DIM,), obs.shape
        return obs.astype(np.float32)

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run([self._out_name], {self._in_name: obs[None, :]})[0]
        return np.asarray(out[0], dtype=np.float64)

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -100.0, 100.0)
        self.last_action = action.copy()
        q, qd = self._joint_state_policy()
        # legs: position target; wheels: velocity target
        q_des = DEFAULT_POS.copy()
        q_des[:12] = DEFAULT_POS[:12] + ACTION_SCALE[:12] * action[:12]
        v_des = np.zeros(16, dtype=np.float64)
        v_des[12:] = ACTION_SCALE[12:] * action[12:]
        tau = KP * (q_des - q) + KD * (v_des - qd)
        tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        for i in range(16):
            self.data.ctrl[self.act_id[i]] = tau[i]

    def step_control(self) -> None:
        if self.teleop.reset_requested:
            self.teleop.reset_requested = False
            self._reset()
        obs = self._build_obs(self.teleop.cmd)
        action = self._policy(obs)
        self._apply_action(action)

    def run(self) -> None:
        print(f"[m20] sim_dt={self.sim_dt} decimation={self.decimation} policy≈{1/(self.sim_dt*self.decimation):.0f}Hz")
        print("[m20] keys: 8/2 vx±  4/6 vy±  7/9 yaw±  5/0 stop  R reset")
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=lambda k: self.teleop.on_key(k)
        ) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20
            step_i = 0
            while viewer.is_running():
                t0 = time.time()
                if step_i % self.decimation == 0:
                    self.step_control()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                step_i += 1
                elapsed = time.time() - t0
                sleep = self.sim_dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_xml = (
        root
        / "upstream_rl_training"
        / "deep_robotics_model"
        / "M20"
        / "mjcf"
        / "M20.xml"
    )
    p = argparse.ArgumentParser(description="M20 MuJoCo ONNX play")
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--xml", type=Path, default=default_xml)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    args = p.parse_args()
    cmd0 = np.array([args.vx, args.vy, args.yaw], dtype=np.float64)
    print(f"[m20] onnx={args.onnx}")
    print(f"[m20] xml={args.xml}")
    print(f"[m20] init cmd=({args.vx}, {args.vy}, {args.yaw})")
    M20Player(args.xml, args.onnx, cmd0).run()


if __name__ == "__main__":
    main()
