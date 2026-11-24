#!/usr/bin/env python3
"""MuJoCo viewer for Wheeldog Lynx RSL-RL ONNX (65→16).

Obs: ang_vel*0.25 | gravity | cmd | q_rel(wheels=0) | qd*0.05 | gait_clock(8) | last_a
Actions: 12 leg pos (+q0, H1=0.125 else 0.25) + 4 wheel vel (*5)
MJCF uses position/velocity actuators (Kp/Kd baked in XML).

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

LEG_JOINTS = [
    "FL_H1_L_j1", "FL_H2_L_j2", "FL_K_L_j3",
    "FR_H1_L_j1", "FR_H2_L_j2", "FR_K_L_j3",
    "RL_H1_L_j1", "RL_H2_L_j2", "RL_K_L_j3",
    "RR_H1_L_j1", "RR_H2_L_j2", "RR_K_L_j3",
]
WHEEL_JOINTS = ["FL_W_L_j4", "FR_W_L_j4", "RL_W_L_j4", "RR_W_L_j4"]
POLICY_JOINTS = LEG_JOINTS + WHEEL_JOINTS
ACTUATOR_NAMES = [f"act_{n}" for n in POLICY_JOINTS]

# Pinocchio stand q0 (policy order: legs then wheels)
DEFAULT_POS = np.array(
    [
        0.02, 0.721326, 1.442678,
        -0.02, -0.721296, -1.442566,
        -0.02, -0.721466, -1.442939,
        0.02, 0.721459, 1.442911,
        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4 + [5.0] * 4, dtype=np.float64)

ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
OBS_DIM = 65  # 57 + HIM gait_clock(8); old 57-D onnx still works if in_dim%57==0
ACT_DIM = 16
HISTORY_LENGTH = 1
GAIT_PERIOD_S = 0.90
GAIT_THETA = (0.0, 0.5, 0.5, 0.0)  # FL, FR, RL, RR diagonal trot
HARD_STAND_THR = 0.10
BASE_Z = 0.505

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
            print("[wd] cmd=0")
            return
        if keycode in self._reset:
            self.reset_requested = True
            self.cmd[:] = 0.0
            print("[wd] reset")
            return
        delta = self._steps.get(keycode)
        if delta is None or (now - self._last_step_t) < self._min_step_dt:
            return
        self._last_step_t = now
        self.cmd = self.cmd + delta
        self._clip()
        print(f"[wd] cmd=({self.cmd[0]:+.2f}, {self.cmd[1]:+.2f}, {self.cmd[2]:+.2f})")


class WheeldogPlayer:
    def __init__(self, xml_path: Path, onnx_path: Path, cmd0: np.ndarray, history_length: int | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._in_name = self.session.get_inputs()[0].name
        self._out_name = self.session.get_outputs()[0].name
        in_dim = int(self.session.get_inputs()[0].shape[-1])
        # Prefer 65 (with gait clock); fall back to legacy 57
        if in_dim % 65 == 0:
            self._frame_dim = 65
        elif in_dim % 57 == 0:
            self._frame_dim = 57
        else:
            raise RuntimeError(f"onnx in_dim={in_dim} not multiple of 65 or 57")
        if history_length is None:
            history_length = max(1, in_dim // self._frame_dim)
        self.history_length = int(history_length)
        if self.history_length * self._frame_dim != in_dim:
            raise RuntimeError(
                f"onnx in_dim={in_dim} != {self.history_length}*{self._frame_dim}"
            )

        self.qpos_adr = np.zeros(16, dtype=np.int32)
        self.dof_adr = np.zeros(16, dtype=np.int32)
        self.act_id = np.zeros(16, dtype=np.int32)
        for i, (jname, aname) in enumerate(zip(POLICY_JOINTS, ACTUATOR_NAMES)):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if jid < 0 or aid < 0:
                raise RuntimeError(f"missing joint/actuator: {jname} / {aname}")
            self.qpos_adr[i] = self.model.jnt_qposadr[jid]
            self.dof_adr[i] = self.model.jnt_dofadr[jid]
            self.act_id[i] = aid

        self.last_action = np.zeros(ACT_DIM, dtype=np.float64)
        self._hist = np.zeros((self.history_length, self._frame_dim), dtype=np.float32)
        self.teleop = DigitStepTeleop()
        self.teleop.cmd[:] = cmd0
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = max(1, int(round(0.02 / self.sim_dt)))
        self._reset()

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, BASE_Z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i in range(16):
            self.data.qpos[self.qpos_adr[i]] = DEFAULT_POS[i]
            self.data.qvel[self.dof_adr[i]] = 0.0
        self.data.qvel[0:6] = 0.0
        self.last_action[:] = 0.0
        self._hist[:] = 0.0
        # hold stand on position actuators
        for i in range(12):
            self.data.ctrl[self.act_id[i]] = DEFAULT_POS[i]
        for i in range(12, 16):
            self.data.ctrl[self.act_id[i]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _joint_state_policy(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.data.qpos[a] for a in self.qpos_adr], dtype=np.float64)
        qd = np.array([self.data.qvel[a] for a in self.dof_adr], dtype=np.float64)
        return q, qd

    def _gait_clock(self, cmd: np.ndarray) -> np.ndarray:
        """Match Lab gait_phase_clock (diagonal trot; freeze when standing)."""
        if float(np.linalg.norm(cmd[:3])) < HARD_STAND_THR:
            return np.zeros(8, dtype=np.float64)
        phi = (float(self.data.time) / float(GAIT_PERIOD_S)) % 1.0
        th = np.asarray(GAIT_THETA, dtype=np.float64)
        phase = (phi + th) % 1.0
        ang = 2.0 * np.pi * phase
        return np.concatenate([np.sin(ang), np.cos(ang)])

    def _build_obs(self, cmd: np.ndarray) -> np.ndarray:
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        q, qd = self._joint_state_policy()
        q_rel = q - DEFAULT_POS
        q_rel[12:] = 0.0
        parts = [
            ang_vel * ANG_VEL_SCALE,
            grav,
            cmd.astype(np.float64),
            q_rel,
            qd * JOINT_VEL_SCALE,
        ]
        # 65-D policies include gait clock before last_action; 57-D omit it
        if self._frame_dim >= 65:
            parts.append(self._gait_clock(cmd))
        parts.append(self.last_action)
        obs = np.concatenate(parts)
        assert obs.shape == (self._frame_dim,), obs.shape
        return obs.astype(np.float32)

    def _policy(self, obs_frame: np.ndarray) -> np.ndarray:
        # oldest → newest flatten (Isaac Lab history order)
        self._hist[:-1] = self._hist[1:]
        self._hist[-1] = obs_frame
        flat = self._hist.reshape(1, -1)
        out = self.session.run([self._out_name], {self._in_name: flat})[0]
        return np.asarray(out[0], dtype=np.float64)

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -2.0, 2.0)
        self.last_action = action.copy()
        q_des = DEFAULT_POS.copy()
        q_des[:12] = DEFAULT_POS[:12] + ACTION_SCALE[:12] * action[:12]
        v_des = ACTION_SCALE[12:] * action[12:]
        for i in range(12):
            self.data.ctrl[self.act_id[i]] = q_des[i]
        for i in range(4):
            self.data.ctrl[self.act_id[12 + i]] = v_des[i]

    def step_control(self) -> None:
        if self.teleop.reset_requested:
            self.teleop.reset_requested = False
            self._reset()
        obs = self._build_obs(self.teleop.cmd)
        self._apply_action(self._policy(obs))

    def run(self) -> None:
        print(
            f"[wd] sim_dt={self.sim_dt} decimation={self.decimation} "
            f"policy≈{1/(self.sim_dt*self.decimation):.0f}Hz hist={self.history_length} "
            f"(blind, no height scan)"
        )
        print("[wd] keys: 8/2 vx±  4/6 vy±  7/9 yaw±  5/0 stop  R reset")
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=lambda k: self.teleop.on_key(k)
        ) as viewer:
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            step_i = 0
            while viewer.is_running():
                t0 = time.time()
                if step_i % self.decimation == 0:
                    self.step_control()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                step_i += 1
                sleep = self.sim_dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_xml = here / "wheeldog_stairs_scene.xml"
    p = argparse.ArgumentParser(description="Wheeldog MuJoCo ONNX play (blind, multi-terrain)")
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--xml", type=Path, default=default_xml)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    args = p.parse_args()
    print(f"[wd] onnx={args.onnx}")
    print(f"[wd] xml={args.xml}")
    print(f"[wd] init cmd=({args.vx}, {args.vy}, {args.yaw})")
    WheeldogPlayer(args.xml, args.onnx, np.array([args.vx, args.vy, args.yaw])).run()


if __name__ == "__main__":
    main()
