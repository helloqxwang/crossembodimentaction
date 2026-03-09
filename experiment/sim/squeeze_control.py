from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np


@dataclass(frozen=True)
class GraspCommandBatch:
    q_init: np.ndarray
    q_target: np.ndarray


@dataclass(frozen=True)
class _JointControlInfo:
    joint_id: int
    joint_name: str
    body_id: int
    lower: float
    upper: float
    link_dir_local: np.ndarray


@dataclass(frozen=True)
class _HandKinematics:
    model: mujoco.MjModel
    controlled_joints: tuple[_JointControlInfo, ...]


_HAND_MODEL_CACHE: dict[tuple[str, str], _HandKinematics] = {}


def _get_link_dir(robot_name: str, joint_name: str) -> np.ndarray | None:
    if joint_name.startswith("virtual"):
        return None
    if robot_name == "allegro":
        if joint_name in {"joint_0.0", "joint_4.0", "joint_8.0", "joint_13.0"}:
            return None
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if robot_name == "barrett":
        if joint_name in {"bh_j11_joint", "bh_j21_joint"}:
            return None
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    if robot_name == "ezgripper":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if robot_name == "robotiq_3finger":
        if joint_name in {"gripper_fingerB_knuckle", "gripper_fingerC_knuckle"}:
            return None
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if robot_name == "shadowhand":
        if joint_name in {"WRJ2", "WRJ1"}:
            return None
        if joint_name == "THJ5":
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if robot_name == "leaphand":
        if joint_name == "13":
            return None
        if joint_name in {"0", "4", "8"}:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if joint_name in {"1", "5", "9", "12", "14"}:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    raise NotImplementedError(f"Unsupported robot for squeeze controller: {robot_name}")


def _infer_urdf_meshdir(urdf_path: Path) -> str:
    root = ElementTree.parse(urdf_path).getroot()
    candidates: list[str] = []
    for mesh in root.findall(".//collision//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename:
            candidates.append(os.path.dirname(filename))
    if not candidates:
        for mesh in root.findall(".//visual//mesh"):
            filename = mesh.attrib.get("filename", "")
            if filename:
                candidates.append(os.path.dirname(filename))
    if not candidates:
        return ""
    counts: dict[str, int] = {}
    for value in candidates:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _load_hand_kinematics(robot_name: str, hand_path: str | Path) -> _HandKinematics:
    resolved_path = Path(hand_path).expanduser().resolve()
    key = (str(robot_name), str(resolved_path))
    cached = _HAND_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    spec = mujoco.MjSpec.from_file(str(resolved_path))
    if resolved_path.suffix.lower() == ".urdf":
        meshdir = _infer_urdf_meshdir(resolved_path)
        spec.meshdir = meshdir
        spec.compiler.meshdir = meshdir
        spec.compiler.fusestatic = 0
        spec.compiler.balanceinertia = 1
        spec.compiler.boundmass = max(float(spec.compiler.boundmass), 1e-6)
        spec.compiler.boundinertia = max(float(spec.compiler.boundinertia), 1e-8)
    model = spec.compile()

    controlled: list[_JointControlInfo] = []
    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if int(model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        link_dir_local = _get_link_dir(robot_name, joint_name)
        if link_dir_local is None:
            continue
        controlled.append(
            _JointControlInfo(
                joint_id=int(joint_id),
                joint_name=joint_name,
                body_id=int(model.jnt_bodyid[joint_id]),
                lower=float(model.jnt_range[joint_id][0]),
                upper=float(model.jnt_range[joint_id][1]),
                link_dir_local=link_dir_local,
            )
        )
    result = _HandKinematics(model=model, controlled_joints=tuple(controlled))
    _HAND_MODEL_CACHE[key] = result
    return result


def compute_grasp_command_batch(
    *,
    robot_name: str,
    hand_path: str | Path,
    q_batch: np.ndarray,
    squeeze_enabled: bool,
) -> GraspCommandBatch:
    q_batch = np.asarray(q_batch, dtype=np.float32)
    if q_batch.ndim == 1:
        q_batch = q_batch[None, :]
    if not squeeze_enabled:
        return GraspCommandBatch(q_init=q_batch.copy(), q_target=q_batch.copy())

    kin = _load_hand_kinematics(robot_name, hand_path)
    model = kin.model
    if q_batch.shape[1] != int(model.nq):
        raise ValueError(f"Squeeze controller expected q dim {int(model.nq)}, got {q_batch.shape[1]}")

    outer = q_batch.astype(np.float64, copy=True)
    inner = q_batch.astype(np.float64, copy=True)
    data = mujoco.MjData(model)
    for sample_idx in range(q_batch.shape[0]):
        data.qpos[:] = q_batch[sample_idx].astype(np.float64, copy=False)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for info in kin.controlled_joints:
            qpos_adr = int(model.jnt_qposadr[info.joint_id])
            axis_dir = np.asarray(data.xaxis[info.joint_id], dtype=np.float64)
            body_rot = np.asarray(data.xmat[info.body_id], dtype=np.float64).reshape(3, 3)
            link_dir = body_rot @ info.link_dir_local
            normal_dir = np.cross(axis_dir, link_dir)
            axis_origin = np.asarray(data.xanchor[info.joint_id], dtype=np.float64)
            origin_dir = -axis_origin / max(float(np.linalg.norm(axis_origin)), 1e-8)
            dot = float(normal_dir @ origin_dir)
            if robot_name == "robotiq_3finger":
                outer_delta = (
                    outer[sample_idx, qpos_adr] - info.lower
                    if dot <= 0.0
                    else outer[sample_idx, qpos_adr] - info.upper
                )
                inner_delta = (
                    inner[sample_idx, qpos_adr] - info.upper
                    if dot <= 0.0
                    else inner[sample_idx, qpos_adr] - info.lower
                )
                outer[sample_idx, qpos_adr] = outer[sample_idx, qpos_adr] + 0.25 * outer_delta
                inner[sample_idx, qpos_adr] = inner[sample_idx, qpos_adr] + 0.15 * inner_delta
            else:
                outer_goal = info.lower if dot >= 0.0 else info.upper
                inner_goal = info.upper if dot >= 0.0 else info.lower
                outer[sample_idx, qpos_adr] = outer[sample_idx, qpos_adr] + 0.25 * (
                    outer_goal - outer[sample_idx, qpos_adr]
                )
                inner[sample_idx, qpos_adr] = inner[sample_idx, qpos_adr] + 0.15 * (
                    inner_goal - inner[sample_idx, qpos_adr]
                )

    return GraspCommandBatch(
        q_init=outer.astype(np.float32, copy=False),
        q_target=inner.astype(np.float32, copy=False),
    )
