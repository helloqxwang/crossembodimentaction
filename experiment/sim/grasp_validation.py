from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree

import mujoco
import numpy as np

from experiment.sim import rerun_scene
from experiment.sim.squeeze_control import compute_grasp_command_batch

os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_DISTURBANCE_DIRECTIONS: tuple[tuple[str, np.ndarray], ...] = (
    ("+x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ("-x", np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
    ("+y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    ("-y", np.array([0.0, -1.0, 0.0], dtype=np.float64)),
    ("+z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    ("-z", np.array([0.0, 0.0, -1.0], dtype=np.float64)),
)


@dataclass(frozen=True)
class ValidationModelMetadata:
    hand_qpos_indices: np.ndarray
    hand_qvel_indices: np.ndarray
    pd_qpos_indices: np.ndarray
    pd_qvel_indices: np.ndarray
    pd_traj_cols: np.ndarray
    virtual_qpos_indices: np.ndarray
    virtual_qvel_indices: np.ndarray
    virtual_traj_cols: np.ndarray
    object_joint_id: int
    object_qpos_adr: int
    object_body_id: int
    hand_geom_ids: tuple[int, ...]
    object_geom_ids: tuple[int, ...]
    floor_geom_id: int


@dataclass
class ValidationModelBundle:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    meta: ValidationModelMetadata


def _joint_qpos_dim(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _joint_qvel_dim(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def pose6_to_quat_wxyz(pose6: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.asarray(pose6[3:6], dtype=np.float64), "XYZ")
    return quat


def quat_error_rad_wxyz(quat_sim: np.ndarray, quat_ref: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(quat_sim * quat_ref, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def quat_error_rad_wxyz_torch(quat_sim: Any, quat_ref: Any) -> Any:
    dots = (quat_sim * quat_ref).sum(dim=1).abs().clamp(min=-1.0, max=1.0)
    return 2.0 * dots.arccos()


def resolve_disturbance_directions(names: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    resolved: list[tuple[str, np.ndarray]] = []
    for name in names:
        key = str(name)
        matched = next((item for item in DEFAULT_DISTURBANCE_DIRECTIONS if item[0] == key), None)
        if matched is None:
            raise ValueError(f"Unsupported disturbance direction: {key}")
        resolved.append((matched[0], matched[1].copy()))
    return resolved


def _infer_urdf_meshdir(urdf_path: Path, prefer_visual: bool) -> str:
    root = ElementTree.parse(urdf_path).getroot()
    collision_mesh_dirs: list[str] = []
    visual_mesh_dirs: list[str] = []
    all_mesh_dirs: list[str] = []

    for coll in root.findall(".//collision"):
        mesh = coll.find(".//mesh")
        if mesh is None:
            continue
        filename = mesh.attrib.get("filename", "")
        if filename:
            collision_mesh_dirs.append(os.path.dirname(filename))

    for vis in root.findall(".//visual"):
        mesh = vis.find(".//mesh")
        if mesh is None:
            continue
        filename = mesh.attrib.get("filename", "")
        if filename:
            visual_mesh_dirs.append(os.path.dirname(filename))

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename:
            all_mesh_dirs.append(os.path.dirname(filename))

    if prefer_visual:
        candidates = [d for d in visual_mesh_dirs if d] or [d for d in collision_mesh_dirs if d]
    else:
        candidates = [d for d in collision_mesh_dirs if d] or [d for d in visual_mesh_dirs if d]
    if not candidates:
        candidates = [d for d in all_mesh_dirs if d]
    if not candidates:
        return ""
    counts: dict[str, int] = {}
    for value in candidates:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _load_hand_spec(
    hand_path: str | Path,
    *,
    prefer_visual_mesh: bool,
    keep_visual_geoms: bool,
) -> mujoco.MjSpec:
    hand_path = Path(hand_path).expanduser().resolve()
    spec = mujoco.MjSpec.from_file(str(hand_path))
    if hand_path.suffix.lower() == ".urdf":
        meshdir = _infer_urdf_meshdir(hand_path, prefer_visual=prefer_visual_mesh)
        spec.meshdir = meshdir
        spec.compiler.meshdir = meshdir
        spec.compiler.fusestatic = 0
        if keep_visual_geoms:
            spec.compiler.discardvisual = 0
        spec.compiler.balanceinertia = 1
        spec.compiler.boundmass = max(float(spec.compiler.boundmass), 1e-6)
        spec.compiler.boundinertia = max(float(spec.compiler.boundinertia), 1e-8)
    return spec


def _build_validation_model(
    *,
    hand_path: str | Path,
    object_mesh_path: str | Path,
    sim_dt: float,
    gravity_enabled: bool,
    object_density: float,
    model_nconmax: int,
    model_njmax: int,
    prefer_visual_mesh: bool,
    keep_visual_geoms: bool,
) -> tuple[mujoco.MjSpec, mujoco.MjModel]:
    hand_path = Path(hand_path).expanduser().resolve()
    spec = _load_hand_spec(
        hand_path,
        prefer_visual_mesh=prefer_visual_mesh,
        keep_visual_geoms=keep_visual_geoms,
    )
    spec.option.timestep = float(sim_dt)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 50
    spec.option.ls_iterations = 50
    spec.option.o_solref = [0.02, 1.0]
    spec.option.o_solimp = [0.0, 0.95, 0.03, 0.5, 2.0]
    if not gravity_enabled:
        spec.option.gravity = [0.0, 0.0, 0.0]
    if int(model_nconmax) > 0:
        spec.nconmax = int(model_nconmax)
    if int(model_njmax) > 0:
        spec.njmax = int(model_njmax)

    object_mesh_path = Path(object_mesh_path).expanduser().resolve()
    object_mesh_file = str(object_mesh_path)
    if hand_path.suffix.lower() == ".urdf":
        spec.strippath = 0
        mesh_root = hand_path.parent / spec.meshdir if spec.meshdir else hand_path.parent
        object_mesh_file = os.path.relpath(object_mesh_path, mesh_root)
    spec.add_mesh(name="validation_object_mesh", file=object_mesh_file)

    object_body = spec.worldbody.add_body(name="validation_object_body", mocap=False)
    object_body.add_joint(
        name="validation_object_free_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
        damping=1.0,
        armature=0.1,
        frictionloss=0.05,
    )
    object_body.add_geom(
        name="validation_object_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="validation_object_mesh",
        density=float(object_density),
        friction=[1.0, 0.01, 0.001],
    )
    object_body.add_site(
        name="validation_object_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, 0.0],
        size=[0.01],
        rgba=[1.0, 0.0, 0.0, 1.0],
    )

    model = spec.compile()
    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not joint_name.startswith("virtual_joint_"):
            continue
        dof_adr = int(model.jnt_dofadr[joint_id])
        dof_dim = _joint_qvel_dim(int(model.jnt_type[joint_id]))
        model.dof_damping[dof_adr : dof_adr + dof_dim] = 2000.0
        model.dof_armature[dof_adr : dof_adr + dof_dim] = 100.0
        model.dof_frictionloss[dof_adr : dof_adr + dof_dim] = 50.0

    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "validation_object_free_joint")
    for joint_id in range(int(model.njnt)):
        if joint_id == object_joint_id:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if joint_name.startswith("virtual_joint_"):
            continue
        dof_adr = int(model.jnt_dofadr[joint_id])
        if _joint_qvel_dim(int(model.jnt_type[joint_id])) != 1:
            continue
        model.dof_damping[dof_adr] = max(float(model.dof_damping[dof_adr]), 2.0)
        model.dof_armature[dof_adr] = max(float(model.dof_armature[dof_adr]), 0.05)
    return spec, model


def _get_hand_state_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "validation_object_free_joint")
    hand_qpos_indices: list[int] = []
    hand_qvel_indices: list[int] = []
    for joint_id in range(int(model.njnt)):
        if joint_id == object_joint_id:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        qpos_dim = _joint_qpos_dim(int(model.jnt_type[joint_id]))
        qvel_dim = _joint_qvel_dim(int(model.jnt_type[joint_id]))
        hand_qpos_indices.extend(range(qpos_adr, qpos_adr + qpos_dim))
        hand_qvel_indices.extend(range(qvel_adr, qvel_adr + qvel_dim))
    return np.asarray(hand_qpos_indices, dtype=np.int32), np.asarray(hand_qvel_indices, dtype=np.int32)


def _get_scalar_hand_pd_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "validation_object_free_joint")
    qpos_ids: list[int] = []
    qvel_ids: list[int] = []
    for joint_id in range(int(model.njnt)):
        if joint_id == object_joint_id:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if joint_name.startswith("virtual_joint_"):
            continue
        if _joint_qpos_dim(int(model.jnt_type[joint_id])) != 1:
            continue
        if _joint_qvel_dim(int(model.jnt_type[joint_id])) != 1:
            continue
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_ids, dtype=np.int32), np.asarray(qvel_ids, dtype=np.int32)


def _get_virtual_hand_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "validation_object_free_joint")
    qpos_ids: list[int] = []
    qvel_ids: list[int] = []
    for joint_id in range(int(model.njnt)):
        if joint_id == object_joint_id:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not joint_name.startswith("virtual_joint_"):
            continue
        if _joint_qpos_dim(int(model.jnt_type[joint_id])) != 1:
            continue
        if _joint_qvel_dim(int(model.jnt_type[joint_id])) != 1:
            continue
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_ids, dtype=np.int32), np.asarray(qvel_ids, dtype=np.int32)


def _get_hand_geom_ids(
    model: mujoco.MjModel,
    *,
    object_geom_ids: tuple[int, ...],
    floor_geom_id: int,
) -> tuple[int, ...]:
    object_geom_set = set(int(geom_id) for geom_id in object_geom_ids)
    return tuple(
        int(geom_id)
        for geom_id in range(int(model.ngeom))
        if geom_id not in object_geom_set and geom_id != int(floor_geom_id)
    )


def _compute_model_metadata(model: mujoco.MjModel) -> ValidationModelMetadata:
    hand_qpos_indices, hand_qvel_indices = _get_hand_state_indices(model)
    pd_qpos_indices, pd_qvel_indices = _get_scalar_hand_pd_indices(model)
    virtual_qpos_indices, virtual_qvel_indices = _get_virtual_hand_indices(model)
    qpos_to_traj_col = {int(qid): i for i, qid in enumerate(hand_qpos_indices.tolist())}
    pd_traj_cols = np.asarray([qpos_to_traj_col[int(qid)] for qid in pd_qpos_indices.tolist()], dtype=np.int32)
    virtual_traj_cols = np.asarray(
        [qpos_to_traj_col[int(qid)] for qid in virtual_qpos_indices.tolist()],
        dtype=np.int32,
    )
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "validation_object_free_joint")
    object_qpos_adr = int(model.jnt_qposadr[object_joint_id])
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "validation_object_body")
    object_geom_ids = tuple(int(x) for x in np.where(model.geom_bodyid == object_body_id)[0].tolist())
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    hand_geom_ids = _get_hand_geom_ids(
        model,
        object_geom_ids=object_geom_ids,
        floor_geom_id=int(floor_geom_id),
    )
    return ValidationModelMetadata(
        hand_qpos_indices=hand_qpos_indices,
        hand_qvel_indices=hand_qvel_indices,
        pd_qpos_indices=pd_qpos_indices,
        pd_qvel_indices=pd_qvel_indices,
        pd_traj_cols=pd_traj_cols,
        virtual_qpos_indices=virtual_qpos_indices,
        virtual_qvel_indices=virtual_qvel_indices,
        virtual_traj_cols=virtual_traj_cols,
        object_joint_id=int(object_joint_id),
        object_qpos_adr=object_qpos_adr,
        object_body_id=int(object_body_id),
        hand_geom_ids=hand_geom_ids,
        object_geom_ids=object_geom_ids,
        floor_geom_id=int(floor_geom_id),
    )


def build_validation_model_bundle(
    *,
    hand_path: str | Path,
    object_mesh_path: str | Path,
    sim_dt: float,
    gravity_enabled: bool = True,
    object_density: float = 500.0,
    model_nconmax: int = 256,
    model_njmax: int = 1024,
    prefer_visual_mesh: bool = False,
    keep_visual_geoms: bool = False,
) -> ValidationModelBundle:
    spec, model = _build_validation_model(
        hand_path=hand_path,
        object_mesh_path=object_mesh_path,
        sim_dt=sim_dt,
        gravity_enabled=gravity_enabled,
        object_density=object_density,
        model_nconmax=model_nconmax,
        model_njmax=model_njmax,
        prefer_visual_mesh=prefer_visual_mesh,
        keep_visual_geoms=keep_visual_geoms,
    )
    return ValidationModelBundle(spec=spec, model=model, meta=_compute_model_metadata(model))


def _set_object_pose_from_pose6(model: mujoco.MjModel, data: mujoco.MjData, pose6: np.ndarray) -> None:
    pose6 = np.asarray(pose6, dtype=np.float64)
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "validation_object_free_joint")
    qpos_adr = int(model.jnt_qposadr[object_joint_id])
    quat = pose6_to_quat_wxyz(pose6)
    data.qpos[qpos_adr : qpos_adr + 3] = pose6[:3]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat


def set_object_pose6_in_qpos(
    qpos: np.ndarray,
    *,
    object_qpos_adr: int,
    pose6: np.ndarray,
) -> None:
    pose6 = np.asarray(pose6, dtype=np.float64)
    quat = pose6_to_quat_wxyz(pose6)
    qpos[..., object_qpos_adr : object_qpos_adr + 3] = pose6[:3]
    qpos[..., object_qpos_adr + 3 : object_qpos_adr + 7] = quat


def disturbance_force_magnitude(
    model: mujoco.MjModel,
    *,
    object_body_id: int,
    force_acceleration: float,
    force_cap: float,
) -> float:
    object_mass = float(model.body_mass[object_body_id])
    return min(float(object_mass * force_acceleration), float(force_cap))


def _has_hand_object_contact(model: mujoco.MjModel, data: mujoco.MjData, meta: ValidationModelMetadata) -> bool:
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if meta.floor_geom_id >= 0 and (geom1 == meta.floor_geom_id or geom2 == meta.floor_geom_id):
            continue
        if (geom1 in meta.object_geom_ids) ^ (geom2 in meta.object_geom_ids):
            return True
    return False


def _snapshot_cpu_state(data: mujoco.MjData) -> dict[str, np.ndarray | float]:
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy() if getattr(data, "act", None) is not None else np.zeros(0, dtype=np.float64),
        "qacc_warmstart": data.qacc_warmstart.copy(),
        "time": float(data.time),
    }


def _restore_cpu_state(data: mujoco.MjData, snapshot: dict[str, np.ndarray | float]) -> None:
    data.qpos[:] = np.asarray(snapshot["qpos"], dtype=np.float64)
    data.qvel[:] = np.asarray(snapshot["qvel"], dtype=np.float64)
    if data.act.size > 0:
        data.act[:] = np.asarray(snapshot["act"], dtype=np.float64)
    data.qacc_warmstart[:] = np.asarray(snapshot["qacc_warmstart"], dtype=np.float64)
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.time = float(snapshot["time"])


def _initialise_data(bundle: ValidationModelBundle, q_init: np.ndarray, object_pose6: np.ndarray) -> mujoco.MjData:
    data = mujoco.MjData(bundle.model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if data.act.size > 0:
        data.act[:] = 0.0
    q_init = np.asarray(q_init, dtype=np.float64)
    if q_init.shape != (len(bundle.meta.hand_qpos_indices),):
        raise ValueError(
            "Target grasp DOF does not match hand qpos DOF: "
            f"{q_init.shape} vs {(len(bundle.meta.hand_qpos_indices),)}"
        )
    data.qpos[bundle.meta.hand_qpos_indices] = q_init
    _set_object_pose_from_pose6(bundle.model, data, object_pose6)
    mujoco.mj_forward(bundle.model, data)
    return data


def _final_hand_object_distance_cpu(
    data: mujoco.MjData,
    meta: ValidationModelMetadata,
) -> float:
    if not meta.hand_geom_ids:
        return float("nan")
    object_pos = data.xpos[meta.object_body_id].copy()
    hand_geom_pos = np.asarray(data.geom_xpos[list(meta.hand_geom_ids)], dtype=np.float64)
    if hand_geom_pos.size == 0:
        return float("nan")
    return float(np.linalg.norm(hand_geom_pos - object_pos[None, :], axis=1).min())


def _rollout_static_grasp_cpu(
    *,
    bundle: ValidationModelBundle,
    data: mujoco.MjData,
    q_target: np.ndarray,
    object_pose6: np.ndarray,
    num_steps: int,
    steps_per_frame: int,
    in_hand_distance_threshold: float,
    hand_kp: float,
    hand_kd: float,
    hand_max_force: float,
    disturbance_force_world: np.ndarray | None,
    disturbance_push_steps: int,
    viewer_body_entity_and_ids: list[tuple[str, int]] | None,
    viewer_time_offset: float,
) -> dict[str, Any]:
    q_target = np.asarray(q_target, dtype=np.float64)
    object_pose6 = np.asarray(object_pose6, dtype=np.float64)
    meta = bundle.meta
    ref_obj_pos = np.repeat(object_pose6[None, :3], max(1, int(num_steps)), axis=0)
    ref_obj_quat = np.repeat(pose6_to_quat_wxyz(object_pose6)[None, :], max(1, int(num_steps)), axis=0)
    sim_obj_pos = np.zeros((max(1, int(num_steps)), 3), dtype=np.float64)
    sim_obj_quat = np.zeros((max(1, int(num_steps)), 4), dtype=np.float64)
    unstable = False
    unstable_step: int | None = None
    filled_steps = 0

    for step_idx in range(int(num_steps)):
        data.xfrc_applied[:] = 0.0
        if disturbance_force_world is not None and step_idx < int(disturbance_push_steps):
            data.xfrc_applied[meta.object_body_id, :3] = disturbance_force_world

        if len(meta.virtual_qpos_indices) > 0:
            data.qpos[meta.virtual_qpos_indices] = q_target[meta.virtual_traj_cols]
            data.qvel[meta.virtual_qvel_indices] = 0.0
            mujoco.mj_forward(bundle.model, data)

        data.qfrc_applied[:] = 0.0
        if len(meta.pd_qvel_indices) > 0:
            q_cur = data.qpos[meta.pd_qpos_indices]
            q_vel = data.qvel[meta.pd_qvel_indices]
            tau = hand_kp * (q_target[meta.pd_traj_cols] - q_cur) - hand_kd * q_vel
            tau = np.clip(tau, -abs(hand_max_force), abs(hand_max_force))
            data.qfrc_applied[meta.pd_qvel_indices] = tau

        for _ in range(max(1, int(steps_per_frame))):
            mujoco.mj_step(bundle.model, data)
            if (not np.isfinite(data.qpos).all()) or (not np.isfinite(data.qvel).all()):
                unstable = True
                unstable_step = int(step_idx)
                break
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        if unstable:
            break

        sim_obj_pos[filled_steps] = data.xpos[meta.object_body_id].copy()
        sim_obj_quat[filled_steps] = data.xquat[meta.object_body_id].copy()
        filled_steps += 1
        if viewer_body_entity_and_ids:
            rerun_scene.log_frame(
                data,
                sim_time=float(data.time) + float(viewer_time_offset),
                viewer_body_entity_and_ids=viewer_body_entity_and_ids,
            )

    if filled_steps == 0:
        sim_obj_pos[0] = data.xpos[meta.object_body_id].copy()
        sim_obj_quat[0] = data.xquat[meta.object_body_id].copy()
        filled_steps = 1

    pos_err = np.linalg.norm(sim_obj_pos[:filled_steps] - ref_obj_pos[:filled_steps], axis=1)
    rot_err = quat_error_rad_wxyz(sim_obj_quat[:filled_steps], ref_obj_quat[:filled_steps])
    final_contact = False if unstable else _has_hand_object_contact(bundle.model, data, meta)
    final_hand_obj_dist = float("nan") if unstable else _final_hand_object_distance_cpu(data, meta)
    final_in_hand = bool((not unstable) and final_contact and final_hand_obj_dist <= float(in_hand_distance_threshold))

    return {
        "num_steps": int(filled_steps),
        "unstable": bool(unstable),
        "unstable_step": unstable_step,
        "summary": {
            "pos_err_mean": float(pos_err.mean()),
            "pos_err_max": float(pos_err.max()),
            "pos_err_final": float(pos_err[-1]),
            "rot_err_mean": float(rot_err.mean()),
            "rot_err_max": float(rot_err.max()),
            "rot_err_final": float(rot_err[-1]),
            "final_contact": bool(final_contact),
            "final_hand_obj_distance": float(final_hand_obj_dist),
            "final_in_hand": bool(final_in_hand),
        },
        "data": data,
    }


def _failed_direction_result(direction_name: str, force_world: np.ndarray) -> dict[str, Any]:
    return {
        "direction": str(direction_name),
        "force_world": np.asarray(force_world, dtype=np.float64),
        "force_magnitude": float(np.linalg.norm(force_world)),
        "num_steps": 0,
        "unstable": True,
        "unstable_step": 0,
        "summary": {
            "pos_err_mean": float("nan"),
            "pos_err_max": float("nan"),
            "pos_err_final": float("nan"),
            "rot_err_mean": float("nan"),
            "rot_err_max": float("nan"),
            "rot_err_final": float("nan"),
            "final_contact": False,
            "final_hand_obj_distance": float("nan"),
            "final_in_hand": False,
        },
    }


def run_static_grasp_validation_cpu(
    *,
    robot_name: str,
    q_target: np.ndarray,
    object_pose6: np.ndarray,
    hand_path: str | Path,
    object_mesh_path: str | Path,
    sim_dt: float,
    num_steps: int,
    steps_per_frame: int,
    in_hand_distance_threshold: float,
    hand_kp: float,
    hand_kd: float,
    hand_max_force: float,
    gravity_enabled: bool = True,
    squeezing_enabled: bool = False,
    object_density: float = 500.0,
    model_nconmax: int = 256,
    model_njmax: int = 1024,
    disturbance_enabled: bool = True,
    disturbance_push_steps: int = 100,
    disturbance_recovery_steps: int = 200,
    disturbance_force_acceleration: float = 2.45,
    disturbance_force_cap: float = 1.0,
    disturbance_directions: Sequence[tuple[str, Sequence[float]]] = DEFAULT_DISTURBANCE_DIRECTIONS,
    show_rerun: bool = False,
    use_visual_mesh_for_rerun: bool = True,
    save_rerun_path: str | Path | None = None,
) -> dict[str, Any]:
    commands = compute_grasp_command_batch(
        robot_name=robot_name,
        hand_path=hand_path,
        q_batch=np.asarray(q_target, dtype=np.float32),
        squeeze_enabled=squeezing_enabled,
    )
    q_init = np.asarray(commands.q_init[0], dtype=np.float64)
    q_target_exec = np.asarray(commands.q_target[0], dtype=np.float64)
    bundle = build_validation_model_bundle(
        hand_path=hand_path,
        object_mesh_path=object_mesh_path,
        sim_dt=sim_dt,
        gravity_enabled=gravity_enabled,
        object_density=object_density,
        model_nconmax=model_nconmax,
        model_njmax=model_njmax,
        prefer_visual_mesh=False,
        keep_visual_geoms=False,
    )

    viewer_body_entity_and_ids: list[tuple[str, int]] = []
    viewer_error: str | None = None
    viewer_enabled = bool(show_rerun) or save_rerun_path is not None
    if viewer_enabled:
        try:
            rerun_scene.init_rerun(app_name="grasp_playback", spawn=bool(show_rerun))
            viewer_bundle = bundle
            if use_visual_mesh_for_rerun:
                viewer_bundle = build_validation_model_bundle(
                    hand_path=hand_path,
                    object_mesh_path=object_mesh_path,
                    sim_dt=sim_dt,
                    gravity_enabled=gravity_enabled,
                    object_density=object_density,
                    model_nconmax=model_nconmax,
                    model_njmax=model_njmax,
                    prefer_visual_mesh=True,
                    keep_visual_geoms=True,
            )
            viewer_body_entity_and_ids = rerun_scene.build_and_log_scene_from_spec(
                viewer_bundle.spec,
                viewer_bundle.model,
            )
        except Exception as exc:  # noqa: BLE001
            viewer_error = str(exc)
            viewer_body_entity_and_ids = []

    data = _initialise_data(bundle, q_init, object_pose6)
    base_result = _rollout_static_grasp_cpu(
        bundle=bundle,
        data=data,
        q_target=q_target_exec,
        object_pose6=object_pose6,
        num_steps=num_steps,
        steps_per_frame=steps_per_frame,
        in_hand_distance_threshold=in_hand_distance_threshold,
        hand_kp=hand_kp,
        hand_kd=hand_kd,
        hand_max_force=hand_max_force,
        disturbance_force_world=None,
        disturbance_push_steps=0,
        viewer_body_entity_and_ids=viewer_body_entity_and_ids,
        viewer_time_offset=0.0,
    )

    disturbance_payload: dict[str, Any] | None = None
    if disturbance_enabled:
        force_magnitude = disturbance_force_magnitude(
            bundle.model,
            object_body_id=bundle.meta.object_body_id,
            force_acceleration=disturbance_force_acceleration,
            force_cap=disturbance_force_cap,
        )
        base_snapshot = _snapshot_cpu_state(base_result["data"])
        direction_results: list[dict[str, Any]] = []
        time_offset = float(base_result["num_steps"] * sim_dt)
        for direction_name, direction_vector in disturbance_directions:
            force_world = force_magnitude * np.asarray(direction_vector, dtype=np.float64)
            if base_result["unstable"]:
                direction_results.append(_failed_direction_result(direction_name, force_world))
                continue
            disturbance_data = mujoco.MjData(bundle.model)
            _restore_cpu_state(disturbance_data, base_snapshot)
            mujoco.mj_forward(bundle.model, disturbance_data)
            result = _rollout_static_grasp_cpu(
                bundle=bundle,
                data=disturbance_data,
                q_target=q_target_exec,
                object_pose6=object_pose6,
                num_steps=int(disturbance_push_steps) + int(disturbance_recovery_steps),
                steps_per_frame=steps_per_frame,
                in_hand_distance_threshold=in_hand_distance_threshold,
                hand_kp=hand_kp,
                hand_kd=hand_kd,
                hand_max_force=hand_max_force,
                disturbance_force_world=force_world,
                disturbance_push_steps=int(disturbance_push_steps),
                viewer_body_entity_and_ids=viewer_body_entity_and_ids,
                viewer_time_offset=time_offset,
            )
            direction_results.append(
                {
                    "direction": str(direction_name),
                    "force_world": force_world.astype(np.float64),
                    "force_magnitude": float(np.linalg.norm(force_world)),
                    "num_steps": int(result["num_steps"]),
                    "unstable": bool(result["unstable"]),
                    "unstable_step": result["unstable_step"],
                    "summary": result["summary"],
                }
            )
            time_offset += float(result["num_steps"] * sim_dt)

        all_directions_in_hand = all(
            bool(direction["summary"]["final_in_hand"]) and (not bool(direction["unstable"]))
            for direction in direction_results
        )
        disturbance_payload = {
            "enabled": True,
            "force_acceleration": float(disturbance_force_acceleration),
            "force_cap": float(disturbance_force_cap),
            "force_magnitude": float(force_magnitude),
            "push_steps": int(disturbance_push_steps),
            "recovery_steps": int(disturbance_recovery_steps),
            "settle_steps": int(base_result["num_steps"]),
            "all_directions_in_hand": bool(all_directions_in_hand),
            "directions": direction_results,
        }

    if save_rerun_path is not None:
        rerun_scene.save_rerun(save_rerun_path)

    return {
        "backend": "cpu",
        "num_steps": int(base_result["num_steps"]),
        "summary": base_result["summary"],
        "unstable": bool(base_result["unstable"]),
        "unstable_step": base_result["unstable_step"],
        "disturbance": disturbance_payload,
        "viewer_error": viewer_error,
    }
