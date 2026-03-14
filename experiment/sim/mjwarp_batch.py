from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import torch

os.environ.setdefault("WARP_CACHE_PATH", str((Path("/tmp") / "warp-cache").resolve()))

import mujoco_warp as mjwarp
import warp as wp

from experiment.sim.grasp_validation import (
    ValidationModelBundle,
    build_validation_model_bundle,
    disturbance_force_magnitude,
    pose6_to_quat_wxyz,
    quat_error_rad_wxyz_torch,
    set_object_pose6_in_qpos,
)
from experiment.sim.squeeze_control import compute_grasp_command_batch

try:
    wp.init()
except RuntimeError:
    pass


@dataclass
class MjWarpRuntime:
    device: str
    torch_device: torch.device
    num_worlds: int
    model_cpu: mujoco.MjModel
    model_wp: Any
    data_wp: Any
    graph: Any | None


def _create_runtime(
    *,
    model: mujoco.MjModel,
    device: str,
    num_worlds: int,
    nconmax_per_world: int,
    njmax_per_world: int,
    capture_cuda_graph: bool,
) -> MjWarpRuntime:
    torch_device = torch.device("cpu" if device == "cpu" else device)
    wp.set_device(device)
    with wp.ScopedDevice(device):
        model_wp = mjwarp.put_model(model)
        data_wp = mjwarp.put_data(
            model,
            mujoco.MjData(model),
            nworld=int(num_worlds),
            nconmax=int(nconmax_per_world),
            njmax=int(njmax_per_world),
        )
        graph = None
        if capture_cuda_graph and device.startswith("cuda"):
            mjwarp.step(model_wp, data_wp)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                mjwarp.step(model_wp, data_wp)
            wp.synchronize()
            graph = capture.graph
    return MjWarpRuntime(
        device=device,
        torch_device=torch_device,
        num_worlds=int(num_worlds),
        model_cpu=model,
        model_wp=model_wp,
        data_wp=data_wp,
        graph=graph,
    )


def _step_runtime(runtime: MjWarpRuntime) -> None:
    with wp.ScopedDevice(runtime.device):
        if runtime.graph is not None:
            wp.capture_launch(runtime.graph)
        else:
            mjwarp.step(runtime.model_wp, runtime.data_wp)
        wp.synchronize()


def _copy_to_warp(dst: Any, value: torch.Tensor) -> None:
    wp.copy(dst, wp.from_torch(value))


def _snapshot_runtime_state(runtime: MjWarpRuntime) -> dict[str, torch.Tensor]:
    fields = [
        "qpos",
        "qvel",
        "qacc",
        "time",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
    ]
    out: dict[str, torch.Tensor] = {}
    for name in fields:
        if hasattr(runtime.data_wp, name):
            out[name] = wp.to_torch(getattr(runtime.data_wp, name)).clone()
    return out


def _restore_runtime_state(runtime: MjWarpRuntime, snapshot: dict[str, torch.Tensor]) -> None:
    for name, tensor in snapshot.items():
        if hasattr(runtime.data_wp, name):
            _copy_to_warp(getattr(runtime.data_wp, name), tensor)


def _set_initial_state(
    runtime: MjWarpRuntime,
    bundle: ValidationModelBundle,
    q_batch: torch.Tensor,
    object_pose6: np.ndarray,
) -> None:
    meta = bundle.meta
    qpos = torch.zeros(
        (runtime.num_worlds, int(bundle.model.nq)),
        dtype=torch.float32,
        device=runtime.torch_device,
    )
    qvel = torch.zeros(
        (runtime.num_worlds, int(bundle.model.nv)),
        dtype=torch.float32,
        device=runtime.torch_device,
    )
    qpos[:, meta.hand_qpos_indices.tolist()] = q_batch.to(torch.float32)
    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    set_object_pose6_in_qpos(
        qpos_np,
        object_qpos_adr=meta.object_qpos_adr,
        pose6=np.asarray(object_pose6, dtype=np.float64),
    )
    qpos = torch.from_numpy(qpos_np).to(runtime.torch_device, dtype=torch.float32)

    _copy_to_warp(runtime.data_wp.qpos, qpos)
    _copy_to_warp(runtime.data_wp.qvel, qvel)
    if hasattr(runtime.data_wp, "qacc"):
        _copy_to_warp(runtime.data_wp.qacc, torch.zeros_like(wp.to_torch(runtime.data_wp.qacc)))
    if hasattr(runtime.data_wp, "time"):
        _copy_to_warp(runtime.data_wp.time, torch.zeros_like(wp.to_torch(runtime.data_wp.time)))
    if hasattr(runtime.data_wp, "qacc_warmstart"):
        _copy_to_warp(
            runtime.data_wp.qacc_warmstart,
            torch.zeros_like(wp.to_torch(runtime.data_wp.qacc_warmstart)),
        )
    _copy_to_warp(runtime.data_wp.qfrc_applied, torch.zeros_like(wp.to_torch(runtime.data_wp.qfrc_applied)))
    _copy_to_warp(runtime.data_wp.xfrc_applied, torch.zeros_like(wp.to_torch(runtime.data_wp.xfrc_applied)))


def _sanitize_inactive_worlds(
    runtime: MjWarpRuntime,
    *,
    inactive_mask: torch.Tensor,
) -> None:
    if not bool(inactive_mask.any()):
        return
    qpos = wp.to_torch(runtime.data_wp.qpos)
    qvel = wp.to_torch(runtime.data_wp.qvel)
    qfrc_applied = wp.to_torch(runtime.data_wp.qfrc_applied)
    xfrc_applied = wp.to_torch(runtime.data_wp.xfrc_applied)
    qpos[inactive_mask] = 0.0
    qvel[inactive_mask] = 0.0
    qfrc_applied[inactive_mask] = 0.0
    xfrc_applied[inactive_mask] = 0.0
    _copy_to_warp(runtime.data_wp.qpos, qpos)
    _copy_to_warp(runtime.data_wp.qvel, qvel)
    _copy_to_warp(runtime.data_wp.qfrc_applied, qfrc_applied)
    _copy_to_warp(runtime.data_wp.xfrc_applied, xfrc_applied)
    if hasattr(runtime.data_wp, "qacc_warmstart"):
        qacc_warmstart = wp.to_torch(runtime.data_wp.qacc_warmstart)
        qacc_warmstart[inactive_mask] = 0.0
        _copy_to_warp(runtime.data_wp.qacc_warmstart, qacc_warmstart)


def _final_contact_mask(
    runtime: MjWarpRuntime,
    bundle: ValidationModelBundle,
) -> torch.Tensor:
    contact_geom = wp.to_torch(runtime.data_wp.contact.geom)
    contact_worldid = wp.to_torch(runtime.data_wp.contact.worldid).long()
    contact_dist = wp.to_torch(runtime.data_wp.contact.dist)
    used = (contact_dist != 0.0) | (contact_geom[:, 0] != 0) | (contact_geom[:, 1] != 0)
    if bundle.meta.floor_geom_id >= 0:
        used &= (contact_geom[:, 0] != bundle.meta.floor_geom_id) & (contact_geom[:, 1] != bundle.meta.floor_geom_id)
    out = torch.zeros(runtime.num_worlds, dtype=torch.bool, device=runtime.torch_device)
    if not bool(used.any()):
        return out
    object_geoms = torch.as_tensor(
        bundle.meta.object_geom_ids,
        dtype=contact_geom.dtype,
        device=runtime.torch_device,
    )
    geom_used = contact_geom[used]
    g0_obj = (geom_used[:, 0:1] == object_geoms[None, :]).any(dim=1)
    g1_obj = (geom_used[:, 1:2] == object_geoms[None, :]).any(dim=1)
    hand_obj = g0_obj ^ g1_obj
    if not bool(hand_obj.any()):
        return out
    world_ids = contact_worldid[used][hand_obj].clamp(min=0, max=runtime.num_worlds - 1)
    out[world_ids] = True
    return out


def _final_hand_object_distance(
    runtime: MjWarpRuntime,
    bundle: ValidationModelBundle,
    object_pos: torch.Tensor,
) -> torch.Tensor:
    if not bundle.meta.hand_geom_ids:
        return torch.full(
            (runtime.num_worlds,),
            float("nan"),
            dtype=torch.float32,
            device=runtime.torch_device,
        )
    geom_xpos = wp.to_torch(runtime.data_wp.geom_xpos)[:, list(bundle.meta.hand_geom_ids), :]
    return torch.linalg.vector_norm(geom_xpos - object_pos[:, None, :], dim=2).amin(dim=1)


def _rollout_static_batch(
    runtime: MjWarpRuntime,
    bundle: ValidationModelBundle,
    *,
    q_batch: torch.Tensor,
    object_pose6: np.ndarray,
    num_steps: int,
    steps_per_frame: int,
    in_hand_distance_threshold: float,
    hand_kp: float,
    hand_kd: float,
    hand_max_force: float,
    active_mask: torch.Tensor | None,
    disturbance_force_world: torch.Tensor | None,
    disturbance_push_steps: int,
) -> dict[str, Any]:
    meta = bundle.meta
    num_worlds = runtime.num_worlds
    if active_mask is None:
        active_mask = torch.ones(num_worlds, dtype=torch.bool, device=runtime.torch_device)
    else:
        active_mask = active_mask.to(device=runtime.torch_device, dtype=torch.bool).clone()

    q_des_pd = q_batch[:, meta.pd_traj_cols.tolist()] if len(meta.pd_traj_cols) > 0 else None
    q_des_virtual = (
        q_batch[:, meta.virtual_traj_cols.tolist()]
        if len(meta.virtual_traj_cols) > 0
        else None
    )
    ref_obj_pos = torch.as_tensor(object_pose6[:3], dtype=torch.float32, device=runtime.torch_device).repeat(num_worlds, 1)
    ref_obj_quat = torch.as_tensor(
        pose6_to_quat_wxyz(np.asarray(object_pose6, dtype=np.float64)),
        dtype=torch.float32,
        device=runtime.torch_device,
    ).repeat(num_worlds, 1)

    pos_err_sum = torch.zeros(num_worlds, dtype=torch.float32, device=runtime.torch_device)
    pos_err_max = torch.zeros(num_worlds, dtype=torch.float32, device=runtime.torch_device)
    rot_err_sum = torch.zeros(num_worlds, dtype=torch.float32, device=runtime.torch_device)
    rot_err_max = torch.zeros(num_worlds, dtype=torch.float32, device=runtime.torch_device)
    valid_steps = torch.zeros(num_worlds, dtype=torch.int64, device=runtime.torch_device)
    unstable = torch.zeros(num_worlds, dtype=torch.bool, device=runtime.torch_device)
    unstable_step = torch.full((num_worlds,), -1, dtype=torch.int64, device=runtime.torch_device)
    last_obj_pos = torch.zeros((num_worlds, 3), dtype=torch.float32, device=runtime.torch_device)
    last_obj_quat = torch.zeros((num_worlds, 4), dtype=torch.float32, device=runtime.torch_device)

    for step_idx in range(int(num_steps)):
        for _ in range(max(1, int(steps_per_frame))):
            qpos = wp.to_torch(runtime.data_wp.qpos)
            qvel = wp.to_torch(runtime.data_wp.qvel)
            qfrc_applied = wp.to_torch(runtime.data_wp.qfrc_applied)
            xfrc_applied = wp.to_torch(runtime.data_wp.xfrc_applied)

            if q_des_virtual is not None:
                qpos[:, meta.virtual_qpos_indices.tolist()] = torch.where(
                    active_mask[:, None],
                    q_des_virtual,
                    qpos[:, meta.virtual_qpos_indices.tolist()],
                )
                qvel[:, meta.virtual_qvel_indices.tolist()] = torch.where(
                    active_mask[:, None],
                    torch.zeros_like(qvel[:, meta.virtual_qvel_indices.tolist()]),
                    qvel[:, meta.virtual_qvel_indices.tolist()],
                )

            qfrc_applied.zero_()
            if q_des_pd is not None:
                q_cur = qpos[:, meta.pd_qpos_indices.tolist()]
                q_vel = qvel[:, meta.pd_qvel_indices.tolist()]
                tau = hand_kp * (q_des_pd - q_cur) - hand_kd * q_vel
                tau = tau.clamp(min=-abs(hand_max_force), max=abs(hand_max_force))
                tau = tau * active_mask[:, None].to(dtype=tau.dtype)
                qfrc_applied[:, meta.pd_qvel_indices.tolist()] = tau

            xfrc_applied.zero_()
            if disturbance_force_world is not None and step_idx < int(disturbance_push_steps):
                xfrc_applied[:, meta.object_body_id, :3] = disturbance_force_world

            _copy_to_warp(runtime.data_wp.qpos, qpos)
            _copy_to_warp(runtime.data_wp.qvel, qvel)
            _copy_to_warp(runtime.data_wp.qfrc_applied, qfrc_applied)
            _copy_to_warp(runtime.data_wp.xfrc_applied, xfrc_applied)

            _step_runtime(runtime)

            qpos_after = wp.to_torch(runtime.data_wp.qpos)
            qvel_after = wp.to_torch(runtime.data_wp.qvel)
            finite_mask = torch.isfinite(qpos_after).all(dim=1) & torch.isfinite(qvel_after).all(dim=1)
            newly_unstable = active_mask & (~finite_mask)
            unstable |= newly_unstable
            unstable_step = torch.where(
                newly_unstable & (unstable_step < 0),
                torch.full_like(unstable_step, int(step_idx)),
                unstable_step,
            )
            active_mask &= finite_mask
            _sanitize_inactive_worlds(runtime, inactive_mask=(~active_mask))
            if not bool(active_mask.any()):
                break
        if not bool(active_mask.any()):
            break

        xpos = wp.to_torch(runtime.data_wp.xpos)
        xquat = wp.to_torch(runtime.data_wp.xquat)
        obj_pos = xpos[:, meta.object_body_id, :]
        obj_quat = xquat[:, meta.object_body_id, :]
        pos_err = torch.linalg.vector_norm(obj_pos - ref_obj_pos, dim=1)
        rot_err = quat_error_rad_wxyz_torch(obj_quat, ref_obj_quat)
        active_float = active_mask.to(dtype=pos_err.dtype)
        pos_err_sum += pos_err * active_float
        rot_err_sum += rot_err * active_float
        pos_err_max = torch.maximum(pos_err_max, pos_err * active_float)
        rot_err_max = torch.maximum(rot_err_max, rot_err * active_float)
        valid_steps += active_mask.to(dtype=torch.int64)
        last_obj_pos = torch.where(active_mask[:, None], obj_pos, last_obj_pos)
        last_obj_quat = torch.where(active_mask[:, None], obj_quat, last_obj_quat)

    zero_steps = valid_steps == 0
    if bool(zero_steps.any()):
        xpos = wp.to_torch(runtime.data_wp.xpos)
        xquat = wp.to_torch(runtime.data_wp.xquat)
        obj_pos = xpos[:, meta.object_body_id, :]
        obj_quat = xquat[:, meta.object_body_id, :]
        pos_err0 = torch.linalg.vector_norm(obj_pos - ref_obj_pos, dim=1)
        rot_err0 = quat_error_rad_wxyz_torch(obj_quat, ref_obj_quat)
        pos_err_sum = torch.where(zero_steps, pos_err0, pos_err_sum)
        pos_err_max = torch.where(zero_steps, pos_err0, pos_err_max)
        rot_err_sum = torch.where(zero_steps, rot_err0, rot_err_sum)
        rot_err_max = torch.where(zero_steps, rot_err0, rot_err_max)
        valid_steps = torch.where(zero_steps, torch.ones_like(valid_steps), valid_steps)
        last_obj_pos = torch.where(zero_steps[:, None], obj_pos, last_obj_pos)
        last_obj_quat = torch.where(zero_steps[:, None], obj_quat, last_obj_quat)

    xpos = wp.to_torch(runtime.data_wp.xpos)
    final_contact = _final_contact_mask(runtime, bundle) & (~unstable)
    final_hand_obj_distance = _final_hand_object_distance(runtime, bundle, last_obj_pos)
    final_hand_obj_distance = torch.where(
        unstable,
        torch.full_like(final_hand_obj_distance, float("nan")),
        final_hand_obj_distance,
    )
    final_in_hand = (~unstable) & final_contact & (final_hand_obj_distance <= float(in_hand_distance_threshold))

    valid_steps_float = valid_steps.to(dtype=torch.float32).clamp(min=1.0)
    return {
        "num_steps": valid_steps.detach().cpu().numpy().astype(np.int64),
        "unstable": unstable.detach().cpu().numpy().astype(bool),
        "unstable_step": unstable_step.detach().cpu().numpy().astype(np.int64),
        "summary": {
            "pos_err_mean": (pos_err_sum / valid_steps_float).detach().cpu().numpy().astype(np.float64),
            "pos_err_max": pos_err_max.detach().cpu().numpy().astype(np.float64),
            "pos_err_final": (
                pos_err_sum.new_zeros(num_worlds) + torch.linalg.vector_norm(last_obj_pos - ref_obj_pos, dim=1)
            ).detach().cpu().numpy().astype(np.float64),
            "rot_err_mean": (rot_err_sum / valid_steps_float).detach().cpu().numpy().astype(np.float64),
            "rot_err_max": rot_err_max.detach().cpu().numpy().astype(np.float64),
            "rot_err_final": quat_error_rad_wxyz_torch(last_obj_quat, ref_obj_quat).detach().cpu().numpy().astype(np.float64),
            "final_contact": final_contact.detach().cpu().numpy().astype(bool),
            "final_hand_obj_distance": final_hand_obj_distance.detach().cpu().numpy().astype(np.float64),
            "final_in_hand": final_in_hand.detach().cpu().numpy().astype(bool),
        },
    }


def evaluate_static_grasp_batch_mjwarp(
    samples: list[dict[str, Any]],
    *,
    robot_name: str,
    hand_path: str,
    object_mesh_path: str,
    object_pose6: np.ndarray,
    sim_dt: float,
    num_steps: int,
    steps_per_frame: int,
    in_hand_distance_threshold: float,
    hand_kp: float,
    hand_kd: float,
    hand_max_force: float,
    gravity_enabled: bool,
    squeezing_enabled: bool,
    object_density: float,
    disturbance_enabled: bool,
    disturbance_push_steps: int,
    disturbance_recovery_steps: int,
    disturbance_force_acceleration: float,
    disturbance_force_cap: float,
    disturbance_directions: Sequence[tuple[str, Sequence[float]]],
    device: str,
    max_worlds_per_batch: int,
    nconmax_per_world: int,
    njmax_per_world: int,
    capture_cuda_graph: bool,
) -> list[dict[str, Any]]:
    if len(samples) == 0:
        return []

    bundle = build_validation_model_bundle(
        hand_path=hand_path,
        object_mesh_path=object_mesh_path,
        sim_dt=sim_dt,
        gravity_enabled=gravity_enabled,
        object_density=object_density,
        model_nconmax=nconmax_per_world,
        model_njmax=njmax_per_world,
        prefer_visual_mesh=False,
        keep_visual_geoms=False,
    )

    records: list[dict[str, Any]] = []
    for start in range(0, len(samples), int(max_worlds_per_batch)):
        chunk = samples[start : start + int(max_worlds_per_batch)]
        q_batch_np = np.stack([np.asarray(sample["q_eval"], dtype=np.float32) for sample in chunk], axis=0)
        commands = compute_grasp_command_batch(
            robot_name=robot_name,
            hand_path=hand_path,
            q_batch=q_batch_np,
            squeeze_enabled=squeezing_enabled,
        )
        q_init_np = np.asarray(commands.q_init, dtype=np.float32)
        q_target_np = np.asarray(commands.q_target, dtype=np.float32)
        runtime = _create_runtime(
            model=bundle.model,
            device=device,
            num_worlds=len(chunk),
            nconmax_per_world=nconmax_per_world,
            njmax_per_world=njmax_per_world,
            capture_cuda_graph=capture_cuda_graph,
        )
        q_target_batch = torch.as_tensor(q_target_np, dtype=torch.float32, device=runtime.torch_device)
        _set_initial_state(
            runtime,
            bundle,
            torch.as_tensor(q_init_np, dtype=torch.float32, device=runtime.torch_device),
            object_pose6,
        )

        base = _rollout_static_batch(
            runtime,
            bundle,
            q_batch=q_target_batch,
            object_pose6=np.asarray(object_pose6, dtype=np.float64),
            num_steps=num_steps,
            steps_per_frame=steps_per_frame,
            in_hand_distance_threshold=in_hand_distance_threshold,
            hand_kp=hand_kp,
            hand_kd=hand_kd,
            hand_max_force=hand_max_force,
            active_mask=None,
            disturbance_force_world=None,
            disturbance_push_steps=0,
        )

        disturbance_payloads: list[dict[str, Any] | None] = [None for _ in chunk]
        if disturbance_enabled:
            saved_state = _snapshot_runtime_state(runtime)
            force_magnitude = disturbance_force_magnitude(
                bundle.model,
                object_body_id=bundle.meta.object_body_id,
                force_acceleration=disturbance_force_acceleration,
                force_cap=disturbance_force_cap,
            )
            base_stable_mask = torch.as_tensor(~base["unstable"], dtype=torch.bool, device=runtime.torch_device)
            per_sample_direction_results = [[] for _ in chunk]
            for direction_name, direction_vector in disturbance_directions:
                _restore_runtime_state(runtime, saved_state)
                force_world = force_magnitude * np.asarray(direction_vector, dtype=np.float32)
                force_batch = torch.zeros((len(chunk), 3), dtype=torch.float32, device=runtime.torch_device)
                force_batch[base_stable_mask] = torch.as_tensor(force_world, dtype=torch.float32, device=runtime.torch_device)
                direction = _rollout_static_batch(
                    runtime,
                    bundle,
                    q_batch=q_target_batch,
                    object_pose6=np.asarray(object_pose6, dtype=np.float64),
                    num_steps=int(disturbance_push_steps) + int(disturbance_recovery_steps),
                    steps_per_frame=steps_per_frame,
                    in_hand_distance_threshold=in_hand_distance_threshold,
                    hand_kp=hand_kp,
                    hand_kd=hand_kd,
                    hand_max_force=hand_max_force,
                    active_mask=base_stable_mask,
                    disturbance_force_world=force_batch,
                    disturbance_push_steps=disturbance_push_steps,
                )
                for idx, stable in enumerate(base["unstable"]):
                    if stable:
                        per_sample_direction_results[idx].append(
                            {
                                "direction": str(direction_name),
                                "force_world": force_world.astype(np.float64),
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
                        )
                        continue
                    per_sample_direction_results[idx].append(
                        {
                            "direction": str(direction_name),
                            "force_world": force_world.astype(np.float64),
                            "force_magnitude": float(np.linalg.norm(force_world)),
                            "num_steps": int(direction["num_steps"][idx]),
                            "unstable": bool(direction["unstable"][idx]),
                            "unstable_step": None
                            if int(direction["unstable_step"][idx]) < 0
                            else int(direction["unstable_step"][idx]),
                            "summary": {
                                key: (
                                    bool(direction["summary"][key][idx])
                                    if key in {"final_contact", "final_in_hand"}
                                    else float(direction["summary"][key][idx])
                                )
                                for key in direction["summary"].keys()
                            },
                        }
                    )

            for idx in range(len(chunk)):
                direction_results = per_sample_direction_results[idx]
                disturbance_payloads[idx] = {
                    "enabled": True,
                    "force_acceleration": float(disturbance_force_acceleration),
                    "force_cap": float(disturbance_force_cap),
                    "force_magnitude": float(force_magnitude),
                    "push_steps": int(disturbance_push_steps),
                    "recovery_steps": int(disturbance_recovery_steps),
                    "settle_steps": int(base["num_steps"][idx]),
                    "all_directions_in_hand": all(
                        (not bool(direction["unstable"])) and bool(direction["summary"]["final_in_hand"])
                        for direction in direction_results
                    ),
                    "directions": direction_results,
                }

        for idx, sample in enumerate(chunk):
            record = {
                "index": int(sample["index"]),
                "robot_name": str(sample["robot_name"]),
                "object_name": str(sample["object_name"]),
                "source": str(sample["source"]),
                "dof": int(sample["dof"]),
                "ordered_idx": int(sample["ordered_idx"]),
                "error": None,
                "error_type": None,
                "summary": {
                    key: (
                        bool(base["summary"][key][idx])
                        if key in {"final_contact", "final_in_hand"}
                        else float(base["summary"][key][idx])
                    )
                    for key in base["summary"].keys()
                },
                "unstable": bool(base["unstable"][idx]),
                "unstable_step": None if int(base["unstable_step"][idx]) < 0 else int(base["unstable_step"][idx]),
                "disturbance": disturbance_payloads[idx],
                "meta": dict(sample.get("meta", {})),
            }
            records.append(record)
    return records
