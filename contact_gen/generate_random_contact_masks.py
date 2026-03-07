from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from typing import Any

import hydra
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)


def _abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    base = hydra.utils.get_original_cwd()
    return os.path.abspath(os.path.join(base, path))


def _stable_seed_from_name(name: str) -> int:
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False) % (2**31 - 1)


def _to_int_list(x: Any) -> list[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    return [int(x)]


def _compute_sample_count_map(robot_names: list[str], sample_count: int, mode: str) -> dict[str, int]:
    n = len(robot_names)
    if n <= 0:
        return {}
    if mode == "total":
        base = int(sample_count) // n
        rem = int(sample_count) % n
        return {name: int(base + (1 if i < rem else 0)) for i, name in enumerate(robot_names)}
    if mode == "per_robot":
        return {name: int(sample_count) for name in robot_names}
    raise ValueError(f"Invalid sample_count_mode: {mode}. Expected 'per_robot' or 'total'.")


def _load_real_masks_by_robot(
    real_masks_path: str,
    num_points_by_robot: dict[str, int],
) -> dict[str, torch.Tensor]:
    payload = torch.load(real_masks_path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Invalid real mask payload: {real_masks_path}")
    out: dict[str, list[torch.Tensor]] = {}
    for s in list(payload["samples"]):
        rn = str(s.get("robot_name", ""))
        if rn not in num_points_by_robot:
            continue
        m = torch.as_tensor(s["hand_contact_mask"]).bool().view(-1)
        if int(m.numel()) != int(num_points_by_robot[rn]):
            continue
        out.setdefault(rn, []).append(m)
    stacked: dict[str, torch.Tensor] = {}
    for rn, arr in out.items():
        if len(arr) > 0:
            stacked[rn] = torch.stack(arr, dim=0).bool()
    return stacked


def _extract_real_count_values(real_masks: torch.Tensor) -> torch.Tensor:
    masks = real_masks.detach().bool().cpu()
    return masks.sum(dim=1).long()


def _build_point_prob_from_real_masks(real_masks: torch.Tensor) -> torch.Tensor:
    masks = real_masks.detach().bool().cpu()
    point_prob = masks.float().mean(dim=0).clamp_min(1e-8)
    return point_prob / point_prob.sum().clamp_min(1e-8)


def _sample_anchor_indices(
    n_anchor: int,
    num_points: int,
    point_prob: torch.Tensor | None,
    anchor_sampling_mode: str,
    generator: torch.Generator | None,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    n = int(max(1, min(int(n_anchor), int(num_points))))
    mode = str(anchor_sampling_mode).lower().strip()
    if mode == "uniform":
        return torch.randperm(int(num_points), device=device, generator=generator)[:n]
    if mode != "point_prob":
        raise ValueError(f"Invalid anchor_sampling_mode={anchor_sampling_mode}. Use 'point_prob' or 'uniform'.")
    if point_prob is None:
        raise ValueError("point_prob is required when anchor_sampling_mode='point_prob'.")

    p = point_prob.to(device=device, dtype=torch.float32).clamp_min(1e-8)
    t = float(max(1e-3, float(temperature)))
    if abs(t - 1.0) > 1e-6:
        p = p.pow(1.0 / t)
    p = p / p.sum().clamp_min(1e-8)
    return torch.multinomial(p, num_samples=n, replacement=False, generator=generator)


def _sample_target_count(
    count_values: torch.Tensor,
    num_points: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    if int(count_values.numel()) == 0:
        raise ValueError("count_values is empty.")
    idx = int(
        torch.randint(
            0,
            int(count_values.numel()),
            (1,),
            device=device,
            generator=generator,
        ).item()
    )
    return int(max(1, min(int(num_points), int(count_values[idx].item()))))


def _resize_mask_to_target_count(mask: torch.Tensor, scores: torch.Tensor, target_count: int) -> torch.Tensor:
    n = int(mask.numel())
    k = int(max(1, min(n, int(target_count))))
    order = torch.argsort(scores, descending=True)
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[order[:k]] = True
    return out


@hydra.main(config_path=".", config_name="config_generate_random_contact_masks", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))

    hparams_path = _abs_path(str(cfg.hparams_path))
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hparams yaml not found: {hparams_path}")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)
    if not isinstance(hparams, dict) or "robots" not in hparams:
        raise ValueError(f"Invalid hparams yaml: {hparams_path}")
    hparams_robots = hparams["robots"]
    if not isinstance(hparams_robots, dict) or len(hparams_robots) == 0:
        raise RuntimeError(f"No robot hyperparameters found in: {hparams_path}")

    robot_names_cfg = list(cfg.robot_names) if cfg.robot_names is not None else []
    robot_names = robot_names_cfg if len(robot_names_cfg) > 0 else sorted(hparams_robots.keys())
    sample_count_mode = str(getattr(cfg, "sample_count_mode", "per_robot"))
    sample_count_map = _compute_sample_count_map(
        robot_names=robot_names,
        sample_count=int(cfg.sample_count),
        mode=sample_count_mode,
    )

    total_jobs = int(sum(sample_count_map.values()))
    print(
        f"[sampling] mode={sample_count_mode} requested={int(cfg.sample_count)} "
        f"robots={len(robot_names)} total_masks_to_generate={total_jobs}"
    )
    for r in robot_names:
        print(f"[sampling] {r}: {sample_count_map[r]} masks")

    num_points_by_robot: dict[str, int] = {}
    for rn in robot_names:
        if rn in hparams_robots:
            num_points_by_robot[rn] = int(hparams_robots[rn]["num_surface_points"])

    real_masks_path = _abs_path(str(cfg.real_masks_path))
    real_masks_by_robot = _load_real_masks_by_robot(
        real_masks_path=real_masks_path,
        num_points_by_robot=num_points_by_robot,
    )

    output_dir = _abs_path(str(cfg.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    sys.path.append(PROJECT_ROOT)
    from robot_model.robot_model import create_robot_model  # type: ignore

    for robot_name in robot_names:
        if robot_name not in hparams_robots:
            print(f"[skip] {robot_name}: not found in {hparams_path}")
            continue
        if robot_name not in real_masks_by_robot:
            raise RuntimeError(f"{robot_name}: no real masks found in {real_masks_path}")

        rcfg = hparams_robots[robot_name]
        resolved_robot_name = str(rcfg["robot_model_name"])
        num_surface_points = int(rcfg["num_surface_points"])
        expected_template_hash = str(rcfg["surface_template_hash"])
        fit_component_range = _to_int_list(rcfg.get("component_range", [1, 4]))
        if len(fit_component_range) < 2:
            fit_component_range = [1, 4]
        comp_lo_fit, comp_hi_fit = int(fit_component_range[0]), int(fit_component_range[1])

        model = create_robot_model(
            robot_name=resolved_robot_name,
            device=torch.device(str(cfg.device)),
            num_points=num_surface_points,
        )
        model_template_hash = model.get_surface_template_hash()
        if bool(cfg.check_template_hash) and model_template_hash != expected_template_hash:
            raise RuntimeError(
                f"{robot_name}: template hash mismatch ({model_template_hash} != {expected_template_hash}). "
                "Regenerate hyperparameters/extracted contacts with current code."
            )

        comp_lo = comp_lo_fit if int(cfg.component_min) <= 0 else int(cfg.component_min)
        comp_hi = comp_hi_fit if int(cfg.component_max) <= 0 else int(cfg.component_max)
        comp_lo = max(1, min(comp_lo, 16))
        comp_hi = max(comp_lo, min(comp_hi, 16))

        anchor_sampling_mode = str(getattr(cfg, "anchor_sampling_mode", "point_prob")).strip().lower()
        if anchor_sampling_mode not in {"point_prob", "uniform"}:
            raise ValueError(
                f"Invalid anchor_sampling_mode={anchor_sampling_mode}. Use 'point_prob' or 'uniform'."
            )
        count_values = _extract_real_count_values(real_masks_by_robot[robot_name])
        point_prob = (
            _build_point_prob_from_real_masks(real_masks_by_robot[robot_name])
            if anchor_sampling_mode == "point_prob"
            else None
        )
        print(
            f"[{robot_name}] model={resolved_robot_name} num_surface_points={num_surface_points} "
            f"component_range_fit={[comp_lo_fit, comp_hi_fit]} component_range_used={[comp_lo, comp_hi]} "
            f"real_masks={int(real_masks_by_robot[robot_name].shape[0])} "
            f"anchor_sampling_mode={anchor_sampling_mode}"
        )

        generator = torch.Generator(device=model.device.type)
        generator.manual_seed(int(cfg.seed) + _stable_seed_from_name(robot_name))

        sampled_masks_list: list[torch.Tensor] = []
        sampled_q_list: list[torch.Tensor] = []
        vis_samples: list[dict[str, torch.Tensor | int | str]] = []
        remain = int(sample_count_map[robot_name])
        chunk = int(max(1, int(cfg.sample_chunk)))
        vis_budget = int(max(0, int(cfg.save_visualization_samples)))
        pbar = tqdm(total=remain, desc=f"sample_masks:{robot_name}")

        while remain > 0:
            b = min(chunk, remain)
            masks_b = torch.zeros((b, num_surface_points), dtype=torch.bool, device=model.device)
            q_b = torch.zeros((b, model.dof), dtype=torch.float32, device=model.device)
            hand_points_b = torch.zeros((b, num_surface_points, 3), dtype=torch.float32, device=model.device)
            obj_points_b = torch.zeros((b, int(cfg.patch_points), 3), dtype=torch.float32, device=model.device)
            obj_mask_b = torch.zeros((b, int(cfg.patch_points)), dtype=torch.bool, device=model.device)
            values_b = torch.zeros((b, num_surface_points), dtype=torch.float32, device=model.device)
            anchors_b = torch.full((b, 16), -1, dtype=torch.long, device=model.device)
            patch_types_b = torch.full((b, 16), -1, dtype=torch.long, device=model.device)

            for i in range(b):
                q = model._sample_q_for_contact(generator=generator)
                hand_points, hand_normals = model.get_surface_points_normals(q=q)

                n_anchor = int(
                    torch.randint(
                        int(comp_lo),
                        int(comp_hi) + 1,
                        (1,),
                        device=model.device,
                        generator=generator,
                    ).item()
                )
                anchor_idx = _sample_anchor_indices(
                    n_anchor=n_anchor,
                    num_points=num_surface_points,
                    point_prob=point_prob,
                    anchor_sampling_mode=anchor_sampling_mode,
                    generator=generator,
                    temperature=float(cfg.anchor_temperature),
                    device=model.device,
                )
                target_count = _sample_target_count(
                    count_values=count_values.to(model.device),
                    num_points=num_surface_points,
                    generator=generator,
                    device=model.device,
                )

                us = float(torch.rand(1, device=model.device, generator=generator).item())
                ue = float(torch.rand(1, device=model.device, generator=generator).item())
                ps = float(max(1e-3, float(cfg.patch_shift_power)))
                pe = float(max(1e-3, float(cfg.patch_extent_power)))
                us = us ** ps
                ue = ue ** pe
                s_lo = float(cfg.patch_anchor_shift_min)
                s_hi = float(cfg.patch_anchor_shift_max)
                e_lo = float(cfg.patch_extent_min)
                e_hi = float(cfg.patch_extent_max)
                anchor_shift = float(s_lo + us * max(1e-8, s_hi - s_lo))
                max_extent = float(e_lo + ue * max(1e-8, e_hi - e_lo))

                ppa = int(
                    torch.randint(
                        int(cfg.patch_points_per_anchor_min),
                        int(cfg.patch_points_per_anchor_max) + 1,
                        (1,),
                        device=model.device,
                        generator=generator,
                    ).item()
                )
                patch_points_gen = int(max(int(cfg.patch_points), int(ppa * n_anchor)))

                obj_pts, patch_meta = model._sample_virtual_object_patches(
                    hand_points=hand_points,
                    hand_normals=hand_normals,
                    num_anchors=n_anchor,
                    total_patch_points=patch_points_gen,
                    generator=generator,
                    anchor_indices=anchor_idx,
                    anchor_shift=anchor_shift,
                    max_plane_extent=max_extent,
                    penetration_clearance=float(cfg.patch_penetration_clearance),
                )
                val = model._compute_gendex_contact_value_source_target(
                    source_points=hand_points,
                    source_normals=hand_normals,
                    target_points=obj_pts,
                    align_exp_scale=float(cfg.align_exp_scale),
                    sigmoid_scale=float(cfg.sigmoid_scale),
                )
                mask = val >= float(cfg.threshold)

                if bool(
                    torch.rand(1, device=model.device, generator=generator).item()
                    < float(max(0.0, min(1.0, float(cfg.patch_exact_count_prob))))
                ):
                    mask = _resize_mask_to_target_count(mask=mask, scores=val, target_count=target_count)
                else:
                    clamp_r = float(max(0.0, float(cfg.patch_count_clamp_ratio)))
                    k_lo = int(max(1, round((1.0 - clamp_r) * float(target_count))))
                    k_hi = int(min(num_surface_points, round((1.0 + clamp_r) * float(target_count))))
                    cur_k = int(mask.sum().item())
                    if cur_k < k_lo:
                        mask = _resize_mask_to_target_count(mask=mask, scores=val, target_count=k_lo)
                    elif cur_k > k_hi:
                        mask = _resize_mask_to_target_count(mask=mask, scores=val, target_count=k_hi)

                masks_b[i] = mask
                values_b[i] = val
                q_b[i] = q
                hand_points_b[i] = hand_points

                if int(obj_pts.shape[0]) > int(cfg.patch_points):
                    keep = torch.randperm(int(obj_pts.shape[0]), device=model.device, generator=generator)[
                        : int(cfg.patch_points)
                    ]
                    obj_vis = obj_pts[keep]
                else:
                    obj_vis = obj_pts
                n_obj = int(min(int(cfg.patch_points), int(obj_vis.shape[0])))
                if n_obj > 0:
                    obj_points_b[i, :n_obj] = obj_vis[:n_obj]
                    obj_mask_b[i, :n_obj] = True
                na = int(min(16, int(patch_meta["anchor_indices"].shape[0])))
                anchors_b[i, :na] = patch_meta["anchor_indices"][:na]
                patch_types_b[i, :na] = patch_meta["patch_types"][:na]

            sampled_masks_list.append(masks_b.detach().cpu())
            sampled_q_list.append(q_b.detach().cpu())

            if vis_budget > 0:
                take = min(vis_budget, b)
                for i in range(take):
                    m = masks_b[i].detach().cpu().bool()
                    hp = hand_points_b[i].detach().cpu().float()
                    op = obj_points_b[i].detach().cpu().float()
                    om = obj_mask_b[i].detach().cpu().bool()
                    vis_samples.append(
                        {
                            "sample_id": len(vis_samples),
                            "robot_name": robot_name,
                            "robot_model_name": resolved_robot_name,
                            "q": q_b[i].detach().cpu().float(),
                            "hand_points": hp,
                            "hand_contact_mask": m,
                            "hand_contact_points": hp[m],
                            "object_patch_points": op[om],
                            "contact_values": values_b[i].detach().cpu().float(),
                            "anchor_indices": anchors_b[i].detach().cpu().long(),
                            "patch_types": patch_types_b[i].detach().cpu().long(),
                        }
                    )
                vis_budget -= take

            remain -= b
            pbar.update(b)
        pbar.close()

        sampled_masks_t = torch.cat(sampled_masks_list, dim=0).bool()
        sampled_q_t = torch.cat(sampled_q_list, dim=0).float()
        sampled_counts = sampled_masks_t.sum(dim=1).long()

        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "robot_name": robot_name,
                "robot_model_name": resolved_robot_name,
                "num_surface_points": int(num_surface_points),
                "sample_count_requested": int(cfg.sample_count),
                "sample_count_mode": sample_count_mode,
                "sample_count_effective": int(sample_count_map[robot_name]),
                "component_range_fit": [int(comp_lo_fit), int(comp_hi_fit)],
                "component_range_used": [int(comp_lo), int(comp_hi)],
                "surface_template_hash": model_template_hash,
                "hparams_path": hparams_path,
                "real_masks_path": real_masks_path,
                "threshold": float(cfg.threshold),
                "align_exp_scale": float(cfg.align_exp_scale),
                "sigmoid_scale": float(cfg.sigmoid_scale),
                "anchor_temperature": float(cfg.anchor_temperature),
                "anchor_sampling_mode": anchor_sampling_mode,
                "sampling_mode": "real_stats_guided_patch_distance",
            },
            "summary": {
                "sampled_num_samples": int(sampled_masks_t.shape[0]),
                "sampled_count_mean": float(sampled_counts.float().mean().item()),
                "sampled_count_std": float(sampled_counts.float().std(unbiased=False).item()),
            },
            "sampled_masks": sampled_masks_t,
            "sampled_q": sampled_q_t,
            "visualization_samples": vis_samples,
        }
        out_path = os.path.join(output_dir, f"{robot_name}_random_masks.pt")
        torch.save(payload, out_path)
        print(f"[saved] {out_path}")
        print(f"[summary] {robot_name}: {payload['summary']}")


if __name__ == "__main__":
    main()
