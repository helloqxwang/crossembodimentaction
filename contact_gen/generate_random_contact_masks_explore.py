from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import hydra
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.append(PROJECT_ROOT)

from contact_gen.sampler_explore import (  # noqa: E402
    RealMaskStats,
    build_real_mask_stats,
    sample_anchor_indices,
    sample_direct_bernoulli_mask,
    sample_direct_surface_mask,
    sample_target_count,
    swap_jitter_mask,
)


def _abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    base = hydra.utils.get_original_cwd()
    return os.path.abspath(os.path.join(base, path))


def _to_int_list(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    return [int(x)]


def _load_real_masks_by_robot(real_masks_path: str, num_points_by_robot: Dict[str, int]) -> Dict[str, torch.Tensor]:
    payload = torch.load(real_masks_path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Invalid real masks payload: {real_masks_path}")
    out: Dict[str, List[torch.Tensor]] = {}
    for s in list(payload["samples"]):
        rn = str(s.get("robot_name", ""))
        if rn not in num_points_by_robot:
            continue
        m = torch.as_tensor(s["hand_contact_mask"]).bool().view(-1)
        if int(m.numel()) != int(num_points_by_robot[rn]):
            continue
        out.setdefault(rn, []).append(m)
    stacked: Dict[str, torch.Tensor] = {}
    for rn, arr in out.items():
        if len(arr) > 0:
            stacked[rn] = torch.stack(arr, dim=0).bool()
    return stacked


def _load_real_q_by_robot(real_masks_path: str, dof_by_robot: Dict[str, int]) -> Dict[str, torch.Tensor]:
    payload = torch.load(real_masks_path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Invalid real masks payload: {real_masks_path}")
    out: Dict[str, List[torch.Tensor]] = {}
    for s in list(payload["samples"]):
        rn = str(s.get("robot_name", ""))
        if rn not in dof_by_robot:
            continue
        q = torch.as_tensor(s.get("q", torch.tensor([]))).float().view(-1)
        if int(q.numel()) != int(dof_by_robot[rn]):
            continue
        out.setdefault(rn, []).append(q)
    stacked: Dict[str, torch.Tensor] = {}
    for rn, arr in out.items():
        if len(arr) > 0:
            stacked[rn] = torch.stack(arr, dim=0).float()
    return stacked


def _sample_component_count(
    comp_lo: int,
    comp_hi: int,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> int:
    if comp_hi <= comp_lo:
        return int(comp_lo)
    return int(torch.randint(int(comp_lo), int(comp_hi) + 1, (1,), device=device, generator=generator).item())


def _resize_mask_to_target_count(mask: torch.Tensor, scores: torch.Tensor, target_count: int) -> torch.Tensor:
    n = int(mask.numel())
    k = int(max(1, min(n, int(target_count))))
    order = torch.argsort(scores, descending=True)
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[order[:k]] = True
    return out


@hydra.main(config_path=".", config_name="config_generate_random_contact_masks_explore", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))

    hparams_path = _abs_path(str(cfg.hparams_path))
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)
    if not isinstance(hparams, dict) or "robots" not in hparams:
        raise ValueError(f"Invalid hparams: {hparams_path}")
    hparams_robots = hparams["robots"]

    robot_names_cfg = list(cfg.robot_names) if cfg.robot_names is not None else []
    robot_names = robot_names_cfg if len(robot_names_cfg) > 0 else sorted(hparams_robots.keys())

    num_points_by_robot: Dict[str, int] = {}
    model_name_by_robot: Dict[str, str] = {}
    comp_range_by_robot: Dict[str, List[int]] = {}
    for rn in robot_names:
        if rn not in hparams_robots:
            continue
        rcfg = hparams_robots[rn]
        num_points_by_robot[rn] = int(rcfg["num_surface_points"])
        model_name_by_robot[rn] = str(rcfg["robot_model_name"])
        cr = _to_int_list(rcfg.get("component_range", [1, 4]))
        if len(cr) < 2:
            cr = [1, 4]
        comp_range_by_robot[rn] = [int(cr[0]), int(cr[1])]

    real_masks_by_robot: Dict[str, torch.Tensor] = {}
    real_q_by_robot: Dict[str, torch.Tensor] = {}
    q_source = str(cfg.q_source)
    need_real = (
        (str(cfg.anchor_source) == "real_distribution")
        or (str(cfg.target_count_source) == "real")
        or (q_source in {"real", "mixed"})
    )
    if need_real:
        real_path = _abs_path(str(cfg.real_masks_path))
        real_masks_by_robot = _load_real_masks_by_robot(real_masks_path=real_path, num_points_by_robot=num_points_by_robot)
        if q_source in {"real", "mixed"}:
            # Build dof map from temporary model instances.
            tmp_dof: Dict[str, int] = {}
            from robot_model.robot_model import create_robot_model  # type: ignore
            for rn in robot_names:
                if rn not in hparams_robots:
                    continue
                rcfg = hparams_robots[rn]
                mtmp = create_robot_model(
                    robot_name=str(rcfg["robot_model_name"]),
                    device=torch.device(str(cfg.device)),
                    num_points=int(rcfg["num_surface_points"]),
                )
                tmp_dof[rn] = int(mtmp.dof)
            real_q_by_robot = _load_real_q_by_robot(real_masks_path=real_path, dof_by_robot=tmp_dof)

    output_dir = _abs_path(str(cfg.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    from robot_model.robot_model import create_robot_model  # type: ignore

    for robot_name in robot_names:
        if robot_name not in hparams_robots:
            print(f"[skip] {robot_name}: missing in hparams")
            continue

        rcfg = hparams_robots[robot_name]
        resolved_robot_name = str(rcfg["robot_model_name"])
        num_surface_points = int(rcfg["num_surface_points"])
        expected_template_hash = str(rcfg["surface_template_hash"])
        fit_comp = comp_range_by_robot[robot_name]
        comp_lo_fit, comp_hi_fit = int(fit_comp[0]), int(fit_comp[1])
        comp_lo = comp_lo_fit if int(cfg.component_min) <= 0 else int(cfg.component_min)
        comp_hi = comp_hi_fit if int(cfg.component_max) <= 0 else int(cfg.component_max)
        comp_lo = max(1, min(comp_lo, 16))
        comp_hi = max(comp_lo, min(comp_hi, 16))

        model = create_robot_model(
            robot_name=resolved_robot_name,
            device=torch.device(str(cfg.device)),
            num_points=num_surface_points,
        )
        h = model.get_surface_template_hash()
        if bool(cfg.check_template_hash) and h != expected_template_hash:
            raise RuntimeError(
                f"{robot_name}: surface template hash mismatch ({h} != {expected_template_hash})"
            )

        real_stats: Optional[RealMaskStats] = None
        if robot_name in real_masks_by_robot:
            real_stats = build_real_mask_stats(real_masks_by_robot[robot_name])
        if str(cfg.anchor_source) == "real_distribution" and real_stats is None:
            raise RuntimeError(f"{robot_name}: anchor_source=real_distribution but no real masks found")
        if str(cfg.target_count_source) == "real" and real_stats is None:
            raise RuntimeError(f"{robot_name}: target_count_source=real but no real masks found")
        if q_source in {"real", "mixed"} and robot_name not in real_q_by_robot:
            raise RuntimeError(f"{robot_name}: q_source={q_source} but no real q found")

        generator = torch.Generator(device=model.device.type)
        generator.manual_seed(int(cfg.seed) + hash(robot_name) % 1000000)

        sampled_masks_list: List[torch.Tensor] = []
        sampled_q_list: List[torch.Tensor] = []
        vis_samples: List[Dict[str, torch.Tensor]] = []
        remain = int(cfg.sample_count)
        chunk = int(max(1, int(cfg.sample_chunk)))
        vis_budget = int(max(0, int(cfg.save_visualization_samples)))
        pbar = tqdm(total=remain, desc=f"sample_explore:{robot_name}")

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
                if q_source == "real":
                    q_pool = real_q_by_robot[robot_name].to(model.device)
                    qi = int(torch.randint(0, int(q_pool.shape[0]), (1,), device=model.device, generator=generator).item())
                    q = model._zero_base_translation_in_q(q_pool[qi].clone())
                elif q_source == "mixed":
                    use_real_q = bool(
                        float(torch.rand(1, device=model.device, generator=generator).item())
                        < float(max(0.0, min(1.0, float(cfg.q_real_mix_prob))))
                    )
                    if use_real_q:
                        q_pool = real_q_by_robot[robot_name].to(model.device)
                        qi = int(torch.randint(0, int(q_pool.shape[0]), (1,), device=model.device, generator=generator).item())
                        q = model._zero_base_translation_in_q(q_pool[qi].clone())
                    else:
                        q = model._sample_q_for_contact(generator=generator)
                else:
                    q = model._sample_q_for_contact(generator=generator)
                hand_points, hand_normals = model.get_surface_points_normals(q=q)
                n_anchor = _sample_component_count(comp_lo=comp_lo, comp_hi=comp_hi, generator=generator, device=model.device)
                anchor_idx = sample_anchor_indices(
                    n_anchor=n_anchor,
                    num_points=num_surface_points,
                    anchor_source=str(cfg.anchor_source),
                    generator=generator,
                    real_point_prob=(None if real_stats is None else real_stats.point_prob),
                    temperature=float(cfg.anchor_temperature),
                    uniform_mix=float(cfg.anchor_uniform_mix),
                    device=model.device,
                )
                target_count = sample_target_count(
                    target_count_source=str(cfg.target_count_source),
                    count_values_real=(None if real_stats is None else real_stats.count_values),
                    num_points=num_surface_points,
                    generator=generator,
                    device=model.device,
                    fallback_min=int(cfg.direct_target_count_min),
                    fallback_max=int(cfg.direct_target_count_max),
                    jitter_frac=float(cfg.target_count_jitter_frac),
                )

                if str(cfg.sampler_family) == "patch_distance":
                    use_tail = bool(
                        float(torch.rand(1, device=model.device, generator=generator).item())
                        < float(max(0.0, min(1.0, float(cfg.patch_tail_mix_prob))))
                    )

                    if use_tail:
                        s_lo = float(cfg.patch_tail_anchor_shift_min)
                        s_hi = float(cfg.patch_tail_anchor_shift_max)
                        e_lo = float(cfg.patch_tail_extent_min)
                        e_hi = float(cfg.patch_tail_extent_max)
                        ps = float(cfg.patch_tail_shift_power)
                        pe = float(cfg.patch_tail_extent_power)
                        ppa_min = int(cfg.patch_tail_points_per_anchor_min)
                        ppa_max = int(cfg.patch_tail_points_per_anchor_max)
                        pc_tail = float(cfg.patch_tail_penetration_clearance)

                        if s_lo < 0.0:
                            s_lo = float(cfg.patch_anchor_shift_min)
                        if s_hi < 0.0:
                            s_hi = float(cfg.patch_anchor_shift_max)
                        if e_lo < 0.0:
                            e_lo = float(cfg.patch_extent_min)
                        if e_hi < 0.0:
                            e_hi = float(cfg.patch_extent_max)
                        if ps <= 0.0:
                            ps = float(cfg.patch_shift_power)
                        if pe <= 0.0:
                            pe = float(cfg.patch_extent_power)
                        if ppa_min <= 0:
                            ppa_min = int(cfg.patch_points_per_anchor_min)
                        if ppa_max <= 0:
                            ppa_max = int(cfg.patch_points_per_anchor_max)
                        if pc_tail < 0.0:
                            pc_tail = float(cfg.patch_penetration_clearance)
                        penetration_clearance_i = pc_tail
                    else:
                        s_lo = float(cfg.patch_anchor_shift_min)
                        s_hi = float(cfg.patch_anchor_shift_max)
                        e_lo = float(cfg.patch_extent_min)
                        e_hi = float(cfg.patch_extent_max)
                        ps = float(cfg.patch_shift_power)
                        pe = float(cfg.patch_extent_power)
                        ppa_min = int(cfg.patch_points_per_anchor_min)
                        ppa_max = int(cfg.patch_points_per_anchor_max)
                        penetration_clearance_i = float(cfg.patch_penetration_clearance)

                    us = float(torch.rand(1, device=model.device, generator=generator).item())
                    ue = float(torch.rand(1, device=model.device, generator=generator).item())
                    ps = float(max(1e-3, ps))
                    pe = float(max(1e-3, pe))
                    us = us ** ps
                    ue = ue ** pe
                    anchor_shift = float(s_lo + us * max(1e-8, s_hi - s_lo))
                    max_extent = float(e_lo + ue * max(1e-8, e_hi - e_lo))
                    ppa = int(
                        torch.randint(
                            int(min(ppa_min, ppa_max)),
                            int(max(ppa_min, ppa_max)) + 1,
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
                        penetration_clearance=float(penetration_clearance_i),
                    )
                    val = model._compute_gendex_contact_value_source_target(
                        source_points=hand_points,
                        source_normals=hand_normals,
                        target_points=obj_pts,
                        align_exp_scale=float(cfg.align_exp_scale),
                        sigmoid_scale=float(cfg.sigmoid_scale),
                    )
                    mask = val >= float(cfg.threshold)
                    rank_score = val
                    ns = float(max(0.0, float(cfg.patch_score_noise_std)))
                    if ns > 0.0:
                        noise = torch.randn(val.shape, device=model.device, generator=generator) * ns
                        rank_score = rank_score + noise
                    if str(cfg.target_count_source) == "real":
                        exact_prob = float(max(0.0, min(1.0, float(cfg.patch_exact_count_prob))))
                        if bool(torch.rand(1, device=model.device, generator=generator).item() < exact_prob):
                            mask = _resize_mask_to_target_count(mask=mask, scores=rank_score, target_count=target_count)
                        else:
                            clamp_r = float(max(0.0, float(cfg.patch_count_clamp_ratio)))
                            k_lo = int(max(1, round((1.0 - clamp_r) * float(target_count))))
                            k_hi = int(min(num_surface_points, round((1.0 + clamp_r) * float(target_count))))
                            cur_k = int(mask.sum().item())
                            if cur_k < k_lo:
                                mask = _resize_mask_to_target_count(mask=mask, scores=rank_score, target_count=k_lo)
                            elif cur_k > k_hi:
                                mask = _resize_mask_to_target_count(mask=mask, scores=rank_score, target_count=k_hi)
                    if bool(torch.rand(1, device=model.device, generator=generator).item() < float(cfg.patch_mask_swap_prob)):
                        mask = swap_jitter_mask(
                            mask=mask,
                            neighbors=model.surface_graph_neighbors,
                            steps=int(cfg.patch_mask_swap_steps),
                            generator=generator,
                            device=model.device,
                            global_jump_prob=float(cfg.patch_mask_swap_global_jump_prob),
                        )

                    values_b[i] = rank_score
                    if int(obj_pts.shape[0]) > int(cfg.patch_points):
                        keep = torch.randperm(int(obj_pts.shape[0]), device=model.device, generator=generator)[: int(cfg.patch_points)]
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
                elif str(cfg.sampler_family) == "direct_surface":
                    if str(cfg.direct_strategy) == "components":
                        mask = sample_direct_surface_mask(
                            neighbors=model.surface_graph_neighbors,
                            num_points=num_surface_points,
                            anchor_indices=anchor_idx,
                            target_count=target_count,
                            generator=generator,
                            device=model.device,
                            compact_prob=float(cfg.direct_compact_prob),
                            filament_prob=float(cfg.direct_filament_prob),
                        )
                    elif str(cfg.direct_strategy) == "bernoulli_smooth":
                        mask = sample_direct_bernoulli_mask(
                            num_points=num_surface_points,
                            target_count=target_count,
                            generator=generator,
                            device=model.device,
                            base_prob=(None if real_stats is None else real_stats.point_prob),
                            neighbors=model.surface_graph_neighbors,
                            noise_scale=float(cfg.direct_noise_scale),
                            smooth_steps=int(cfg.direct_smooth_steps),
                            smooth_add_thr=float(cfg.direct_smooth_add_thr),
                            smooth_keep_thr=float(cfg.direct_smooth_keep_thr),
                        )
                    else:
                        raise ValueError(f"Unknown direct_strategy: {cfg.direct_strategy}")
                    values_b[i] = mask.float()
                    na = int(min(16, int(anchor_idx.numel())))
                    anchors_b[i, :na] = anchor_idx[:na]
                else:
                    raise ValueError(f"Unknown sampler_family: {cfg.sampler_family}")

                masks_b[i] = mask
                q_b[i] = q
                hand_points_b[i] = hand_points

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
                "sample_count": int(cfg.sample_count),
                "sampler_family": str(cfg.sampler_family),
                "anchor_source": str(cfg.anchor_source),
                "target_count_source": str(cfg.target_count_source),
                "q_source": str(cfg.q_source),
                "q_real_mix_prob": float(cfg.q_real_mix_prob),
                "component_range_fit": [int(comp_lo_fit), int(comp_hi_fit)],
                "component_range_used": [int(comp_lo), int(comp_hi)],
                "surface_template_hash": h,
                "hparams_path": hparams_path,
                "threshold": float(cfg.threshold),
                "align_exp_scale": float(cfg.align_exp_scale),
                "sigmoid_scale": float(cfg.sigmoid_scale),
                "patch_tail_mix_prob": float(cfg.patch_tail_mix_prob),
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
