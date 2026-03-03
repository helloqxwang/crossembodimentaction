from __future__ import annotations

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


def _to_int_list(x: Any) -> list[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    return [int(x)]


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

    output_dir = _abs_path(str(cfg.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    sys.path.append(PROJECT_ROOT)
    from robot_model.robot_model import create_robot_model  # type: ignore

    for robot_name in robot_names:
        if robot_name not in hparams_robots:
            print(f"[skip] {robot_name}: not found in {hparams_path}")
            continue
        rcfg = hparams_robots[robot_name]
        if not isinstance(rcfg, dict):
            print(f"[skip] {robot_name}: invalid robot config")
            continue

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
        comp_lo = max(1, min(comp_lo, 8))
        comp_hi = max(comp_lo, min(comp_hi, 8))

        print(
            f"[{robot_name}] model={resolved_robot_name} num_surface_points={num_surface_points} "
            f"component_range_fit={[comp_lo_fit, comp_hi_fit]} component_range_used={[comp_lo, comp_hi]}"
        )

        sampled_masks_list = []
        sampled_q_list = []
        vis_samples = []
        remain = int(cfg.sample_count)
        chunk = int(max(1, int(cfg.sample_chunk)))
        vis_budget = int(max(0, int(cfg.save_visualization_samples)))
        pbar = tqdm(total=remain, desc=f"sample_masks:{robot_name}")
        while remain > 0:
            b = min(chunk, remain)
            out = model.sample_contact_masks_with_details(
                B=b,
                threshold=float(cfg.threshold),
                align_exp_scale=float(cfg.align_exp_scale),
                sigmoid_scale=float(cfg.sigmoid_scale),
                patch_points=int(cfg.patch_points),
                component_min=comp_lo,
                component_max=comp_hi,
                seed=None,
            )
            sampled_masks_list.append(out["masks"].bool())
            sampled_q_list.append(out["q"].float())

            if vis_budget > 0:
                take = min(vis_budget, b)
                for i in range(take):
                    m = out["masks"][i].bool()
                    hp = out["hand_points"][i].float()
                    op = out["object_patch_points"][i].float()
                    op_mask = out["object_patch_mask"][i].bool()
                    vis_samples.append(
                        {
                            "sample_id": len(vis_samples),
                            "robot_name": robot_name,
                            "robot_model_name": resolved_robot_name,
                            "q": out["q"][i].float(),
                            "hand_points": hp,
                            "hand_contact_mask": m,
                            "hand_contact_points": hp[m],
                            "object_patch_points": op[op_mask],
                            "contact_values": out["contact_values"][i].float(),
                            "anchor_indices": out["anchor_indices"][i].long(),
                            "patch_types": out["patch_types"][i].long(),
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
                "threshold": float(cfg.threshold),
                "align_exp_scale": float(cfg.align_exp_scale),
                "sigmoid_scale": float(cfg.sigmoid_scale),
                "sample_count": int(cfg.sample_count),
                "component_range_fit": [int(comp_lo_fit), int(comp_hi_fit)],
                "component_range_used": [int(comp_lo), int(comp_hi)],
                "surface_template_hash": model_template_hash,
                "hparams_path": hparams_path,
                "sampling_mode": "dataless_runtime",
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
