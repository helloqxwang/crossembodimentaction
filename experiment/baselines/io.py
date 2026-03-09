from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch

from networks.cvae_baselines import CrossEmbodimentActionCVAE, SingleEmbodimentActionCVAE


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def get_robot_asset_paths(dro_root: str, robot_name: str, object_name: str) -> tuple[str, str]:
    urdf_meta_path = os.path.join(dro_root, "data/data_urdf/robot/urdf_assets_meta.json")
    urdf_meta = json.load(open(urdf_meta_path, "r", encoding="utf-8"))
    hand_path = os.path.join(dro_root, urdf_meta["urdf_path"][robot_name])

    dataset_name, mesh_name = object_name.split("+")
    object_mesh_path = os.path.join(
        dro_root,
        "data/data_urdf/object",
        dataset_name,
        mesh_name,
        f"{mesh_name}.stl",
    )
    return hand_path, object_mesh_path


def load_dataset_meta(dataset_meta_path: str) -> Dict[str, Any]:
    data = torch.load(dataset_meta_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"dataset_meta must be a dict, got: {type(data)}")

    robot_names = data.get("robot_names")
    if robot_names is None:
        raise KeyError(f"Missing 'robot_names' in dataset_meta: {dataset_meta_path}")

    baseline_type = str(data.get("baseline_type", "cross_embodiment"))
    trained_robot_names = list(data.get("trained_robot_names", robot_names))
    all_robot_names = list(data.get("all_robot_names", robot_names))
    robot_dofs = {str(k): int(v) for k, v in data.get("robot_dofs", {}).items()}
    if not robot_dofs:
        raise KeyError(f"Missing 'robot_dofs' in dataset_meta: {dataset_meta_path}")

    normalized = dict(data)
    normalized["baseline_type"] = baseline_type
    normalized["trained_robot_names"] = trained_robot_names
    normalized["all_robot_names"] = all_robot_names
    normalized["robot_names"] = list(robot_names)
    normalized["robot_dofs"] = robot_dofs
    normalized["action_dim"] = int(data["action_dim"])

    if baseline_type == "cross_embodiment":
        robot_to_idx = data.get("robot_to_idx")
        if robot_to_idx is None:
            raise KeyError(f"Missing 'robot_to_idx' in dataset_meta: {dataset_meta_path}")
        normalized["robot_to_idx"] = {str(k): int(v) for k, v in robot_to_idx.items()}
        normalized["conditioning_mode"] = str(data.get("conditioning_mode", "embodiment"))
    else:
        normalized["robot_name"] = str(data.get("robot_name", trained_robot_names[0]))
        normalized["conditioning_mode"] = str(data.get("conditioning_mode", "none"))

    normalized["model_kwargs"] = dict(data.get("model_kwargs", {}))
    return normalized


def load_ckpt(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint must be a dict: {ckpt_path}")
    return state


def _candidate_model_kwargs_from_config(
    cfg_arch: Any,
    *,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_type = str(meta["baseline_type"])
    kwargs: Dict[str, Any] = {
        "action_dim": int(meta["action_dim"]),
        "object_emb_dim": int(cfg_arch.object_emb_dim),
        "latent_dim": int(cfg_arch.latent_dim),
        "encoder_hidden_dims": tuple(int(x) for x in cfg_arch.encoder_hidden_dims),
        "decoder_hidden_dims": tuple(int(x) for x in cfg_arch.decoder_hidden_dims),
        "dgcnn_k": int(cfg_arch.dgcnn_k),
    }
    if baseline_type == "cross_embodiment":
        kwargs["num_embodiments"] = len(meta["all_robot_names"])
        kwargs["embodiment_dim"] = int(cfg_arch.embodiment_dim)
    return kwargs


def _candidate_model_kwargs_from_meta(meta: Dict[str, Any]) -> Dict[str, Any] | None:
    model_kwargs = meta.get("model_kwargs")
    if not isinstance(model_kwargs, dict) or not model_kwargs:
        return None
    return dict(model_kwargs)


def _candidate_model_kwargs_from_ckpt(
    ckpt: Dict[str, Any],
    *,
    meta: Dict[str, Any],
) -> Dict[str, Any] | None:
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        return None
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return None

    kwargs: Dict[str, Any] = {
        "action_dim": int(meta["action_dim"]),
        "object_emb_dim": int(model_cfg["object_emb_dim"]),
        "latent_dim": int(model_cfg["latent_dim"]),
        "encoder_hidden_dims": tuple(int(x) for x in model_cfg["encoder_hidden_dims"]),
        "decoder_hidden_dims": tuple(int(x) for x in model_cfg["decoder_hidden_dims"]),
        "dgcnn_k": int(model_cfg["dgcnn_k"]),
    }
    if str(meta["baseline_type"]) == "cross_embodiment":
        kwargs["num_embodiments"] = len(meta["all_robot_names"])
        kwargs["embodiment_dim"] = int(model_cfg["embodiment_dim"])
    return kwargs


def _build_model(baseline_type: str, kwargs: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    if baseline_type == "cross_embodiment":
        return CrossEmbodimentActionCVAE(**kwargs).to(device)
    if baseline_type == "single_embodiment":
        return SingleEmbodimentActionCVAE(**kwargs).to(device)
    raise ValueError(f"Unsupported baseline_type: {baseline_type}")


def create_model_from_ckpt(
    *,
    ckpt: Dict[str, Any],
    cfg_arch: Any,
    meta: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any], str]:
    state_dict = ckpt.get("model", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no valid model state_dict.")

    baseline_type = str(meta["baseline_type"])
    candidates: list[tuple[str, Dict[str, Any]]] = []
    meta_candidate = _candidate_model_kwargs_from_meta(meta)
    if meta_candidate is not None:
        candidates.append(("dataset_meta", meta_candidate))
    candidates.append(("config_file", _candidate_model_kwargs_from_config(cfg_arch, meta=meta)))
    ckpt_candidate = _candidate_model_kwargs_from_ckpt(ckpt, meta=meta)
    if ckpt_candidate is not None:
        candidates.append(("checkpoint", ckpt_candidate))

    errors: list[str] = []
    for source, kwargs in candidates:
        model = _build_model(baseline_type, kwargs, device)
        try:
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model, kwargs, source
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{source}] {exc}")

    raise RuntimeError(
        "Failed to build/load model from checkpoint. Tried candidates:\n" + "\n".join(errors)
    )
