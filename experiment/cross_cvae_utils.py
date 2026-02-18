from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch

from networks.cross_cvae_mvp import CrossEmbodimentActionCVAE


def load_dataset_meta(dataset_meta_path: str) -> Dict[str, Any]:
    data = torch.load(dataset_meta_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"dataset_meta must be a dict, got: {type(data)}")
    required = ("robot_names", "robot_to_idx", "robot_dofs", "action_dim")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing '{key}' in dataset_meta: {dataset_meta_path}")
    return data


def load_ckpt(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint must be a dict: {ckpt_path}")
    return state


def _candidate_model_kwargs_from_config(
    cfg_model: Any,
    *,
    action_dim: int,
    num_embodiments: int,
) -> Dict[str, Any]:
    return {
        "action_dim": int(action_dim),
        "num_embodiments": int(num_embodiments),
        "embodiment_dim": int(cfg_model.embodiment_dim),
        "object_emb_dim": int(cfg_model.object_emb_dim),
        "latent_dim": int(cfg_model.latent_dim),
        "encoder_hidden_dims": tuple(int(x) for x in cfg_model.encoder_hidden_dims),
        "decoder_hidden_dims": tuple(int(x) for x in cfg_model.decoder_hidden_dims),
        "dgcnn_k": int(cfg_model.dgcnn_k),
    }


def _candidate_model_kwargs_from_ckpt(
    ckpt: Dict[str, Any],
    *,
    action_dim: int,
    num_embodiments: int,
) -> Dict[str, Any] | None:
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        return None
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return None
    required = (
        "embodiment_dim",
        "object_emb_dim",
        "latent_dim",
        "encoder_hidden_dims",
        "decoder_hidden_dims",
        "dgcnn_k",
    )
    if any(k not in model_cfg for k in required):
        return None
    return {
        "action_dim": int(action_dim),
        "num_embodiments": int(num_embodiments),
        "embodiment_dim": int(model_cfg["embodiment_dim"]),
        "object_emb_dim": int(model_cfg["object_emb_dim"]),
        "latent_dim": int(model_cfg["latent_dim"]),
        "encoder_hidden_dims": tuple(int(x) for x in model_cfg["encoder_hidden_dims"]),
        "decoder_hidden_dims": tuple(int(x) for x in model_cfg["decoder_hidden_dims"]),
        "dgcnn_k": int(model_cfg["dgcnn_k"]),
    }


def create_model_from_ckpt(
    *,
    ckpt: Dict[str, Any],
    cfg_model: Any,
    action_dim: int,
    num_embodiments: int,
    device: torch.device,
) -> tuple[CrossEmbodimentActionCVAE, Dict[str, Any], str]:
    state_dict = ckpt.get("model", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no valid model state_dict.")

    candidates: list[tuple[str, Dict[str, Any]]] = []
    candidates.append(
        (
            "config_file",
            _candidate_model_kwargs_from_config(
                cfg_model,
                action_dim=action_dim,
                num_embodiments=num_embodiments,
            ),
        )
    )
    ckpt_candidate = _candidate_model_kwargs_from_ckpt(
        ckpt,
        action_dim=action_dim,
        num_embodiments=num_embodiments,
    )
    if ckpt_candidate is not None:
        candidates.append(("checkpoint", ckpt_candidate))

    errors: list[str] = []
    for source, kwargs in candidates:
        model = CrossEmbodimentActionCVAE(**kwargs).to(device)
        try:
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model, kwargs, source
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[{source}] {exc}")

    raise RuntimeError(
        "Failed to build/load model from checkpoint. Tried candidates:\n"
        + "\n".join(errors)
    )


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


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
