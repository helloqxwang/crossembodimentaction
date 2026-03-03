from __future__ import annotations

import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import viser
from omegaconf import OmegaConf

from data_process.nfk_dataset import NFKBoxFaceDataset
from robot_model.robot_model import create_robot_model
from train_nfk import _prepare_device, build_model


def _load_robot_names(list_file: Path, list_key: str) -> List[str]:
    with open(list_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if list_key not in payload:
        raise KeyError(f"Key '{list_key}' not in {list_file}. Available: {list(payload.keys())}")
    names = payload[list_key]
    if not isinstance(names, list) or not names:
        raise ValueError(f"Empty or invalid list for key '{list_key}' in {list_file}")
    return list(names)


def _load_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> int | None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    epoch = state.get("epoch") if isinstance(state, dict) else None
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict) and "models" in state:
        state_dict = state["models"]
    else:
        state_dict = state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return epoch


def _generate_one_sample(
    model,
    seed: int,
) -> Dict[str, torch.Tensor]:
    rng = random.Random(seed)
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(seed)

    q_zero = torch.zeros(model.dof, dtype=torch.float32, device=model.device)
    q_rand = NFKBoxFaceDataset._sample_q(model, generator=generator)

    points_zero, normals_zero = model.get_box_face_points_normals(q=q_zero)
    points_rand, normals_rand = model.get_box_face_points_normals(q=q_rand)

    zero_pairs, valid_zero, chosen_faces = NFKBoxFaceDataset._pick_one_face_per_link(
        points_zero,
        normals_zero,
        generator=generator,
    )
    rand_pairs = torch.zeros_like(zero_pairs)
    valid_rand = torch.zeros_like(valid_zero)

    for i in range(zero_pairs.shape[0]):
        face_id = int(chosen_faces[i].item())
        if face_id < 0:
            continue
        p = points_rand[i, face_id]
        n = normals_rand[i, face_id]
        ok = torch.isfinite(p).all() and torch.isfinite(n).all()
        if ok:
            rand_pairs[i, :3] = p
            rand_pairs[i, 3:] = n
            valid_rand[i] = True

    pair_valid_mask = valid_zero & valid_rand
    contact_mask = NFKBoxFaceDataset._sample_contact_mask(
        pair_valid_mask,
        rng=rng,
        generator=generator,
    )
    active = pair_valid_mask & contact_mask
    return {
        "q_rand": q_rand,
        "zero_masked": zero_pairs[active],
        "rand_masked": rand_pairs[active],
    }


@torch.no_grad()
def _run_inference_sample(
    model_net: torch.nn.Module,
    robot_model,
    seed: int,
    min_points: int = 1,
    max_attempts: int = 8,
) -> Dict[str, np.ndarray]:
    sample = None
    for attempt in range(max_attempts):
        sample = _generate_one_sample(robot_model, seed + attempt)
        if sample["rand_masked"].shape[0] >= min_points:
            break

    q_rand = sample["q_rand"]
    zero_masked = sample["zero_masked"]
    rand_masked = sample["rand_masked"]

    m = zero_masked.shape[0]
    if m == 0:
        pred = torch.zeros((0, 6), dtype=torch.float32)
    else:
        zero_in = zero_masked.unsqueeze(0).to(next(model_net.parameters()).device)
        rand_in = rand_masked.unsqueeze(0).to(next(model_net.parameters()).device)
        mask_m = torch.ones((1, m), dtype=torch.bool, device=zero_in.device)
        outputs = model_net(zero_in, rand_in, mask_m)
        pred = outputs["recon_pairs"][0].detach().cpu()

    mesh = robot_model.get_trimesh_q(q_rand, mode="original")["visual"]
    return {
        "q_rand": q_rand.detach().cpu().numpy().astype(np.float32),
        "gt_pairs": rand_masked.detach().cpu().numpy().astype(np.float32),
        "pred_pairs": pred.detach().cpu().numpy().astype(np.float32),
        "mesh_vertices": mesh.vertices.astype(np.float32),
        "mesh_faces": mesh.faces.astype(np.int32),
    }


def _empty_pc(server: viser.ViserServer, name: str) -> None:
    server.scene.add_point_cloud(
        name=name,
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        point_size=0.004,
        point_shape="circle",
    )


def _empty_lines(server: viser.ViserServer, name: str) -> None:
    server.scene.add_line_segments(
        name=name,
        points=np.zeros((0, 2, 3), dtype=np.float32),
        colors=(1.0, 1.0, 1.0),
        line_width=1.0,
    )


def _empty_mesh(server: viser.ViserServer, name: str) -> None:
    server.scene.add_mesh_simple(
        name=name,
        vertices=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        color=(0.5, 0.7, 1.0),
        opacity=0.0,
    )


def _add_points_and_normals(
    server: viser.ViserServer,
    *,
    name_prefix: str,
    pairs: np.ndarray,
    center: np.ndarray,
    point_color: tuple[float, float, float],
    normal_color: tuple[float, float, float],
    point_size: float,
    normal_scale: float,
) -> None:
    if pairs.shape[0] == 0:
        _empty_pc(server, f"{name_prefix}_pc")
        _empty_lines(server, f"{name_prefix}_normals")
        return

    pts = pairs[:, :3] - center[None, :]
    nrms = pairs[:, 3:]
    nrm_norm = np.linalg.norm(nrms, axis=1, keepdims=True)
    valid = nrm_norm[:, 0] > 1e-8
    if not np.any(valid):
        _empty_pc(server, f"{name_prefix}_pc")
        _empty_lines(server, f"{name_prefix}_normals")
        return

    pts = pts[valid]
    nrms = nrms[valid] / nrm_norm[valid]

    pc_colors = np.zeros((pts.shape[0], 3), dtype=np.float32)
    pc_colors[:, 0] = point_color[0]
    pc_colors[:, 1] = point_color[1]
    pc_colors[:, 2] = point_color[2]

    server.scene.add_point_cloud(
        name=f"{name_prefix}_pc",
        points=pts.astype(np.float32),
        colors=pc_colors,
        point_size=point_size,
        point_shape="circle",
    )

    seg = np.stack([pts, pts + nrms * float(normal_scale)], axis=1).astype(np.float32)
    server.scene.add_line_segments(
        name=f"{name_prefix}_normals",
        points=seg,
        colors=normal_color,
        line_width=1.6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NFK model and visualize predictions via viser.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9412)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--assets-dir", type=str, default="")
    parser.add_argument("--robot-list-file", type=str, default="")
    parser.add_argument("--robot-list-key", type=str, default="")
    parser.add_argument("--normal-scale", type=float, default=0.02)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device_str = args.device if args.device else str(cfg.training.device)
    device = _prepare_device(device_str)
    torch.manual_seed(int(cfg.training.seed))

    assets_dir = Path(args.assets_dir) if args.assets_dir else Path(cfg.data.assets_dir)
    assets_dir = assets_dir.resolve()

    robot_list_file = (
        Path(args.robot_list_file)
        if args.robot_list_file
        else (Path(cfg.data.robot_list_file) if str(cfg.data.robot_list_file) else assets_dir / "manipulator_robot_lists.json")
    )
    if not robot_list_file.is_absolute():
        robot_list_file = (Path.cwd() / robot_list_file).resolve()

    robot_list_key = args.robot_list_key if args.robot_list_key else str(cfg.data.robot_list_key)

    model_net = build_model(cfg, device)
    epoch = _load_ckpt(model_net, Path(args.ckpt), device)
    print(f"Loaded checkpoint: {args.ckpt} (epoch={epoch})")

    robot_names = _load_robot_names(robot_list_file, robot_list_key)
    robot_models = {}
    failed = {}
    dataset_device = str(getattr(cfg.data, "dataset_device", "cpu"))
    dataset_torch_device = torch.device(dataset_device)
    for name in robot_names:
        try:
            rm = create_robot_model(
                robot_name=name,
                device=dataset_torch_device,
                num_points=int(cfg.data.link_num_points),
                assets_dir=assets_dir,
            )
            rm.switch_to_box_links_only_mode(override=False)
            robot_models[name] = rm
        except Exception as exc:
            failed[name] = str(exc)

    robot_names = sorted(robot_models.keys())
    if not robot_names:
        raise RuntimeError(f"No robots loaded. failures={list(failed.items())[:5]}")
    if failed:
        print(f"Skipped {len(failed)} robots due to load errors.")

    splits = ["train", "test"]
    split_seed = {
        "train": int(cfg.training.seed),
        "test": int(cfg.training.seed) + 100000,
    }
    n_samples = int(args.n_samples)
    results: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]] = {s: {} for s in splits}

    print("Running inference...")
    for s in splits:
        for ridx, robot_name in enumerate(robot_names):
            rm = robot_models[robot_name]
            items = []
            for i in range(n_samples):
                seed = split_seed[s] + ridx * 1000 + i
                item = _run_inference_sample(model_net, rm, seed=seed)
                items.append(item)
            results[s][robot_name] = items
    print(f"Inference done. split_count={len(splits)} robots={len(robot_names)} samples_per_robot={n_samples}")

    if args.no_vis:
        return

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("world", show_axes=True, axes_length=0.12, axes_radius=0.004)

    split_slider = server.gui.add_slider("dataset (0=train,1=test)", min=0, max=1, step=1, initial_value=0)
    robot_slider = server.gui.add_slider("robot_index", min=0, max=len(robot_names) - 1, step=1, initial_value=0)
    sample_slider = server.gui.add_slider("sample_index", min=0, max=n_samples - 1, step=1, initial_value=0)
    vis_slider = server.gui.add_slider(
        "vis_mode (0=gt+pred points+normals,1=mode0+gt mesh)",
        min=0,
        max=1,
        step=1,
        initial_value=1,
    )

    def render() -> None:
        split_name = splits[int(split_slider.value)]
        robot_name = robot_names[int(robot_slider.value)]
        sample_idx = int(sample_slider.value)
        vis_mode = int(vis_slider.value)

        record = results[split_name][robot_name][sample_idx]
        gt_pairs = record["gt_pairs"]
        pred_pairs = record["pred_pairs"]
        mesh_vertices = record["mesh_vertices"]
        mesh_faces = record["mesh_faces"]

        if mesh_vertices.shape[0] > 0:
            center = mesh_vertices.mean(axis=0).astype(np.float32)
        elif gt_pairs.shape[0] > 0:
            center = gt_pairs[:, :3].mean(axis=0).astype(np.float32)
        elif pred_pairs.shape[0] > 0:
            center = pred_pairs[:, :3].mean(axis=0).astype(np.float32)
        else:
            center = np.zeros((3,), dtype=np.float32)

        _add_points_and_normals(
            server,
            name_prefix="gt",
            pairs=gt_pairs,
            center=center,
            point_color=(0.22, 0.62, 0.98),
            normal_color=(0.22, 0.62, 0.98),
            point_size=0.008,
            normal_scale=float(args.normal_scale),
        )
        _add_points_and_normals(
            server,
            name_prefix="pred",
            pairs=pred_pairs,
            center=center,
            point_color=(1.00, 0.45, 0.20),
            normal_color=(1.00, 0.45, 0.20),
            point_size=0.008,
            normal_scale=float(args.normal_scale),
        )

        if vis_mode == 1 and mesh_vertices.shape[0] > 0 and mesh_faces.shape[0] > 0:
            server.scene.add_mesh_simple(
                name="robot_gt_mesh",
                vertices=(mesh_vertices - center[None, :]).astype(np.float32),
                faces=mesh_faces.astype(np.int32),
                color=(0.75, 0.82, 1.0),
                opacity=0.35,
            )
        else:
            _empty_mesh(server, "robot_gt_mesh")

    @split_slider.on_update
    def _(_) -> None:
        render()

    @robot_slider.on_update
    def _(_) -> None:
        render()

    @sample_slider.on_update
    def _(_) -> None:
        render()

    @vis_slider.on_update
    def _(_) -> None:
        render()

    render()
    print(f"Viser running on http://{args.host}:{args.port}")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
