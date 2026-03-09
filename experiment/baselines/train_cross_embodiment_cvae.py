from __future__ import annotations

import os
import sys

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from experiment.baselines.common import (  # noqa: E402
    create_loader,
    ensure_dir,
    finish_wandb,
    init_wandb,
    prepare_device,
    run_epoch,
    save_ckpt,
    set_seed,
)
from experiment.baselines.datasets import CVAEActionDataset, infer_robot_dofs  # noqa: E402
from networks.cvae_baselines import CrossEmbodimentActionCVAE  # noqa: E402


@hydra.main(version_base="1.2", config_path="../conf/baselines", config_name="train_cross_embodiment_cvae")
def main(cfg: DictConfig) -> None:
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    set_seed(int(cfg.seed))
    device = prepare_device(str(cfg.training.device))

    dro_root = to_absolute_path(str(cfg.dataset.dro_root))
    train_robot_names = [str(x) for x in cfg.dataset.train_robot_names]
    all_robot_names = [str(x) for x in cfg.dataset.all_robot_names]
    missing = sorted(set(train_robot_names) - set(all_robot_names))
    if missing:
        raise ValueError(f"train_robot_names must be a subset of all_robot_names, missing: {missing}")

    robot_to_idx = {name: i for i, name in enumerate(all_robot_names)}
    robot_dofs = infer_robot_dofs(dro_root, all_robot_names)
    action_dim = max(robot_dofs.values())

    train_dataset = CVAEActionDataset(
        dro_root=dro_root,
        robot_names=train_robot_names,
        split="train",
        num_points=int(cfg.dataset.num_points),
        action_dim=action_dim,
        robot_to_idx=robot_to_idx,
        include_embodiment_idx=True,
    )
    validate_dataset = CVAEActionDataset(
        dro_root=dro_root,
        robot_names=train_robot_names,
        split="validate",
        num_points=int(cfg.dataset.num_points),
        action_dim=action_dim,
        robot_to_idx=robot_to_idx,
        include_embodiment_idx=True,
    )

    train_loader = create_loader(
        train_dataset,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=True,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
        drop_last=bool(cfg.dataset.drop_last),
    )
    validate_loader = create_loader(
        validate_dataset,
        batch_size=int(cfg.dataset.val_batch_size),
        shuffle=False,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
        drop_last=False,
    )

    model_kwargs = {
        "action_dim": action_dim,
        "num_embodiments": len(all_robot_names),
        "embodiment_dim": int(cfg.model.embodiment_dim),
        "object_emb_dim": int(cfg.model.object_emb_dim),
        "latent_dim": int(cfg.model.latent_dim),
        "encoder_hidden_dims": tuple(int(x) for x in cfg.model.encoder_hidden_dims),
        "decoder_hidden_dims": tuple(int(x) for x in cfg.model.decoder_hidden_dims),
        "dgcnn_k": int(cfg.model.dgcnn_k),
    }
    model = CrossEmbodimentActionCVAE(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else str(cfg.experiment.name)
    init_wandb(cfg, run_name=run_name)

    save_dir = to_absolute_path(str(cfg.training.save_dir))
    ensure_dir(save_dir)
    torch.save(
        {
            "baseline_type": "cross_embodiment",
            "conditioning_mode": "embodiment",
            "trained_robot_names": train_robot_names,
            "all_robot_names": all_robot_names,
            "robot_names": all_robot_names,
            "robot_to_idx": robot_to_idx,
            "robot_dofs": robot_dofs,
            "action_dim": action_dim,
            "model_kwargs": model_kwargs,
        },
        os.path.join(save_dir, "dataset_meta.pt"),
    )

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            recon_weight=float(cfg.training.recon_weight),
            kl_weight=float(cfg.training.kl_weight),
            grad_clip_norm=float(cfg.training.grad_clip_norm),
            log_every=int(cfg.training.log_every),
            global_step=global_step,
            is_train=True,
            max_batches=int(cfg.training.max_train_batches),
        )
        val_metrics, global_step = run_epoch(
            model=model,
            loader=validate_loader,
            device=device,
            optimizer=None,
            recon_weight=float(cfg.training.recon_weight),
            kl_weight=float(cfg.training.kl_weight),
            grad_clip_norm=None,
            log_every=int(cfg.training.log_every),
            global_step=global_step,
            is_train=False,
            max_batches=int(cfg.training.max_val_batches),
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_epoch": train_metrics["loss"],
                "train/recon_mse_epoch": train_metrics["recon_mse"],
                "train/recon_mse_valid_epoch": train_metrics["recon_mse_valid"],
                "train/kl_epoch": train_metrics["kl"],
                "validate/loss_epoch": val_metrics["loss"],
                "validate/recon_mse_epoch": val_metrics["recon_mse"],
                "validate/recon_mse_valid_epoch": val_metrics["recon_mse_valid"],
                "validate/kl_epoch": val_metrics["kl"],
            },
            step=global_step,
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_recon_valid={val_metrics['recon_mse_valid']:.6f}"
        )

        save_ckpt(
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            path=os.path.join(save_dir, "last.pt"),
        )
        if epoch % int(cfg.training.save_every_n_epoch) == 0:
            save_ckpt(
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                path=os.path.join(save_dir, f"epoch_{epoch}.pt"),
            )

        if val_metrics["recon_mse_valid"] < best_val:
            best_val = val_metrics["recon_mse_valid"]
            save_ckpt(
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                path=os.path.join(save_dir, "best.pt"),
            )

    finish_wandb()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    main()
