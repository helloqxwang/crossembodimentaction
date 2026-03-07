from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.contact_denoiser import ContactConditionalDenoiser
from networks.contact_pointnet import MaskedMultiStagePointNetEncoder
from networks.mlp import MLP
from networks.simple_diffusion import SimpleDDIMScheduler


class ContactDiffusionPolicy(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        p_hat_encoder_cfg: Dict[str, int],
        contact_encoder_cfg: Dict[str, int],
        fusion_hidden_dims: tuple[int, ...] = (256,),
        global_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        num_inference_steps: int = 10,
        noise_scheduler: SimpleDDIMScheduler | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_inference_steps = int(num_inference_steps)

        self.p_hat_encoder = MaskedMultiStagePointNetEncoder(**p_hat_encoder_cfg)
        self.contact_encoder = MaskedMultiStagePointNetEncoder(**contact_encoder_cfg)
        fused_input_dim = int(p_hat_encoder_cfg["out_channels"]) + int(contact_encoder_cfg["out_channels"])
        self.fusion_mlp = MLP(
            input_dim=fused_input_dim,
            output_dim=int(global_cond_dim),
            hidden_dims=tuple(int(x) for x in fusion_hidden_dims),
            activation="gelu",
            dropout=0.0,
            use_layer_norm=False,
        )
        self.model = ContactConditionalDenoiser(
            action_dim=self.action_dim,
            global_cond_dim=int(global_cond_dim),
            time_embed_dim=int(diffusion_step_embed_dim),
            hidden_dims=tuple(int(x) for x in down_dims),
        )
        self.noise_scheduler = noise_scheduler or SimpleDDIMScheduler(
            num_train_timesteps=50,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="sample",
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_obs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        p_hat = batch["p_hat"].to(self.device, dtype=torch.float32)
        contact_cloud = batch["contact_cloud"].to(self.device, dtype=torch.float32)
        contact_valid_mask = batch["contact_valid_mask"].to(self.device).bool()
        p_hat_mask = torch.ones(p_hat.shape[:2], dtype=torch.bool, device=self.device)

        p_hat_feat = self.p_hat_encoder(p_hat, point_mask=p_hat_mask)
        contact_feat = self.contact_encoder(contact_cloud, point_mask=contact_valid_mask)
        fused = torch.cat([p_hat_feat, contact_feat], dim=-1)
        return self.fusion_mlp(fused)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        action = batch["action"].to(self.device, dtype=torch.float32).unsqueeze(1)
        global_cond = self.encode_obs(batch)

        noise = torch.randn_like(action)
        batch_size = int(action.shape[0])
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        pred = self.model(
            sample=noisy_action,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
        )

        prediction_type = str(self.noise_scheduler.config.prediction_type)
        if prediction_type == "sample":
            target = action
        elif prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t = self.noise_scheduler.alpha_t[timesteps].view(batch_size, 1, 1)
            sigma_t = self.noise_scheduler.sigma_t[timesteps].view(batch_size, 1, 1)
            target = alpha_t * noise - sigma_t * action
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        loss = F.mse_loss(pred, target)
        return loss, {"diffusion_loss": float(loss.item())}

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_cond = self.encode_obs(batch)
        action = torch.randn(
            (global_cond.shape[0], 1, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.model(
                sample=action,
                timestep=timestep,
                local_cond=None,
                global_cond=global_cond,
            )
            action = self.noise_scheduler.step(model_output, timestep, action).prev_sample
        action = action[:, 0, :]
        return {
            "action": action,
            "action_pred": action,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action(batch)
