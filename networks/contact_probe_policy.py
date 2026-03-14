from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from networks.contact_denoiser import ContactConditionalDenoiser
from networks.contact_pointnet import MaskedMultiStagePointNetEncoder
from networks.mlp import MLP
from networks.simple_diffusion import SimpleDDIMScheduler


# Probe policy: same contact-policy denoiser stack, but conditioned only on contact observations.
class ContactDiffusionProbePolicy(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        robot_names: tuple[str, ...] | list[str] | None = None,
        action_normalizer_scale: torch.Tensor | None = None,
        action_normalizer_offset: torch.Tensor | None = None,
        contact_encoder_cfg: Dict[str, int],
        obs_hidden_dims: tuple[int, ...] = (256,),
        global_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        num_inference_steps: int = 10,
        noise_scheduler: SimpleDDIMScheduler | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.robot_names = tuple(str(x) for x in (robot_names or []))

        # Encode the contact cloud once into a global conditioning feature for diffusion.
        self.contact_encoder = MaskedMultiStagePointNetEncoder(**contact_encoder_cfg)
        self.obs_mlp = MLP(
            input_dim=int(contact_encoder_cfg["out_channels"]),
            output_dim=int(global_cond_dim),
            hidden_dims=tuple(int(x) for x in obs_hidden_dims),
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

        # Keep the same per-robot affine action normalization interface as the main policy.
        num_robots = int(action_normalizer_scale.shape[0]) if action_normalizer_scale is not None else max(1, len(self.robot_names))
        if action_normalizer_scale is None:
            action_normalizer_scale = torch.ones((num_robots, self.action_dim), dtype=torch.float32)
        if action_normalizer_offset is None:
            action_normalizer_offset = torch.zeros((num_robots, self.action_dim), dtype=torch.float32)
        action_normalizer_scale = torch.as_tensor(action_normalizer_scale, dtype=torch.float32)
        action_normalizer_offset = torch.as_tensor(action_normalizer_offset, dtype=torch.float32)
        if action_normalizer_scale.shape != action_normalizer_offset.shape:
            raise ValueError(
                "action normalizer scale/offset shape mismatch: "
                f"{tuple(action_normalizer_scale.shape)} vs {tuple(action_normalizer_offset.shape)}"
            )
        if action_normalizer_scale.ndim != 2 or int(action_normalizer_scale.shape[1]) != self.action_dim:
            raise ValueError(
                f"action normalizer must have shape (num_robots, {self.action_dim}), got {tuple(action_normalizer_scale.shape)}"
            )
        if len(self.robot_names) not in (0, int(action_normalizer_scale.shape[0])):
            raise ValueError(
                f"robot_names length {len(self.robot_names)} does not match normalizer rows {action_normalizer_scale.shape[0]}"
            )
        self.register_buffer("action_normalizer_scale", action_normalizer_scale)
        self.register_buffer("action_normalizer_offset", action_normalizer_offset)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # Contact-only observation encoding used by both training and inference.
    def encode_obs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        contact_cloud = batch["contact_cloud"].to(self.device, dtype=torch.float32)
        contact_valid_mask = batch["contact_valid_mask"].to(self.device).bool()
        contact_feat = self.contact_encoder(contact_cloud, point_mask=contact_valid_mask)
        return self.obs_mlp(contact_feat)

    # Valid-dim masking lets the probe reuse the full ShadowHand action space while supervising only a subset.
    def _build_action_valid_mask(self, batch: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        if "action_valid_mask" in batch:
            action_valid_mask = batch["action_valid_mask"].to(self.device).bool()
            if (
                action_valid_mask.ndim == 2
                and int(action_valid_mask.shape[0]) == int(batch_size)
                and int(action_valid_mask.shape[1]) == self.action_dim
            ):
                return action_valid_mask
            raise ValueError(
                f"action_valid_mask must have shape (B, {self.action_dim}), got {tuple(action_valid_mask.shape)}"
            )
        if "local_dof" in batch:
            local_dof = batch["local_dof"].to(self.device, dtype=torch.long).view(-1)
            if int(local_dof.shape[0]) != int(batch_size):
                raise ValueError(f"local_dof batch mismatch: expected {batch_size}, got {tuple(local_dof.shape)}")
            dim_idx = torch.arange(self.action_dim, device=self.device, dtype=torch.long).unsqueeze(0)
            return dim_idx < local_dof.unsqueeze(1)
        return torch.ones((int(batch_size), self.action_dim), dtype=torch.bool, device=self.device)

    # Robot indices select the matching normalization row; the probe typically uses a single robot row.
    def _build_robot_index(self, batch: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        if "robot_index" in batch:
            robot_index = batch["robot_index"].to(self.device, dtype=torch.long).view(-1)
            if int(robot_index.shape[0]) != int(batch_size):
                raise ValueError(f"robot_index batch mismatch: expected {batch_size}, got {tuple(robot_index.shape)}")
        elif int(self.action_normalizer_scale.shape[0]) == 1:
            robot_index = torch.zeros((int(batch_size),), dtype=torch.long, device=self.device)
        else:
            raise ValueError("robot_index is required for multi-robot action normalization")
        if bool((robot_index < 0).any()) or bool((robot_index >= int(self.action_normalizer_scale.shape[0])).any()):
            raise ValueError(
                f"robot_index must be in [0, {int(self.action_normalizer_scale.shape[0]) - 1}], "
                f"got min={int(robot_index.min().item())} max={int(robot_index.max().item())}"
            )
        return robot_index

    def _get_action_normalizer_params(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_index = self._build_robot_index(batch, batch_size=batch_size)
        scale = self.action_normalizer_scale.index_select(0, robot_index)
        offset = self.action_normalizer_offset.index_select(0, robot_index)
        return scale, offset

    # Shared helpers for masking and affine normalization keep train/inference behavior identical.
    @staticmethod
    def _apply_action_valid_mask(sample: torch.Tensor, action_valid_mask: torch.Tensor) -> torch.Tensor:
        if sample.ndim == 2:
            return sample * action_valid_mask.to(dtype=sample.dtype)
        if sample.ndim == 3:
            return sample * action_valid_mask.unsqueeze(1).to(dtype=sample.dtype)
        raise ValueError(f"Unsupported sample ndim={sample.ndim}")

    def _normalize_action(
        self,
        action: torch.Tensor,
        *,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim == 2:
            return action * scale.to(dtype=action.dtype) + offset.to(dtype=action.dtype)
        if action.ndim == 3:
            scale3 = scale.unsqueeze(1).to(dtype=action.dtype)
            offset3 = offset.unsqueeze(1).to(dtype=action.dtype)
            return action * scale3 + offset3
        raise ValueError(f"Unsupported action ndim={action.ndim}")

    def _unnormalize_action(
        self,
        action: torch.Tensor,
        *,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim == 2:
            return (action - offset.to(dtype=action.dtype)) / scale.to(dtype=action.dtype)
        if action.ndim == 3:
            scale3 = scale.unsqueeze(1).to(dtype=action.dtype)
            offset3 = offset.unsqueeze(1).to(dtype=action.dtype)
            return (action - offset3) / scale3
        raise ValueError(f"Unsupported action ndim={action.ndim}")

    # Diffusion training predicts the masked target action under the current scheduler parameterization.
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, float]] | tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        action = batch["action"].to(self.device, dtype=torch.float32).unsqueeze(1)
        batch_size = int(action.shape[0])
        action_valid_mask = self._build_action_valid_mask(batch, batch_size=batch_size)
        norm_scale, norm_offset = self._get_action_normalizer_params(batch, batch_size=batch_size)
        action = self._apply_action_valid_mask(action, action_valid_mask)
        action = self._normalize_action(action, scale=norm_scale, offset=norm_offset)
        action = self._apply_action_valid_mask(action, action_valid_mask)
        global_cond = self.encode_obs(batch)

        noise = torch.randn_like(action)
        noise = self._apply_action_valid_mask(noise, action_valid_mask)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        noisy_action = self._apply_action_valid_mask(noisy_action, action_valid_mask)
        pred = self.model(
            sample=noisy_action,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
        )

        prediction_type = str(self.noise_scheduler.config.prediction_type)
        if prediction_type == "sample":
            target = action
            pred_x0 = pred
        elif prediction_type == "epsilon":
            target = noise
            self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
            alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].view(batch_size, 1, 1)
            pred_x0 = (noisy_action - torch.sqrt(1.0 - alpha_t) * pred) / torch.sqrt(alpha_t).clamp_min(1e-8)
        elif prediction_type == "v_prediction":
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t = self.noise_scheduler.alpha_t[timesteps].view(batch_size, 1, 1)
            sigma_t = self.noise_scheduler.sigma_t[timesteps].view(batch_size, 1, 1)
            target = alpha_t * noise - sigma_t * action
            pred_x0 = alpha_t * noisy_action - sigma_t * pred
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        mask3 = action_valid_mask.unsqueeze(1).to(dtype=pred.dtype)
        sq_error = (pred - target).pow(2) * mask3
        loss_per_sample = sq_error.sum(dim=(1, 2)) / mask3.sum(dim=(1, 2)).clamp_min(1.0)
        loss = loss_per_sample.mean()

        pred_x0 = self._apply_action_valid_mask(pred_x0, action_valid_mask)
        pred_action = self._unnormalize_action(pred_x0, scale=norm_scale, offset=norm_offset)
        pred_action = self._apply_action_valid_mask(pred_action, action_valid_mask)
        loss_dict = {"diffusion_loss": float(loss.item())}
        if return_aux:
            return loss, loss_dict, {
                "pred_action": pred_action[:, 0, :],
                "action_valid_mask": action_valid_mask,
            }
        return loss, loss_dict

    # Inference starts from masked Gaussian noise and iteratively denoises into the supervised action subspace.
    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_cond = self.encode_obs(batch)
        batch_size = int(global_cond.shape[0])
        action_valid_mask = self._build_action_valid_mask(batch, batch_size=batch_size)
        norm_scale, norm_offset = self._get_action_normalizer_params(batch, batch_size=batch_size)
        action = torch.randn(
            (batch_size, 1, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        action = self._apply_action_valid_mask(action, action_valid_mask)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.model(
                sample=action,
                timestep=timestep,
                local_cond=None,
                global_cond=global_cond,
            )
            action = self.noise_scheduler.step(model_output, timestep, action).prev_sample
            action = self._apply_action_valid_mask(action, action_valid_mask)
        action = self._unnormalize_action(action, scale=norm_scale, offset=norm_offset)
        action = action[:, 0, :]
        action = self._apply_action_valid_mask(action, action_valid_mask)
        return {
            "action": action,
            "action_pred": action,
        }

    # Forward is just single-step action prediction for the probe policy.
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action(batch)
