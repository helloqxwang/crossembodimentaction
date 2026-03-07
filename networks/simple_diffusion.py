from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class SchedulerStepOutput:
    prev_sample: torch.Tensor


def _betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(time_step: float) -> float:
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class SimpleDDIMScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = False,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "sample",
    ) -> None:
        self.config = SimpleNamespace(
            num_train_timesteps=int(num_train_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            beta_schedule=str(beta_schedule),
            clip_sample=bool(clip_sample),
            set_alpha_to_one=bool(set_alpha_to_one),
            steps_offset=int(steps_offset),
            prediction_type=str(prediction_type),
        )
        if self.config.beta_schedule == "squaredcos_cap_v2":
            betas = _betas_for_alpha_bar(self.config.num_train_timesteps)
        elif self.config.beta_schedule == "linear":
            betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.num_train_timesteps,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported beta_schedule: {self.config.beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0, dtype=torch.float32)
            if self.config.set_alpha_to_one
            else self.alphas_cumprod[0]
        )
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1.0 - self.alphas_cumprod)
        self.timesteps = torch.arange(
            self.config.num_train_timesteps - 1,
            -1,
            -1,
            dtype=torch.long,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_prod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)[timesteps]
        expand_shape = (timesteps.shape[0],) + (1,) * (original_samples.ndim - 1)
        sqrt_alpha = torch.sqrt(alpha_prod).view(expand_shape)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_prod).view(expand_shape)
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

    def set_timesteps(self, num_inference_steps: int) -> None:
        if int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.timesteps = torch.linspace(
            self.config.num_train_timesteps - 1,
            0,
            int(num_inference_steps),
        ).round().long()

    def _get_prev_timestep(self, timestep: int) -> int:
        matches = torch.nonzero(self.timesteps == int(timestep), as_tuple=False).view(-1)
        if int(matches.numel()) == 0:
            if len(self.timesteps) <= 1:
                return -1
            step = max(1, self.config.num_train_timesteps // len(self.timesteps))
            return max(int(timestep) - step, -1)
        idx = int(matches[0].item())
        if idx + 1 >= int(self.timesteps.numel()):
            return -1
        return int(self.timesteps[idx + 1].item())

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerStepOutput:
        if torch.is_tensor(timestep):
            timestep = int(timestep.item())

        alpha_prod_t = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)[timestep]
        prev_timestep = self._get_prev_timestep(timestep)
        if prev_timestep >= 0:
            alpha_prod_prev = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)[prev_timestep]
        else:
            alpha_prod_prev = self.final_alpha_cumprod.to(device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1.0 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt().clamp_min(1e-8)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt().clamp_min(1e-8)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        else:
            raise ValueError(f"Unsupported prediction type: {self.config.prediction_type}")

        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        prev_sample = alpha_prod_prev.sqrt() * pred_original_sample + (1.0 - alpha_prod_prev).sqrt() * pred_epsilon
        return SchedulerStepOutput(prev_sample=prev_sample)
