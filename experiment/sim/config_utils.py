from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def experiment_conf_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "conf"


def mujoco_test_config_path() -> Path:
    return experiment_conf_dir() / "config_mujoco_test_cross_cvae.yaml"


def load_mujoco_test_config(path: str | None = None) -> DictConfig:
    config_path = Path(path).expanduser().resolve() if path is not None else mujoco_test_config_path()
    cfg = OmegaConf.load(str(config_path))
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Expected DictConfig from {config_path}")
    return cfg
