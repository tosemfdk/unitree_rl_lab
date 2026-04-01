from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize rigid-body CoM offsets by adding sampled xyz values."""
    asset: Articulation = env.scene[asset_cfg.name]
    # CoM APIs require CPU tensors for ids and assignments.
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    range_list = [com_range.get(axis, (0.0, 0.0)) for axis in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    coms = asset.root_physx_view.get_coms().clone()
    coms[:, body_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


def apply_external_force_torque_xyz(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    force_range: dict[str, tuple[float, float]] | None = None,
    torque_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply persistent external wrench with independent xyz sampling per axis.

    Unlike Isaac Lab's default ``apply_external_force_torque``, this helper allows
    specifying different ranges for x/y/z force and torque components.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.body_ids == slice(None):
        num_bodies = asset.num_bodies
    else:
        num_bodies = len(asset_cfg.body_ids)

    def _sample(range_cfg: dict[str, tuple[float, float]] | None) -> torch.Tensor:
        range_cfg = {} if range_cfg is None else range_cfg
        range_list = [range_cfg.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")]
        ranges = torch.tensor(range_list, dtype=torch.float32, device=asset.device)
        return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), num_bodies, 3), device=asset.device)

    forces = _sample(force_range)
    torques = _sample(torque_range)
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)
