from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def applied_action(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """Returns the raw action that is actually applied after any action delay."""
    if action_name is None:
        active_terms = env.action_manager.active_terms
        if len(active_terms) != 1:
            raise ValueError(
                "action_name must be specified when multiple action terms are active. "
                f"got active_terms={active_terms}"
            )
        action_name = active_terms[0]

    term = env.action_manager.get_term(action_name)
    if hasattr(term, "applied_raw_actions"):
        return term.applied_raw_actions
    return env.action_manager.action
