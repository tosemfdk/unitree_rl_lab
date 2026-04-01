from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def disturbance_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    push_term_name: str = "push_robot",
    wrench_term_name: str = "base_wrench_disturbance",
    start_step: int = 12_000,
    num_steps: int = 48_000,
    push_range_start: float = 0.4,
    push_range_end: float = 1.1,
    push_yaw_start: float = 0.1,
    push_yaw_end: float = 0.35,
    push_interval_start: tuple[float, float] = (6.0, 9.0),
    push_interval_end: tuple[float, float] = (4.0, 6.0),
    wrench_force_xy_start: float = 8.0,
    wrench_force_xy_end: float = 28.0,
    wrench_force_z_low_start: float = -10.0,
    wrench_force_z_low_end: float = -45.0,
    wrench_force_z_high_start: float = 0.0,
    wrench_force_z_high_end: float = 10.0,
    wrench_torque_xy_start: float = 1.0,
    wrench_torque_xy_end: float = 3.5,
    wrench_torque_z_start: float = 0.5,
    wrench_torque_z_end: float = 2.0,
) -> torch.Tensor:
    """Ramp push and persistent base-wrench disturbances during training."""

    step = int(env.common_step_counter)
    if num_steps <= 0:
        alpha = 1.0 if step >= start_step else 0.0
    else:
        alpha = (step - start_step) / float(num_steps)
        alpha = float(max(0.0, min(1.0, alpha)))

    def _lerp(a: float, b: float) -> float:
        return float(a + (b - a) * alpha)

    event_manager = getattr(env, "event_manager", None)
    if event_manager is None:
        return torch.tensor(alpha, device=env.device)

    push_cfg = event_manager.get_term_cfg(push_term_name)
    push_mag = _lerp(push_range_start, push_range_end)
    push_yaw = _lerp(push_yaw_start, push_yaw_end)
    push_cfg.interval_range_s = (_lerp(push_interval_start[0], push_interval_end[0]), _lerp(push_interval_start[1], push_interval_end[1]))
    push_cfg.params["velocity_range"] = {
        "x": (-push_mag, push_mag),
        "y": (-push_mag, push_mag),
        "yaw": (-push_yaw, push_yaw),
    }
    event_manager.set_term_cfg(push_term_name, push_cfg)

    wrench_cfg = event_manager.get_term_cfg(wrench_term_name)
    wrench_xy = _lerp(wrench_force_xy_start, wrench_force_xy_end)
    torque_xy = _lerp(wrench_torque_xy_start, wrench_torque_xy_end)
    torque_z = _lerp(wrench_torque_z_start, wrench_torque_z_end)
    wrench_cfg.params["force_range"] = {
        "x": (-wrench_xy, wrench_xy),
        "y": (-wrench_xy, wrench_xy),
        "z": (
            _lerp(wrench_force_z_low_start, wrench_force_z_low_end),
            _lerp(wrench_force_z_high_start, wrench_force_z_high_end),
        ),
    }
    wrench_cfg.params["torque_range"] = {
        "x": (-torque_xy, torque_xy),
        "y": (-torque_xy, torque_xy),
        "z": (-torque_z, torque_z),
    }
    event_manager.set_term_cfg(wrench_term_name, wrench_cfg)

    if hasattr(env.cfg, "events"):
        if hasattr(env.cfg.events, push_term_name):
            cfg_term = getattr(env.cfg.events, push_term_name)
            cfg_term.interval_range_s = push_cfg.interval_range_s
            cfg_term.params["velocity_range"] = push_cfg.params["velocity_range"]
        if hasattr(env.cfg.events, wrench_term_name):
            cfg_term = getattr(env.cfg.events, wrench_term_name)
            cfg_term.params["force_range"] = wrench_cfg.params["force_range"]
            cfg_term.params["torque_range"] = wrench_cfg.params["torque_range"]

    return torch.tensor(alpha, device=env.device)
