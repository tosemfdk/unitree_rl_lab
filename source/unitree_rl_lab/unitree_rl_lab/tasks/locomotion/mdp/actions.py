from __future__ import annotations

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction, JointPositionAction
from isaaclab.utils import configclass


class GravityCompJointPositionAction(JointPositionAction):
    """Joint position action with model-based gravity compensation."""

    cfg: "GravityCompJointPositionActionCfg"

    def _get_generalized_gravity_forces(self) -> torch.Tensor:
        view = self._asset.root_physx_view
        for method_name in (
            "get_gravity_compensation_forces",
            "get_generalized_gravity_forces",
            "get_gravity_forces",
        ):
            method = getattr(view, method_name, None)
            if callable(method):
                forces = method()
                return torch.as_tensor(forces, device=self.device, dtype=self.processed_actions.dtype)
        raise AttributeError("Articulation view does not expose a gravity compensation method.")

    def apply_actions(self):
        super().apply_actions()

        if self.cfg.gravity_comp_scale == 0.0:
            return

        generalized_forces = self._get_generalized_gravity_forces()
        root_dofs = generalized_forces.shape[1] - self._asset.num_joints
        if root_dofs < 0:
            raise RuntimeError(
                "Gravity compensation vector is smaller than the joint dimension. "
                f"got={generalized_forces.shape[1]}, num_joints={self._asset.num_joints}"
            )

        joint_forces = generalized_forces[:, root_dofs:]
        feedforward_effort = joint_forces[:, self._joint_ids] * float(self.cfg.gravity_comp_scale)

        if self.cfg.gravity_comp_max_torque is not None:
            limit = abs(float(self.cfg.gravity_comp_max_torque))
            feedforward_effort = torch.clamp(feedforward_effort, min=-limit, max=limit)

        self._asset.set_joint_effort_target(feedforward_effort, joint_ids=self._joint_ids)


@configclass
class GravityCompJointPositionActionCfg(actions_cfg.JointPositionActionCfg):
    """Configuration for joint position control with gravity compensation."""

    class_type: type = GravityCompJointPositionAction

    gravity_comp_scale: float = 0.0
    gravity_comp_max_torque: float | None = None


class GravityCompPerLegStiffnessAction(JointAction):
    """Joint position action with per-leg variable stiffness and optional gravity compensation.

    Action layout:
    - 0:11  -> joint position actions
    - 12:16 -> per-leg stiffness actions (FL, FR, RL, RR by default)
    """

    cfg: "GravityCompPerLegStiffnessActionCfg"

    def __init__(self, cfg: "GravityCompPerLegStiffnessActionCfg", env):
        super().__init__(cfg, env)

        if self.cfg.kp_max <= self.cfg.kp_min:
            raise ValueError(f"kp_max must be greater than kp_min. got {self.cfg.kp_max} <= {self.cfg.kp_min}")
        self._kp_action_clip_min = float(self.cfg.kp_action_clip[0])
        self._kp_action_clip_max = float(self.cfg.kp_action_clip[1])
        if self._kp_action_clip_max <= self._kp_action_clip_min:
            raise ValueError(
                "kp_action_clip must satisfy high > low. "
                f"got {self._kp_action_clip_max} <= {self._kp_action_clip_min}"
            )
        self._kp_mapping_mode = str(self.cfg.kp_mapping_mode).strip().lower()
        if self._kp_mapping_mode not in {"normalized", "default_scale"}:
            raise ValueError(
                "kp_mapping_mode must be one of {'normalized', 'default_scale'}. "
                f"got '{self.cfg.kp_mapping_mode}'"
            )

        leg_order = tuple(str(name).upper() for name in self.cfg.leg_order)
        expected_legs = {"FL", "FR", "RL", "RR"}
        if len(leg_order) != 4 or set(leg_order) != expected_legs:
            raise ValueError(f"leg_order must be a permutation of {sorted(expected_legs)}. got {leg_order}")
        self._leg_order = leg_order
        self._leg_name_to_id = {name: i for i, name in enumerate(self._leg_order)}

        self._joint_leg_ids = self._resolve_joint_leg_ids(self._joint_names)
        self._action_joint_global_ids = self._resolve_action_joint_global_ids()
        self._asset_to_action_col = torch.full((self._asset.num_joints,), -1, device=self.device, dtype=torch.long)
        self._asset_to_action_col[self._action_joint_global_ids] = torch.arange(
            self._num_joints, device=self.device, dtype=torch.long
        )

        # Keep the first 12 dimensions as position scaling, and force Kp action dimensions to identity scale.
        if isinstance(self._scale, float):
            self._scale = torch.full((self.num_envs, self.action_dim), float(self._scale), device=self.device)
        else:
            self._scale = self._scale.clone()
        self._scale[:, self._num_joints :] = 1.0

        # Export/runtime parity: explicitly assign finite clip bounds to the 4 stiffness action dimensions.
        self._clip[:, self._num_joints :, 0] = self._kp_action_clip_min
        self._clip[:, self._num_joints :, 1] = self._kp_action_clip_max

        # Keep default offsets for position actions only.
        if self.cfg.use_default_offset:
            if isinstance(self._offset, float):
                self._offset = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            else:
                self._offset = self._offset.clone()
            self._offset[:, : self._num_joints] = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._action_delay_steps = max(0, int(self.cfg.action_delay_steps))
        self._applied_raw_actions = torch.zeros_like(self._raw_actions)
        if self._action_delay_steps > 0:
            self._delay_buf_size = self._action_delay_steps + 1
            self._raw_action_history = torch.zeros(
                (self.num_envs, self._delay_buf_size, self.action_dim), device=self.device, dtype=self._raw_actions.dtype
            )
            self._raw_action_history_write_idx = 0

    @property
    def action_dim(self) -> int:
        return self._num_joints + 4

    @property
    def leg_order(self) -> tuple[str, str, str, str]:
        return self._leg_order

    @property
    def applied_raw_actions(self) -> torch.Tensor:
        return self._applied_raw_actions

    def _process_raw_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        processed_actions = raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            processed_actions = torch.clamp(processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        processed_actions[:, self._num_joints :] = torch.clamp(
            processed_actions[:, self._num_joints :],
            min=self._kp_action_clip_min,
            max=self._kp_action_clip_max,
        )
        return processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self._action_delay_steps > 0:
            self._raw_action_history[:, self._raw_action_history_write_idx] = self._raw_actions
            read_idx = (self._raw_action_history_write_idx + 1) % self._delay_buf_size
            self._applied_raw_actions[:] = self._raw_action_history[:, read_idx]
            self._raw_action_history_write_idx = read_idx
        else:
            self._applied_raw_actions[:] = self._raw_actions

        self._processed_actions = self._process_raw_actions(self._applied_raw_actions)

    def compute_leg_gains_from_actions(self, leg_kp_actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps the 4 per-leg stiffness actions to per-leg Kp/Kd gains."""
        if leg_kp_actions is None:
            leg_kp_actions = self.processed_actions[:, self._num_joints :]
        if leg_kp_actions.shape[-1] != 4:
            raise ValueError(f"Expected per-leg stiffness actions with shape (*, 4). got {tuple(leg_kp_actions.shape)}")

        kp_min = float(self.cfg.kp_min)
        kp_max = float(self.cfg.kp_max)
        if self._kp_mapping_mode == "default_scale":
            kp_default = float(self.cfg.kp_default)
            kp_action_scale = float(self.cfg.kp_action_scale)
            target_kp_per_leg = kp_default + leg_kp_actions * kp_action_scale
            target_kp_per_leg = torch.clamp(target_kp_per_leg, min=kp_min, max=kp_max)
        else:
            kp_action_clip_low = float(self.cfg.kp_action_clip[0])
            kp_action_clip_high = float(self.cfg.kp_action_clip[1])
            clip_span = max(kp_action_clip_high - kp_action_clip_low, 1e-6)
            leg_kp_norm = (leg_kp_actions - kp_action_clip_low) / clip_span
            target_kp_per_leg = kp_min + leg_kp_norm * (kp_max - kp_min)

        target_kd_per_leg = float(self.cfg.kd_sqrt_scale) * torch.sqrt(torch.clamp(target_kp_per_leg, min=0.0))
        return target_kp_per_leg, target_kd_per_leg

    def _resolve_action_joint_global_ids(self) -> torch.Tensor:
        if isinstance(self._joint_ids, slice):
            return torch.arange(self._asset.num_joints, device=self.device, dtype=torch.long)
        return torch.as_tensor(self._joint_ids, device=self.device, dtype=torch.long)

    def _resolve_joint_leg_ids(self, joint_names: list[str]) -> torch.Tensor:
        leg_ids = []
        for joint_name in joint_names:
            leg_prefix = joint_name.split("_", 1)[0].upper()
            if leg_prefix not in self._leg_name_to_id:
                raise ValueError(f"Cannot parse leg prefix from joint '{joint_name}'. leg_order={self._leg_order}")
            leg_ids.append(self._leg_name_to_id[leg_prefix])
        return torch.tensor(leg_ids, device=self.device, dtype=torch.long)

    def _apply_per_joint_gains(self, kp_joint: torch.Tensor, kd_joint: torch.Tensor):
        for actuator in self._asset.actuators.values():
            if isinstance(actuator.joint_indices, slice):
                actuator_global_joint_ids = torch.arange(self._asset.num_joints, device=self.device, dtype=torch.long)
            else:
                actuator_global_joint_ids = torch.as_tensor(actuator.joint_indices, device=self.device, dtype=torch.long)

            action_cols = self._asset_to_action_col[actuator_global_joint_ids]
            valid_mask = action_cols >= 0
            if not torch.any(valid_mask):
                continue

            actuator_joint_cols = torch.nonzero(valid_mask, as_tuple=False).flatten()
            action_cols = action_cols[actuator_joint_cols]

            actuator.stiffness[:, actuator_joint_cols] = kp_joint[:, action_cols]
            actuator.damping[:, actuator_joint_cols] = kd_joint[:, action_cols]

            if isinstance(actuator, ImplicitActuator):
                global_joint_ids = actuator_global_joint_ids[actuator_joint_cols]
                self._asset.write_joint_stiffness_to_sim(kp_joint[:, action_cols], joint_ids=global_joint_ids)
                self._asset.write_joint_damping_to_sim(kd_joint[:, action_cols], joint_ids=global_joint_ids)

    def _get_generalized_gravity_forces(self) -> torch.Tensor:
        view = self._asset.root_physx_view
        for method_name in (
            "get_gravity_compensation_forces",
            "get_generalized_gravity_forces",
            "get_gravity_forces",
        ):
            method = getattr(view, method_name, None)
            if callable(method):
                forces = method()
                return torch.as_tensor(forces, device=self.device, dtype=self.processed_actions.dtype)
        raise AttributeError("Articulation view does not expose a gravity compensation method.")

    def apply_actions(self):
        # Position actions
        joint_pos_actions = self.processed_actions[:, : self._num_joints]
        self._asset.set_joint_position_target(joint_pos_actions, joint_ids=self._joint_ids)

        # Per-leg stiffness actions
        leg_kp_actions = self.processed_actions[:, self._num_joints :]
        target_kp_per_leg, target_kd_per_leg = self.compute_leg_gains_from_actions(leg_kp_actions)
        target_kp = target_kp_per_leg[:, self._joint_leg_ids]
        target_kd = target_kd_per_leg[:, self._joint_leg_ids]
        self._apply_per_joint_gains(target_kp, target_kd)

        # Optional model-based gravity compensation
        if self.cfg.gravity_comp_scale == 0.0:
            return

        generalized_forces = self._get_generalized_gravity_forces()
        root_dofs = generalized_forces.shape[1] - self._asset.num_joints
        if root_dofs < 0:
            raise RuntimeError(
                "Gravity compensation vector is smaller than the joint dimension. "
                f"got={generalized_forces.shape[1]}, num_joints={self._asset.num_joints}"
            )

        joint_forces = generalized_forces[:, root_dofs:]
        feedforward_effort = joint_forces[:, self._joint_ids] * float(self.cfg.gravity_comp_scale)

        if self.cfg.gravity_comp_max_torque is not None:
            limit = abs(float(self.cfg.gravity_comp_max_torque))
            feedforward_effort = torch.clamp(feedforward_effort, min=-limit, max=limit)

        self._asset.set_joint_effort_target(feedforward_effort, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids=env_ids)
        self._processed_actions[env_ids] = 0.0
        self._applied_raw_actions[env_ids] = 0.0
        if self._action_delay_steps > 0:
            self._raw_action_history[env_ids] = 0.0


@configclass
class GravityCompPerLegStiffnessActionCfg(actions_cfg.JointPositionActionCfg):
    """Configuration for 12-dof position + 4-dim per-leg stiffness action."""

    class_type: type = GravityCompPerLegStiffnessAction

    kp_min: float = 20.0
    kp_max: float = 60.0
    kp_mapping_mode: str = "normalized"
    kp_default: float = 40.0
    kp_action_scale: float = 20.0
    kd_sqrt_scale: float = 0.2
    kp_action_clip: tuple[float, float] = (-1.0, 1.0)
    leg_order: tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")
    action_delay_steps: int = 0

    gravity_comp_scale: float = 0.0
    gravity_comp_max_torque: float | None = None
