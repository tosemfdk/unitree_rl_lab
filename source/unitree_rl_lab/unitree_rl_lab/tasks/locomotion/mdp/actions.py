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
    - 0:12  -> joint position actions
    - 12:16 -> per-leg stiffness actions (FL, FR, RL, RR by default)
    """

    cfg: "GravityCompPerLegStiffnessActionCfg"

    def __init__(self, cfg: "GravityCompPerLegStiffnessActionCfg", env):
        super().__init__(cfg, env)

        if self.cfg.kp_max <= self.cfg.kp_min:
            raise ValueError(f"kp_max must be greater than kp_min. got {self.cfg.kp_max} <= {self.cfg.kp_min}")
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

        # Keep default offsets for position actions only.
        if self.cfg.use_default_offset:
            if isinstance(self._offset, float):
                self._offset = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            else:
                self._offset = self._offset.clone()
            self._offset[:, : self._num_joints] = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return self._num_joints + 4

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        kp_clip_min = float(self.cfg.kp_action_clip[0])
        kp_clip_max = float(self.cfg.kp_action_clip[1])
        self._processed_actions[:, self._num_joints :] = torch.clamp(
            self._processed_actions[:, self._num_joints :], min=kp_clip_min, max=kp_clip_max
        )

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
        kp_min = float(self.cfg.kp_min)
        kp_max = float(self.cfg.kp_max)
        if self._kp_mapping_mode == "default_scale":
            # Genesis-style mapping:
            #   kp = kp_default + action * kp_action_scale, then clamp to [kp_min, kp_max].
            kp_default = float(self.cfg.kp_default)
            kp_action_scale = float(self.cfg.kp_action_scale)
            target_kp_per_leg = kp_default + leg_kp_actions * kp_action_scale
            target_kp_per_leg = torch.clamp(target_kp_per_leg, min=kp_min, max=kp_max)
        else:
            # Legacy normalized mapping:
            #   kp_action_clip low/high maps linearly to [kp_min, kp_max].
            kp_action_clip_low = float(self.cfg.kp_action_clip[0])
            kp_action_clip_high = float(self.cfg.kp_action_clip[1])
            clip_span = max(kp_action_clip_high - kp_action_clip_low, 1e-6)
            leg_kp_norm = (leg_kp_actions - kp_action_clip_low) / clip_span
            target_kp_per_leg = kp_min + leg_kp_norm * (kp_max - kp_min)

        target_kp = target_kp_per_leg[:, self._joint_leg_ids]
        target_kd = float(self.cfg.kd_sqrt_scale) * torch.sqrt(torch.clamp(target_kp, min=0.0))
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

    gravity_comp_scale: float = 0.0
    gravity_comp_max_torque: float | None = None
