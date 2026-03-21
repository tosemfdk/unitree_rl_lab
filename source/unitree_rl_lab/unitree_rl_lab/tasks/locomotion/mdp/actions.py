from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
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
