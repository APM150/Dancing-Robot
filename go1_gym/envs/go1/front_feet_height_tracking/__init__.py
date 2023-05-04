from isaacgym import gymutil, gymapi
import torch
import numpy as np
import pickle as pkl
import glob
from params_proto import Meta
from typing import Union

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.front_feet_height_tracking.motion_tracking import MotionTracking


class FrontFeetHeightTrackingEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)

        # preprocess the referenced motion data
        all_joint3d = np.zeros((7, 900, 24, 3))
        for i, file_name in enumerate(glob.glob("../../../../motion_folder/*.pkl")):
            with open(file_name, 'rb') as f:
                data = pkl.load(f)
            joint3d = data["full_pose"]
            feet_idx = [7, 8]
            left_feet_heights = joint3d[:, feet_idx[0], 2]
            right_feet_heights = joint3d[:, feet_idx[1], 2]
            min_height = min(np.min(right_feet_heights), np.min(left_feet_heights))
            max_height = max(np.max(right_feet_heights), np.max(left_feet_heights))
            # normalize to [0, 0.2]
            joint3d[:, feet_idx[0], 2] = (left_feet_heights - min_height) / (max_height - min_height) * 0.2
            joint3d[:, feet_idx[1], 2] = (right_feet_heights - min_height) / (max_height - min_height) * 0.2
            all_joint3d[i] = joint3d
        self.all_joint3d = torch.from_numpy(all_joint3d).to(sim_device)

        # motion tracking
        self.motion_tracking = MotionTracking(self, self.all_joint3d)

    def step(self, actions):
        # reset motion tracking
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.motion_tracking.reset(env_ids)

        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        # update motion tracking
        self.motion_tracking.update()

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.motion_tracking.reset(torch.arange(self.num_envs, device=self.device))
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs


if __name__ == '__main__':
    print(glob.glob("../../../../motion_folder/*.pkl"))
    all_joint3d = np.zeros((7, 900, 24, 3))
    for i, file_name in enumerate(glob.glob("../../../../motion_folder/*.pkl")):
        with open(file_name, 'rb') as f:
            data = pkl.load(f)
        joint3d = data["full_pose"]
        all_joint3d[i] = joint3d
        feet_idx = [7, 8]
        left_feet_heights = joint3d[:, feet_idx[0], 2]
        right_feet_heights = joint3d[:, feet_idx[1], 2]
        min_height = min(np.min(right_feet_heights), np.min(left_feet_heights))
        max_height = max(np.max(right_feet_heights), np.max(left_feet_heights))
        print(max_height)
        print(min_height)
    all_joint3d = all_joint3d.reshape((-1, 24, 3))
