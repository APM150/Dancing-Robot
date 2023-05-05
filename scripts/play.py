import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.go1.front_feet_height_tracking import FrontFeetHeightTrackingEnv

from tqdm import tqdm
import random
import cv2
import imageio

from go1_gym_learn.ppo_cse import RunnerArgs
from go1_gym_learn.ppo_cse import Runner

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.env.observe_edge_targets = False  # deploying, doesn't use EDGE training data
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = FrontFeetHeightTrackingEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/2023-05-05/train_dance"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 900
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["bounding"])
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    # test_1sqE6P3XyiQ.pkl
    # test_2RicaUqd9Hg.pkl
    # test_9i6bCWIdhBw.pkl
    # test_ABfQuZqq8wg.pkl
    # test_ggJI9dKBk48.pkl
    # test_P-sGt5E2epc.pkl
    # test_UHXGc2oWyJ4.pkl
    with open("../motion_folder/test_9i6bCWIdhBw.pkl", 'rb') as f:
        data = pkl.load(f)
    joint3d = data["full_pose"]
    feet_idx = [7, 8]
    left_feet_heights = joint3d[:, feet_idx[0], 2]
    right_feet_heights = joint3d[:, feet_idx[1], 2]
    min_height = min(np.min(right_feet_heights), np.min(left_feet_heights))
    max_height = max(np.max(right_feet_heights), np.max(left_feet_heights))
    FL_height = []
    FR_height = []
    for i in range(num_eval_steps):
        FL_height.append((left_feet_heights[i] - min_height) / (max_height - min_height) * 2.0)
        FR_height.append((right_feet_heights[i] - min_height) / (max_height - min_height) * 2.0)
    print(np.array(FL_height).mean())
    print(np.array(FR_height).mean())

    frames = []
    measured_FL_height = np.zeros(num_eval_steps)
    measured_FR_height = np.zeros(num_eval_steps)
    target_FL_height = np.array(FL_height)
    target_FR_height = np.array(FR_height)
    joint_positions = np.zeros((num_eval_steps, 12))
    reward_FL = np.zeros(num_eval_steps)
    reward_FR = np.zeros(num_eval_steps)
    difference_FL = np.zeros(num_eval_steps)
    difference_FR = np.zeros(num_eval_steps)

    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):  # 1/50
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = 0 #footswing[i]
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        env.commands[:, 15] = FL_height[i]
        env.commands[:, 16] = FR_height[i]
        # print("play obs", obs['obs'][0, 18])
        # print("play cmd", env.commands[:, 15])
        # print("play FL", FL_height[i])
        obs, rew, done, info = env.step(actions)

        frame = env.render(mode='rgb_array')
        frames.append(frame)


        measured_FL_height[i] = env.foot_positions[:, 0, :].cpu().numpy()[0][2]
        measured_FR_height[i] = env.foot_positions[:, 1, :].cpu().numpy()[0][2]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        reward_FL[i] = env.reward_container._reward_FL_foot_height_tracking().cpu().numpy()[0]
        reward_FR[i] = env.reward_container._reward_FR_foot_height_tracking().cpu().numpy()[0]
        difference_FL[i] = abs(measured_FL_height[i] - FL_height[i])
        difference_FR[i] = abs(measured_FR_height[i] - FR_height[i])


    video_name = 'deploy.mp4'
    fps = 30  # Set the frames per second value
    # Convert the frames to a video using OpenCV
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frames)):
        out.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
    out.release()

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_FL_height, color='black', linestyle="-", label="Measured FL")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_FL_height, color='black', linestyle="--", label="Desired FL")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), reward_FL, color='red', linestyle="--", label="Reward FL")
    axs[0].legend()
    axs[0].set_title("FL height")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Height (m)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_FR_height, color='black', linestyle="-", label="Measured FR")
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_FR_height, color='black', linestyle="--", label="Desired FR")
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), reward_FR, color='red', linestyle="--", label="Reward FR")
    axs[1].legend()
    axs[1].set_title("FL height")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Height (m)")
    plt.savefig("deploy")




if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
