import torch


class MotionTracking:
    def __init__(self, env, all_joint3d):
        self.env = env
        self.all_joint3d = all_joint3d    # (num_songs, 900, 24, 3)
        self.num_songs = all_joint3d.shape[0]
        self.songID = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.song_timestep = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids):
        """ Call when resetting the env """
        randomID = torch.randint(self.num_songs, (env_ids.shape[0], ), device=self.env.device)
        self.songID[env_ids] = randomID
        self.song_timestep[env_ids] = torch.zeros(env_ids.shape[0], device=self.env.device, dtype=torch.long)

    def update(self):
        """ Call when stepping the env """
        # linear transformation that map agent timestep [0-1499] to EDGE timestep [0-899]
        self.song_timestep = torch.round((self.env.episode_length_buf * 899) / 1501).long()
