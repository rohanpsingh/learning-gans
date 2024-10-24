import os
import random
import pickle
import numpy as np
import torch

ARM_JOINTS = ['RSC', 'RSP', 'RSR', 'RSY', 'REP', 'RWRR',
              'LSC', 'LSP', 'LSR', 'LSY', 'LEP', 'LWRR']
LEG_JOINTS = ['RCY', 'RCR', 'RCP', 'RKP', 'RAP', 'RAR',
              'LCY', 'LCR', 'LCP', 'LKP', 'LAP', 'LAR']
WAIST_JOINTS = ['WP','WR','WY']

class RobotStateDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_pkl, train = True):
        self.train = train
        meanstd = {}

        self.joint_names = ARM_JOINTS + LEG_JOINTS + WAIST_JOINTS

        # load reference data
        with open(path_to_pkl, 'rb') as f:
            pkl_data = pickle.load(f)

        dt = pkl_data["dt"]
        feats = ["root_pose", "relative_link_pose", "joint_position", "joint_velocity",
                 "force_lfoot", "force_rfoot", "force_hand"]
        pkl_data["force_rhand"] = pkl_data["force_hand"]
        pkl_data["force_lhand"] = pkl_data["force_hand"]

        # create dataset
        dataset = []
        for idx in range(1, len(pkl_data["root_pose"])):
            s = np.concatenate((
                [pkl_data["joint_position"][k][idx] for k in self.joint_names],
                [pkl_data["joint_velocity"][k][idx] for k in self.joint_names],
            ))
            s_ = np.concatenate((
                [pkl_data["joint_position"][k][idx-1] for k in self.joint_names],
                [pkl_data["joint_velocity"][k][idx-1] for k in self.joint_names],
            ))
            x = np.concatenate((s_, s))
            dataset.append(x)

        random.shuffle(dataset)

        l = int(0.8*len(dataset))
        if self.train:
            self.dataset = dataset[:l]
        else:
            self.dataset = dataset[l:]

        if meanstd == {}:
            # compute mean and std-dev
            self.mean = np.array(self.dataset).mean(axis=0)
            self.std = np.array(self.dataset).std(axis=0)
            meanstd = {'mean': self.mean,
                       'std': self.std}
        else:
            self.mean = meanstd['mean']
            self.std = meanstd['std']

        if self.train:
            data_dir = os.path.dirname(path_to_pkl)
            torch.save(meanstd, os.path.join(data_dir, 'mean.pth.tar'))
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = (self.dataset[idx] - self.mean) / self.std
        x = torch.from_numpy(x).float()
        return x, x
