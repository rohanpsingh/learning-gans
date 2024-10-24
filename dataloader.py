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

        self.joint_names = ARM_JOINTS + LEG_JOINTS + WAIST_JOINTS

        # load reference data
        with open(path_to_pkl, 'rb') as f:
            pkl_data = pickle.load(f)

        dt = pkl_data["dt"]
        feats = ["root_pose", "relative_link_pose", "joint_position", "joint_velocity",
                 "force_lfoot", "force_rfoot", "force_hand"]
        pkl_data["force_rhand"] = pkl_data["force_hand"]
        pkl_data["force_lhand"] = pkl_data["force_hand"]

        self.dataset = pkl_data

    def __len__(self):
        return len(self.dataset["root_pose"]) - 1

    def __getitem__(self, idx):
        s_t1 = np.concatenate((
            [self.dataset["joint_position"][k][idx] for k in self.joint_names],
            [self.dataset["joint_velocity"][k][idx] for k in self.joint_names],
        ))
        s_t2 = np.concatenate((
            [self.dataset["joint_position"][k][idx+1] for k in self.joint_names],
            [self.dataset["joint_velocity"][k][idx+1] for k in self.joint_names],
        ))
        x = np.concatenate((s_t1, s_t2))
        x = torch.from_numpy(x).float()
        return x, x
