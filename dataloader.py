from pathlib import Path
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
        filenames = []
        if Path(path_to_pkl).is_dir():
            for path in Path(path_to_pkl).rglob('*.pkl'):
                print(path)
                filenames.append(path)
        else:
            filenames = [Path(path_to_pkl)]

        dataset = []
        for fn in filenames:
            dataset += self.load(fn)

        self.dataset = dataset

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
            data_dir = Path(path_to_pkl).parent
            torch.save(meanstd, data_dir / 'mean.pth.tar')
        return

    def load(self, path_to_pkl):
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
            x = []
            for j in range(2):
                s = np.concatenate((
                    [pkl_data["joint_position"][k][idx-j] for k in self.joint_names],
                    [pkl_data["joint_velocity"][k][idx-j] for k in self.joint_names],
                ))
                x.append(s)
            dataset.append(np.array(x).flatten())
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = (self.dataset[idx] - self.mean) / self.std
        x = torch.from_numpy(x).float()
        return x, x
