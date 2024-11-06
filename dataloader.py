from scipy.spatial.transform import Rotation as R
from pathlib import Path
import pickle
import numpy as np
import torch

ARM_JOINTS = ['RSC', 'RSP', 'RSR', 'RSY', 'REP', 'RWRR',
              'LSC', 'LSP', 'LSR', 'LSY', 'LEP', 'LWRR']
LEG_JOINTS = ['RCY', 'RCR', 'RCP', 'RKP', 'RAP', 'RAR',
              'LCY', 'LCR', 'LCP', 'LKP', 'LAP', 'LAR']
WAIST_JOINTS = ['WP','WR','WY']

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).float() * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RobotStateDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_pkl, meanstd = {}, transform = None, train = True):
        self.train = train
        self.transform = transform

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
        l = 2
        for idx in range(l, len(pkl_data["root_pose"])):
            x = []
            for j in range(l+1):
                root_pose = pkl_data["root_pose"][idx-j]
                r = R.from_quat(root_pose[3:])
                euler = r.as_euler('zyx', degrees=False)
                s = np.concatenate((
                    [root_pose[2], euler[1], euler[2]],
                    [pkl_data["joint_position"][k][idx-j] for k in self.joint_names],
                ))
                x.append(s)
            dataset.append(np.array(x).flatten())
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = (self.dataset[idx] - self.mean) / self.std
        x = torch.from_numpy(x).float()
        if self.transform:
            x = self.transform(x)
        return x, x
