import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataloader import RobotStateDataset

from net import *

# Use CUDA if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))

    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return KL

@torch.no_grad()
def store_NLL(x, recon, mu, logvar, z):
    sigma = torch.exp(0.5*logvar)
    b = x.size(0)
    recon = recon.contiguous()
    #cross_entropy = torch.nn.functional.cross_entropy(recon, target, reduction='none')
    mse_loss = nn.functional.mse_loss(x, recon, reduction='none')#.sum(axis=1)

    log_p_x_z = -torch.sum(mse_loss.view(b ,-1), 1)
    log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
    z_eps = (z - mu)/sigma
    z_eps = z_eps.view(b, -1)
    log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)

    weights = log_p_x_z+log_p_z-log_q_z_x
    return weights

@torch.no_grad()
def compute_NLL(weights):
    NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max())
    return NLL_loss

@torch.no_grad()
def likelihood(net, data):
    error = []
    for batch_idx, (x, _) in enumerate(data):
        x = x.expand(200, -1).contiguous()
        x = x.to(device)
        b = x.size(0)
        weights_agg  = []

        for batch_number in range(5):

            mu, logvar = net.encode(x)
            z = net.reparameterization(mu, torch.exp(0.5*logvar))
            recon = net.decode(z)

            weights = store_NLL(x, recon, mu, logvar, z)

        weights_agg.append(weights)
        weights_agg = torch.stack(weights_agg).view(-1)

        nll_loss = compute_NLL(weights_agg)
        error.append(nll_loss.item())
    return error

@torch.no_grad()
def reconstruction_loss(net, data):
    error = []
    for batch_idx, (x, _) in enumerate(data):
        x = x.to(device)
        x_hat, mean, log_var = net(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction='sum')
        error.append(loss.item())
    return error

def main():
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        required=True,
                        type=Path,
                        help="path to trained model dir",
    )
    parser.add_argument("--test-data",
                        required=True,
                        type=Path,
                        help="path to dataset to test",
    )
    parser.add_argument("--id-data",
                        required=False,
                        default=Path("data/mocap/carrybox/reference_motion.pkl"),
                        type=Path,
                        help="path to in-distribution dataset to test",
    )
    args = parser.parse_args()

    # Load the models
    net = torch.load(args.model, weights_only=False)
    net.eval()
    manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    print("====Loaded weights====")

    meanstd = torch.load('data/mean.pth.tar', weights_only=False)

    # load data (OOD)
    dataset = RobotStateDataset(Path(args.test_data), meanstd=meanstd, train=False)
    data = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)
    print("test data: {} batches of batch size {}".format(len(data), 1))

    error_ood = reconstruction_loss(net, data)
    likelihood_ood = likelihood(net, data)

    # load data (ID)
    dataset = RobotStateDataset(args.id_data, meanstd=meanstd, train=False)
    data = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)
    print("test data: {} batches of batch size {}".format(len(data), 1))

    error_id = reconstruction_loss(net, data)
    likelihood_id = likelihood(net, data)

    print("Mean OOD error = {:.3f}. Mean ID error = {:.3f}".format(
        np.mean(error_ood), np.mean(error_id))
    )
    print("Mean OOD likelihood = {:.3f}. Mean ID likelihood = {:.3f}".format(
        np.mean(likelihood_ood), np.mean(likelihood_id))
    )

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(error_ood, label='ood')
    axs[0].plot(error_id, label='id')
    axs[0].legend()
    axs[1].plot(likelihood_ood, label='ood')
    axs[1].plot(likelihood_id, label='id')
    axs[1].legend()
    plt.show()

if __name__=="__main__":
    main()
