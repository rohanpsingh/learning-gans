import argparse
import os
from pathlib import Path
from datetime import datetime
import logging
import random

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid

from net import *

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

## CONFIG
dataroot = "data/" # Root directory for dataset

hidden_dim = 400
latent_dim = 200

ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def train(net, train_data, criterion, optimizer):
    net.train()
    losses = []

    for batch_idx, (x, _) in enumerate(train_data):
        x = x.to(device)

        x_hat, mean, log_var = net(x)
        loss = criterion(x, x_hat, mean, log_var)

        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(losses)/len(losses)

@torch.no_grad()
def valid(net, valid_data, criterion):
    net.eval()
    error = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(valid_data):
            x = x.to(device)

            x_hat, mean, log_var = net(x)
            loss = criterion(x, x_hat, mean, log_var)
            error.append(loss.data)
    return sum(error)/len(error)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", required=False, default=100, type=int)
    ap.add_argument("-d", "--dataset", required=False, default="mnist")
    ap.add_argument("--lr", required=False, default=1e-3, type=float)
    ap.add_argument("--batch_size", required=False, default=128, type=int)
    args = ap.parse_args()

    # Create experiment dir
    REPO_BASE_DIR = Path(__file__).absolute().parent
    EXP_DIR = Path("exps/exp_" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    MODELS_DIR = REPO_BASE_DIR / EXP_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Experiment directory created at {EXP_DIR}")

    # Set up logger
    log_file_path = EXP_DIR / 'train.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        filename=log_file_path,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)

    # download the MNIST datasets
    if args.dataset=="mnist":
        MNIST.mirrors = ['https://ossci-datasets.s3.amazonaws.com/mnist/']
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        train_dataset = MNIST(dataroot, transform = transform, train=True, download = True)
        valid_dataset = MNIST(dataroot, transform = transform, train=False, download = True)
        x_dim = 784
    else:
        from dataloader import RobotStateDataset
        train_dataset = RobotStateDataset(Path(args.dataset), train=True)
        valid_dataset = RobotStateDataset(Path(args.dataset), train=False)
        x_dim = 108

    # create train and valid dataloaders
    train_data = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    logging.info("train data: %d batches of batch size %d", len(train_data), args.batch_size)

    valid_data = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    logging.info("valid data: %d batches of batch size %d", len(valid_data), args.batch_size)

    ################################################################
    # # Get 25 sample training images for visualization (optional) #
    # dataiter = iter(train_data)                                  #
    # image = next(dataiter)                                       #
    # sample_images = [image[0][i,0] for i in range(25)]           #
    # fig = plt.figure(figsize=(5, 5))                             #
    # grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1) #
    # for ax, im in zip(grid, sample_images):                      #
    #     ax.imshow(im, cmap='gray')                               #
    #     ax.axis('off')                                           #
    # plt.show()                                                   #
    ################################################################

    # Set network model, loss criterion and optimizer
    net = VAE(x_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    logging.info(repr(optimizer))

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

    criterion = loss_function

    # train/test the network
    for epoch in range(args.epochs):
        train_loss = train(net, train_data, criterion, optimizer)
        valid_loss = valid(net, valid_data, criterion)
        logging.info("iters: %d train_loss: %f valid_loss: %f lr: %f",
                     epoch,
                     train_loss.item()/args.batch_size,
                     valid_loss.item()/args.batch_size,
                     optimizer.param_groups[0]['lr'])
    logging.info("Finished Training.")

    # save trained models
    torch.save(net.state_dict(), MODELS_DIR / "model.pth")
    logging.info(f"Saved model under {MODELS_DIR}/.")

if __name__=="__main__":
    main()
