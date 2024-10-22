import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import random
import matplotlib.pyplot as plt

from net import *

def generate_digit(net, mean, var, device):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = net.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array
    plt.imshow(digit, cmap='gray')
    plt.title("[{:.3f}, {:.3f}]".format(mean, var), fontsize=28)
    plt.axis('off')
    plt.show()

def initialize_net(trained_weights):
    net = VAE()
    net.eval()
    net.load_state_dict(torch.load(trained_weights, weights_only=True))
    manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    print("====Loaded weights====")
    return net

def main():

    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        required=True,
                        type=Path,
                        help="path to trained model dir",
    )
    args = parser.parse_args()

    # Use CUDA if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the models
    net = initialize_net(args.model).to(device)

    # generate new images
    for _ in range(10):
        z = np.random.uniform(-2, 2, 2)
        generate_digit(net, *z, device)

if __name__=="__main__":
    main()
