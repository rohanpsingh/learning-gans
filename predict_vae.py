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

def plot_latent_space(net, device, scale=1.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = net.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization', fontsize=28)
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x, fontsize=18)
    plt.yticks(pixel_range, sample_range_y, fontsize=18)
    plt.xlabel("mean, z [0]", fontsize=28)
    plt.ylabel("var, z [1]", fontsize=28)
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

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
    for _ in range(5):
        z = np.random.uniform(-2, 2, 2)
        generate_digit(net, *z, device)

    # plot the latent space
    plot_latent_space(net, device)

if __name__=="__main__":
    main()
