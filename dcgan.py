# -*- coding: utf-8 -*-
"""
DCGAN Tutorial (taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
==============
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from net import *

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

## CONFIG
dataroot = "data/celeba" # Root directory for dataset
batch_size = 128 # Batch size during training
num_epochs = 20 # Number of training epochs

lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers

image_size = 64 # Spatial size of training images. All images will be resized to this size
nc = 3 # Number of channels in the training images. For color images this is 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator

ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def save(netG, netD, outdir):
    torch.save(netD, outdir / "model_discriminator.pth")
    torch.save(netG, outdir / "model_generator.pth")
    print(f"Saved models under {outdir}/.")

def train(netG, netD, optimizerG, optimizerD, data, criterion):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################

    ## Train with all-real batch
    netD.zero_grad()
    # Format batch
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    # Forward pass real batch through D
    output = netD(real_cpu).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################

    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()

    return errD.item(), errG.item(), D_x, D_G_z1, D_G_z2

def main():

    # Create experiment dir
    REPO_BASE_DIR = Path(__file__).absolute().parent
    EXP_DIR = Path("exps/exp_" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    MODELS_DIR = REPO_BASE_DIR / EXP_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Experiment directory created at {EXP_DIR}")


    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    workers = os.cpu_count() # Number of workers for dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Create the Generator and Discriminator network
    netG = Generator(nz, nc, ngf, ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(nc, ndf, ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)
    print(netD)

    # Initialize the ``BCELoss`` function, and Setup Adam optimizers for both G and D
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    lr_schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    lr_schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    ######################################################################
    # **TRAINING LOOP *

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            errD, errG, D_x, D_G_z1, D_G_z2 = train(netG, netD, optimizerG, optimizerD, data, criterion)

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tLR:( %f %f)'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD, errG, D_x, D_G_z1, D_G_z2, optimizerG.param_groups[0]['lr'], optimizerD.param_groups[0]['lr']))

            # Save Losses for plotting later
            G_losses.append(errG)
            D_losses.append(errD)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img = vutils.make_grid(fake, padding=2, normalize=True)
                vutils.save_image(img, EXP_DIR / "netG_out_at_{}.png".format(iters))
                img_list.append(img)

            iters += 1

        # LR step
        if epoch % 2 == 0:
            lr_schedulerD.step()
            lr_schedulerG.step()

    print("Finished Training.")

    ######################################################################
    # save trained models
    save(netG, netD, MODELS_DIR)

    ######################################################################
    # **Loss versus training iteration**
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    now = datetime.now()
    fn = "plot_" + now.strftime("%Y-%m-%d-%H-%M-%S") + ".png"
    plt.savefig(EXP_DIR / fn, bbox_inches='tight')
    plt.show()

    ######################################################################
    # **Visualization of G’s progression**
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    ######################################################################
    # **Real Images vs. Fake Images**

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

if __name__=="__main__":
    main()
