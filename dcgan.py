# -*- coding: utf-8 -*-
"""
DCGAN Tutorial (taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
==============
"""

#%matplotlib inline
import argparse
import os
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
from IPython.display import HTML

from net import *

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results


## Config
dataroot = "data/celeba" # Root directory for dataset
batch_size = 1024 # Batch size during training
num_epochs = 5 # Number of training epochs

lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers

image_size = 64 # Spatial size of training images. All images will be resized to this size
nc = 3 # Number of channels in the training images. For color images this is 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator

ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.


# save the model
def save():
    from pathlib import Path
    REPO_BASE_DIR = Path(__file__).absolute().parent
    MODELS_DIR = REPO_BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(netD, MODELS_DIR / "model_discriminator.pth")
    torch.save(netG, MODELS_DIR / "model_generator.pth")

def train():
    ######################################################################
    # Training
    # ~~~~~~~~
    # 
    # Finally, now that we have all of the parts of the GAN framework defined,
    # we can train it. Be mindful that training GANs is somewhat of an art
    # form, as incorrect hyperparameter settings lead to mode collapse with
    # little explanation of what went wrong. Here, we will closely follow
    # Algorithm 1 from the `Goodfellow’s paper <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__, 
    # while abiding by some of the best
    # practices shown in `ganhacks <https://github.com/soumith/ganhacks>`__.
    # Namely, we will “construct different mini-batches for real and fake”
    # images, and also adjust G’s objective function to maximize
    # :math:`log(D(G(z)))`. Training is split up into two main parts. Part 1
    # updates the Discriminator and Part 2 updates the Generator.
    # 
    # **Part 1 - Train the Discriminator**
    # 
    # Recall, the goal of training the discriminator is to maximize the
    # probability of correctly classifying a given input as real or fake. In
    # terms of Goodfellow, we wish to “update the discriminator by ascending
    # its stochastic gradient”. Practically, we want to maximize
    # :math:`log(D(x)) + log(1-D(G(z)))`. Due to the separate mini-batch
    # suggestion from `ganhacks <https://github.com/soumith/ganhacks>`__,
    # we will calculate this in two steps. First, we
    # will construct a batch of real samples from the training set, forward
    # pass through :math:`D`, calculate the loss (:math:`log(D(x))`), then
    # calculate the gradients in a backward pass. Secondly, we will construct
    # a batch of fake samples with the current generator, forward pass this
    # batch through :math:`D`, calculate the loss (:math:`log(1-D(G(z)))`),
    # and *accumulate* the gradients with a backward pass. Now, with the
    # gradients accumulated from both the all-real and all-fake batches, we
    # call a step of the Discriminator’s optimizer.
    # 
    # **Part 2 - Train the Generator**
    # 
    # As stated in the original paper, we want to train the Generator by
    # minimizing :math:`log(1-D(G(z)))` in an effort to generate better fakes.
    # As mentioned, this was shown by Goodfellow to not provide sufficient
    # gradients, especially early in the learning process. As a fix, we
    # instead wish to maximize :math:`log(D(G(z)))`. In the code we accomplish
    # this by: classifying the Generator output from Part 1 with the
    # Discriminator, computing G’s loss *using real labels as GT*, computing
    # G’s gradients in a backward pass, and finally updating G’s parameters
    # with an optimizer step. It may seem counter-intuitive to use the real
    # labels as GT labels for the loss function, but this allows us to use the
    # :math:`log(x)` part of the ``BCELoss`` (rather than the :math:`log(1-x)`
    # part) which is exactly what we want.
    # 
    # Finally, we will do some statistic reporting and at the end of each
    # epoch we will push our fixed_noise batch through the generator to
    # visually track the progress of G’s training. The training statistics
    # reported are:
    # 
    # -  **Loss_D** - discriminator loss calculated as the sum of losses for
    #    the all real and all fake batches (:math:`log(D(x)) + log(1 - D(G(z)))`).
    # -  **Loss_G** - generator loss calculated as :math:`log(D(G(z)))`
    # -  **D(x)** - the average output (across the batch) of the discriminator
    #    for the all real batch. This should start close to 1 then
    #    theoretically converge to 0.5 when G gets better. Think about why
    #    this is.
    # -  **D(G(z))** - average discriminator outputs for the all fake batch.
    #    The first number is before D is updated and the second number is
    #    after D is updated. These numbers should start near 0 and converge to
    #    0.5 as G gets better. Think about why this is.
    # 
    # **Note:** This step might take a while, depending on how many epochs you
    # run and if you removed some data from the dataset.
    # 

    # Training Loop

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

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

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    return G_loses, D_losses, img_list

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

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# Create the generator
netG = Generator(nz, nc, ngf, ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)
print(netG)

# Create the Discriminator
netD = Discriminator(nc, ndf, ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)
print(netD)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# training loop
G_losses, D_losses, img_list = train()

# save trained models
save()


######################################################################
# **Loss versus training iteration**
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


######################################################################
# **Visualization of G’s progression**
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


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
