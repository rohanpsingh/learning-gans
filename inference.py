import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from net import *


def load():
    from pathlib import Path
    REPO_BASE_DIR = Path(__file__).absolute().parent
    MODELS_DIR = REPO_BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    netD = torch.load(MODELS_DIR / "model_discriminator.pth", weights_only=False)
    netG = torch.load(MODELS_DIR / "model_generator.pth", weights_only=False)
    return netG, netD

def main():

    dataroot = "data/celeba" # Root directory for dataset
    image_size = 64
    batch_size = 4

    # Use CUDA if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset from disk
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Load the models
    netG, netD = load()
    netG.eval()
    netD.eval()

    nz = 100

    for _ in range(5):
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        real = next(iter(dataloader))[0].to(device)

        with torch.no_grad():
            fake = netG(z).detach()
            fake_pred = netD(fake).view(-1)
            real_pred = netD(real).view(-1)

        fake_pred = np.round(fake_pred.detach().cpu().numpy(), 2)
        real_pred = np.round(real_pred.detach().cpu().numpy(), 2)
        fake_img = vutils.make_grid(fake.cpu(), padding=2, normalize=True)
        real_img = vutils.make_grid(real.cpu(), padding=2, normalize=True)

        # Plot the images and predictions
        plt.figure(figsize=(12.8,4.8), tight_layout=True, dpi=200)
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.text(0.5, -0.1, f"Discriminator: {real_pred}",
                 fontsize=12, ha='center', va='top', transform=plt.gca().transAxes)
        plt.imshow(np.transpose(real_img,(1,2,0)))

        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.text(0.5, -0.1, f"Discriminator: {fake_pred}",
                 fontsize=12, ha='center', va='top', transform=plt.gca().transAxes)
        plt.imshow(np.transpose(fake_img,(1,2,0)))

        plt.show()


if __name__=="__main__":
    main()
