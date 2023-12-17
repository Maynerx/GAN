import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from GAN import Discriminator, Generator, weights_init
from dataset import MyDataset
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# * CONST
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
dataroot = "data"
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 150
# Learning rate for optimizers
lr = 2e-3
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5



def loadDataset(path, batch_size):
    dataset = datasets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
    return dataloader

def Train(path, num_epochs, lr = 1e-3, batch_size = 64, print_rate = 5):
    dataloader = loadDataset(path, batch_size)
    netD = Discriminator(nc=nc, ndf=ndf).to(DEVICE)
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)
    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=DEVICE)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(num_epochs):
    # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=DEVICE)
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

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if epoch % print_rate == 0:
            print(f"epoch : {epoch}/{num_epochs}, loss_G : {np.mean(G_losses)}, loss_D : {np.mean(D_losses)}, D(x) : {D_x}, D(G(z)) : {D_G_z1} / {D_G_z2}")

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1

    return netG

def Generate(netG):
    fixed_noise = torch.randn(1, nz, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    return vutils.make_grid(fake, padding=2, normalize=True)

'''

if __name__ == '__main__':        

    netG = Train(
        path=dataroot,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=5,
        print_rate=25
    )

    plt.imshow(np.transpose(Generate(netG)))
    plt.show()
'''