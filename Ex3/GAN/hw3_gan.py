# -*- coding: utf-8 -*-

gpu_enable = 0
retrain_flag = 0

#foldername = '/content/drive/MyDrive/Ex1/'
foldername = './data/'

dcgan_train = retrain_flag
wgan_train = retrain_flag

# !nvidia-smi

# Enable Cuda
import torch
print('cuda is available =',torch.cuda.is_available())
if torch.cuda.is_available() and gpu_enable:
    mydevice = torch.cuda.current_device()
    use_cuda = True
else:
    mydevice = torch.device("cpu")
    use_cuda = False
print('mydevice =',mydevice)

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
from IPython.display import HTML, clear_output
from torch.autograd import Variable

import time
from torchvision.utils import make_grid

# manualSeed = 444
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True)
# torch.use_deterministic_algorithms(True)
# dataroot = "C:/Users/markg/DataSets/"
dataroot = foldername
dcgan_gen_fname = foldername + "dcgan_gen_model.pt"
dcgan_dis_fname = foldername + "dcgan_dis_model.pt"
dcgan_gen_loss_fname = foldername + "dcgan_gen_loss.npy"
dcgan_dis_loss_fname = foldername + "dcgan_dis_loss.npy"
dcgan_img_list_fname = foldername + "dcgan_img_list.npy"
wgan_gen_fname = foldername + "wgan_gen_model.pt"
wgan_dis_fname = foldername + "wgan_dis_model.pt"
wgan_gen_loss_fname = foldername + "wgan_gen_loss.npy"
wgan_dis_loss_fname = foldername + "wgan_dis_loss.npy"
noise_fname = foldername + "noise.npy"

def to_img(x):
  x = x.clamp(0, 1)
  return x
def visualise_output(images, x, y):
  with torch.no_grad():
    images = images.cpu()
    images = to_img(images)
    np_imagegrid = make_grid(images, x, y).numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
    plt.show()

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images.
image_size = 28

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lr = 0.0002
lr_wgan = 0.0001

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# WGAN clip value
clip_value = .01

# number of epochs to train critic per one epoch to train generator
n_critic = 25

# Sample interval
sample_interval = 400

img_shape = (nc, image_size, image_size)

dataloader = torch.utils.data.DataLoader(
  dset.FashionMNIST(
    dataroot,
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ),
  ),
  batch_size=batch_size,
  shuffle=True,
)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 6 x 6``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 13 x 13``
            nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 0, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 28 x 28``
        )
    def forward(self, input):
        return self.main(input)

# Create the generator
# netG = Generator(ngpu).to(device)
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
# if (device.type == 'cuda') and (ngpu > 1):
    # netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 28 x 28``
            nn.Conv2d(nc, ndf*2, 4, 2, 0, bias=False),
            # nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 13 x 13''
            nn.Conv2d(ndf*2, ndf * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 6 x 6``
            nn.Conv2d(ndf * 4, ndf * 6, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 6),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*6) x 1 x 1``
            nn.Conv2d(ndf * 6, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator_Lin(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_Lin, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 28 x 28``
            nn.Conv2d(nc, ndf*2, 4, 2, 0, bias=False),
            # nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 13 x 13''
            nn.Conv2d(ndf*2, ndf * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 6 x 6``
            nn.Conv2d(ndf * 4, ndf * 6, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 6),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*6) x 1 x 1``
            nn.Conv2d(ndf * 6, 1, 1, 1, 0, bias=False),
        )
    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)

# Commented out IPython magic to ensure Python compatibility.
# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
if os.path.isfile(noise_fname) and not retrain_flag:
    fixed_noise = torch.tensor(np.load(noise_fname)).to(device)
    print('loaded noise from file')
else:
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    np.save(noise_fname, fixed_noise.cpu())
    print('generated new noise')


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
t0=time.time()

if not dcgan_train:
    netG.load_state_dict(torch.load(dcgan_gen_fname,weights_only=False,map_location=torch.device('cpu')))
    netD.load_state_dict(torch.load(dcgan_dis_fname,weights_only=False,map_location=torch.device('cpu')))
    G_losses = np.load(dcgan_gen_loss_fname)
    D_losses = np.load(dcgan_dis_loss_fname)
    print('Loaded DCGAN Generator and Discriminator')
else:
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
                clear_output()
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # visualise_output(fake.data[:50],10, 10)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    print('dcgan time: ', time.time()-t0, 'sec')
    torch.save(netG.state_dict(), dcgan_gen_fname)
    torch.save(netD.state_dict(), dcgan_dis_fname)
    np.save(dcgan_gen_loss_fname, G_losses)
    np.save(dcgan_dis_loss_fname, D_losses)
    print('Saved DCGAN Generator and Discriminator')

plt.figure(figsize=(10,5))
plt.title("DCGAN: Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# fig = plt.figure(figsize=(8,8))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
#
# HTML(ani.to_jshtml())

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
plt.title("Fake Images DCGAN")
fake = netG(fixed_noise).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
fake_last_cdgan = img_list[-1]
plt.imshow(np.transpose(fake_last_cdgan,(1,2,0)))
# plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

netG2 = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG2 = nn.DataParallel(netG2, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG2.apply(weights_init)

# Print the model
print(netG2)

netD2 = Discriminator_Lin(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD2 = nn.DataParallel(netD2, list(range(ngpu)))
netD2.apply(weights_init)
print(netD2)

netG2.apply(weights_init)
netD2.apply(weights_init)
optimizer_G = torch.optim.RMSprop(netG2.parameters(), lr=lr_wgan)
optimizer_D = torch.optim.RMSprop(netD2.parameters(), lr=lr_wgan)
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
WGAN_G_losses = []
WGAN_D_losses = []

t1 = time.time()
if not wgan_train:
    netG2.load_state_dict(torch.load(wgan_gen_fname,weights_only=False,map_location=torch.device('cpu')))
    netD2.load_state_dict(torch.load(wgan_dis_fname,weights_only=False,map_location=torch.device('cpu')))
    WGAN_G_losses = list(np.load(wgan_gen_loss_fname))
    WGAN_D_losses = list(np.load(wgan_dis_loss_fname))
    print('Loaded WGAN Generator and Discriminator')
else:
    for epoch in range(num_epochs):
      for i, (imgs, _) in enumerate(dataloader):
          valid = Variable(torch.tensor(np.ones((imgs.shape[0], 1)), dtype=torch.float32, device=mydevice), requires_grad=False)
          fake = Variable(torch.tensor(np.zeros((imgs.shape[0], 1)), dtype=torch.float32, device=mydevice), requires_grad=False)

          real_imgs = Variable(imgs.type(Tensor))

          optimizer_G.zero_grad()
          fake_imgs = netG2(fixed_noise).detach()
          d_loss = -torch.mean(netD2(real_imgs)) + torch.mean(netD2(fake_imgs))
          d_loss.backward()
          optimizer_D.step()

          for p in netD2.parameters():
            p.data.clamp_(-clip_value, clip_value)

          if i % n_critic == 0:
            optimizer_G.zero_grad()
            fake_images_from_generator = netG2(fixed_noise)
            g_loss = -torch.mean(netD2(fake_images_from_generator))

            g_loss.backward()
            optimizer_G.step()

          WGAN_G_losses.append(g_loss.item())
          WGAN_D_losses.append(d_loss.item())

          batches_done = epoch * len(dataloader) + i
          if batches_done % sample_interval == 0:
            clear_output()
            print(f"Epoch:{epoch}:It{i}:DLoss{d_loss.item()}:GLoss{g_loss.item()}")
            visualise_output(fake_images_from_generator.data[:50],10, 10)
    print('wgan time: ',round(time.time()-t1,1),'sec')
    torch.save(netG2.state_dict(), wgan_gen_fname)
    torch.save(netD2.state_dict(), wgan_dis_fname)
    np.save(wgan_gen_loss_fname, WGAN_G_losses)
    np.save(wgan_dis_loss_fname, WGAN_D_losses)
    print('Saved WGAN Generator and Discriminator')

plt.figure(figsize=(10,5))
plt.title("WGAN: Generator and Discriminator Loss During Training")
plt.plot(WGAN_G_losses,label="G")
plt.plot(WGAN_D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch DCGAN
with torch.no_grad():
   fake_dcgan = netG(fixed_noise).detach().cpu()
   img_list_dcgan=vutils.make_grid(fake_dcgan, padding=2, normalize=True)

plt.subplot(1,3,2)
plt.axis("off")
plt.title("DCGAN Fake Images")
plt.imshow(np.transpose(img_list_dcgan,(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,3,3)
plt.axis("off")
plt.title("WGAN Fake Images")
with torch.no_grad():
    fake_wgan = netG2(fixed_noise).detach().cpu()
    fake_last_cdgan = vutils.make_grid(fake_wgan, padding=2, normalize=True)
plt.imshow(np.transpose(vutils.make_grid(fake_last_cdgan.to(device).data[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()

select_random_input = 1
if select_random_input:
    real_idx = np.random.randint(fixed_noise.shape[0],size=2)
    dcgan_idx1 = np.random.randint(fixed_noise.shape[0],size=2)
    wgan_idx2 = np.random.randint(fixed_noise.shape[0],size=2)
else:
    real_idx = [50, 39]
    dcgan_idx1 = [41, 53]
    wgan_idx2 = [30, 27]

real_images_sample = real_batch[0].to(device)[real_idx]
test_input_noise1 = fixed_noise[dcgan_idx1]
test_input_noise2 = fixed_noise[wgan_idx2]

# Generate noise
# test_input_noise1 = torch.randn(2, nz, 1, 1, device=device)
# test_input_noise2 = torch.randn(2, nz, 1, 1, device=device)

# Generate image from noise using DCGAN
fake_cdgan = netG(test_input_noise1).detach().cpu()
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.axis("off")
plt.title("Two Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_images_sample, padding=5, normalize=True).cpu(),(1,2,0)))
plt.subplot(1,3,2)
plt.axis("off")
plt.title("Two DCGAN Ouputs")
plt.imshow(np.transpose(vutils.make_grid(fake_cdgan.to(device).data[:2], padding=5, normalize=True).cpu(),(1,2,0)))
# Generate image from noise using DCGAN
fake_wgan = netG2(test_input_noise2).detach().cpu()
plt.subplot(1,3,3)
plt.axis("off")
plt.title("Two WGAN Ouputs")
plt.imshow(np.transpose(vutils.make_grid(fake_wgan.to(device).data[:2], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()

