import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from utils import save_generator_image, weights_init
from utils import label_fake, label_real, create_noise
from dc_gan import Generator, Discriminator
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from shutil import rmtree
from tqdm import tqdm
from os import mkdir
import pathlib
import imageio
import glob

matplotlib.style.use('ggplot')

# Clear and make output image folder
current_file_path = pathlib.Path(__file__).parent.absolute()
path = current_file_path.parents[0] / 'outputs' / 'output_images'
rmtree(path, ignore_errors=True)
mkdir(path)

image_size = 96
batch_size = 128
nz = 25 # latent vector size
beta1 = 0.5 # beta1 value for Adam optimizer
beta2 = 0.999 # beta2 value for Adam optimizer
lr_g = 0.00015
lr_d = 0.00006
sample_size = 64 # number of generated example images
epochs = 150

# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', str(device).upper(), '\n')

# image transforms
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# prepare the data
train_data = datasets.ImageFolder(
    root='../input_images',
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device) 

# initialize generator weights
generator.apply(weights_init)

# initialize discriminator weights
discriminator.apply(weights_init)

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2), weight_decay=0.000001)
optim_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2), weight_decay=0.0)

# loss function
criterion = nn.BCELoss()

losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch

def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)

    # get the real label vector
    real_label = label_real(b_size)

    # get the fake label vector
    fake_label = label_fake(b_size)

    optimizer.zero_grad()

    # get the outputs by doing real data forward pass
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)

    # get the outputs by doing fake data forward pass
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)

    # compute gradients of real loss 
    loss_real.backward()

    # compute gradients of fake loss
    loss_fake.backward()

    # update discriminator parameters
    optimizer.step()

    return loss_real + loss_fake

def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)

    # get the real label vector
    real_label = label_real(b_size)

    optimizer.zero_grad()

    # output by doing a forward pass of the fake data through discriminator
    output = discriminator(data_fake)
    loss = criterion(output, real_label)

    # compute gradients of loss
    loss.backward()

    # update generator parameters
    optimizer.step()

    return loss

# create the noise vector
noise = create_noise(sample_size, nz)

generator.train()
discriminator.train()

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
        image, _ = data
        image = image.to(device)
        b_size = len(image)
        # forward pass through generator to create fake data
        data_fake = generator(create_noise(b_size, nz)).detach()
        data_real = image
        loss_d += train_discriminator(optim_d, data_real, data_fake)
        data_fake = generator(create_noise(b_size, nz))
        loss_g += train_generator(optim_g, data_fake)

    # final forward pass through generator to create fake data...
    # ...after training for current epoch
    generated_img = generator(noise).cpu().detach()
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"../outputs/output_images/gen_img{(epoch+1):04}.png")
    epoch_loss_g = loss_g / bi # total generator loss for the epoch
    epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)

    print(f"Epoch {epoch+1} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

print('DONE TRAINING')
# save the model weights to disk
torch.save(generator.state_dict(), '../outputs/generator.pth')

# Save GIF of images
images = [imageio.imread(file) for file in sorted(glob.glob(str(path / '*.png')))]
duration = ([0.075] * (len(images) - 1))
duration.append(5)
imageio.mimwrite(str(path / 'outputs.gif'), images, duration=duration)

# plot and save the generator and discriminator loss
plt.figure()
plt.plot(losses_g, label='Generator loss')
plt.plot(losses_d, label='Discriminator Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.close()

# Remove __pycache__ folder
rmtree(pathlib.Path(__file__).parent.absolute() / '__pycache__')
