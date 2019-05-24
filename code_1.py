# made by Aditya Dubey 1610110007 
# written in python using pytorch library
import matplotlib
import torchvision
from torch.utils.data import DataLoader as DataLoader
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
import numpy
import torch
import os


epochs = 10
batch_size = 100;

DATA_PATH = '/Users/adityadubey/PycharmProjects/dsp-project'
#/Users/adityadubey/PycharmProjects/dsp-project
MODEL_STORE_PATH = '/Users/adityadubey/PycharmProjects/dsp-project/pytorch_models\\'

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
# divides the training data into batch sizes of 100 and shuffles the data
test_loader =  DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#

# min min V(D,G)
#data = train
# discrimnator
# image 93 almost 8 


class Discrimnator(nn.Module):
    def __init__(self):
        super(Discrimnator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784,256),
            nn.LeakyReLU(0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256,100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(100,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

discrimnator = Discrimnator()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(100,256),
            nn.LeakyReLU(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(512, 784),
            nn.LeakyReLU(0.3)
        )
        self.layer4 = nn.Sequential(
            nn.Tanh()
        )

    def forward(self,x):
        x = self.layer1(x);
        x = self.layer2(x);
        x = self.layer3(x);
        x = self.layer4(x);
        return x

generator = Generator()

# optimizers and loss

learning_rate = 0.0002;

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

optimizer_dis = torch.optim.Adam(discrimnator.parameters(), lr=learning_rate)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=learning_rate)

loss = criterion = nn.BCELoss()

def train_gan(images,fake_labels,real_labels,batch_size):
    # discrimnator

    # train it on real images for discrimnator
    output = discrimnator.forward(images)
    d_loss_real = loss(output,real_labels)
    real_score = output

    # train it on fake images for discrimnator

    output = discrimnator.forward(images)
    d_loss_fake = loss(output,fake_labels)

    # collect the loss and backpropagate
    d_loss = d_loss_fake + d_loss_real
    d_1 = d_loss.item()
    optimizer_dis.zero_grad()
    d_loss.backward()
    optimizer_dis.step()

    # generator
    # input a noise image and produce a fake image

    global fake_image

    global img

    img = torch.rand(batch_size,100)
    global fake_images
    fake_image = generator.forward(img)
    # test it in discrimnator

    error_dis = discrimnator.forward(fake_image.reshape(batch_size, -1))
    error = loss(error_dis,real_labels)
    value = error.item()
    # backpropagate

    optimizer_gen.zero_grad()
    error.backward()
    optimizer_gen.step()

    return value,d_1

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# train discrimnator and generator
num_epochs = 100;

Error = []

DATA_PATH_1 = '/Users/adityadubey/PycharmProjects/dsp-project/images'


for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        error1 = []
        error2 = []
        batch_size = 100
        images = images.reshape(batch_size, -1)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        #batch_size = 100

        error_gen = train_gan(images,real_labels,fake_labels,batch_size)

        error1.append(error_gen[0])
        error2.append(error_gen[1])

    error1 = numpy.array(error1)
    error2 = numpy.array(error2)
    error1.mean()
    error2.mean()

    print("epoch - no ---> {} generator-error --> {}  discrimnator --> {} ".format(epoch,error1,error2))

    # save the image
    # Save real images
    try :
        images = images.reshape(img.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(DATA_PATH_1, 'real_images-{}.png'.format(epoch)))
        imag2 = fake_image.reshape(fake_image.size(0), 1, 28, 28)
        save_image(denorm(imag2), os.path.join(DATA_PATH_1, 'fake_images-{}.png'.format(epoch)))

    except Exception as e:
        print(e)


    # Save sampled images
    #fake_images = fake_image.reshape(fake_image.size(0), 1, 28, 28)
    #save_image(denorm(fake_images), os.path.join(DATA_PATH, 'fake_images-{}.png'.format(epoch + 1)))

















