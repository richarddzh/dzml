import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # h_out = (h_in - 1) * stride - 2 * padding + kernel_size
        # h_out = (2-1)*2 - 2*1 + 6 = 6
        self.dconv1 = nn.ConvTranspose2d(1, 4, 6, stride=2, padding=1)
        self.dconv1_bn = nn.BatchNorm2d(4)
        # h_out = (6-1)*2 - 2*1 + 6 = 14
        self.dconv2 = nn.ConvTranspose2d(4, 2, 6, stride=2, padding=1)
        self.dconv2_bn = nn.BatchNorm2d(2)
        # h_out = (14-1)*2 - 2*1 + 4 = 28
        self.dconv3 = nn.ConvTranspose2d(2, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = x.view(-1, 1, 2, 2)
        x = F.relu(self.dconv1_bn(self.dconv1(x)))
        x = F.relu(self.dconv2_bn(self.dconv2(x)))
        x = (torch.tanh(self.dconv3(x)) + 1) * 0.5
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # h_out = floor((h_in + 2 * padding - kernel_size) / stride + 1)
        # h_out = (28 + 2*1 - 4) / 2 + 1 = 14
        self.conv1 = nn.Conv2d(1, 4, 4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(4)
        # h_out = (14 + 2*1 - 6) / 2 + 1 = 6
        self.conv2 = nn.Conv2d(4, 2, 6, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(2)
        # h_out = (6 + 2*0 - 4) / 2 + 1 = 2
        self.conv3 = nn.Conv2d(2, 1, 4, stride=2, padding=0)
        self.conv3_bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, 4)
        x = torch.sigmoid(self.fc1(x))
        return x


class Trainer:
    def __init__(self):
        self.dataset = datasets.MNIST('data', download=True)
        self.netD = Discriminator()
        self.netG = Generator()
        self.optimD = optim.Adam(self.netD.parameters(), lr=0.0001)
        self.optimG = optim.Adam(self.netG.parameters(), lr=0.00006)
        self.loss = nn.BCELoss()

    def save(self):
        torch.save({
            'net_g': self.netG.state_dict(),
            'net_d': self.netD.state_dict()
        }, 'mnist-dcgan.pt')

    def load(self):
        nets = torch.load('mnist-dcgan.pt')
        self.netD.load_state_dict(nets['net_d'])
        self.netD.train()
        self.netG.load_state_dict(nets['net_g'])
        self.netG.train()

    def get_batch(self, skip, batch_size):
        x = self.dataset.train_data[skip:skip+batch_size].view(-1, 1, 28, 28).float()
        x = x / x.max()
        return x

    def train(self, skip, batch_size):
        self.optimD.zero_grad()
        x1 = self.get_batch(skip, batch_size)
        y1 = self.netD(x1)
        label1 = torch.full([batch_size, 1], 1)
        loss1 = self.loss(y1, label1)
        loss1.backward()
        x0 = torch.randn([batch_size, 4])
        x0 = self.netG(x0)
        y0 = self.netD(x0.detach())
        label0 = torch.full([batch_size, 1], 0)
        loss0 = self.loss(y0, label0)
        loss0.backward()
        self.optimD.step()
        print("D Loss 1: {0}, 0: {1}".format(loss1.item(), loss0.item()))
        self.optimG.zero_grad()
        y2 = self.netD(x0)
        loss2 = self.loss(y2, label1)
        loss2.backward()
        self.optimG.step()
        print("G Loss: {0}".format(loss2.item()))


a = Trainer()
a.load()
for j in range(100):
    for i in range(0, 50000, 40):
        print("Epoch {0}, {1}".format(j, i))
        a.train(i, 40)
a.save()

a.netG.eval()
NR = 8
NC = 10
for i in range(NC):
    for j in range(NR):
        x = torch.randn([1, 4])
        print(x)
        x = a.netG(x)
        plt.subplot(NR, NC, i * NR + j + 1)
        plt.imshow(x.data.numpy()[0][0])
        plt.axis('off')

plt.show()

