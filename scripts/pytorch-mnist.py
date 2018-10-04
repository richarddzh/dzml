import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(5 * 5 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, 1)
        return x


class MyTrainer:
    def __init__(self):
        self.dataset = datasets.MNIST('data', download=True)
        self.net = MnistNet()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1)

    def get_batch(self, skip, batch_size):
        x = self.dataset.train_data[skip:skip+batch_size].view(-1, 1, 28, 28).float()
        x = x / x.max()
        y = self.dataset.train_labels[skip:skip+batch_size]
        return x, y

    def train(self, skip, batch_size):
        self.optimizer.zero_grad()
        x, y = self.get_batch(skip, batch_size)
        output = self.net(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        self.optimizer.step()
        print('Loss: {:.6f}'.format(loss.item()))

    def save(self):
        torch.save(self.net.state_dict(), 'mnist.pt')

    def load(self):
        self.net.load_state_dict(torch.load('mnist.pt'))
        self.net.train()


a = MyTrainer()
a.load()
for i in range(1):
    for j in range(0, 50000, 20):
        print('Epoch: {0}'.format(i))
        a.train(j, 20)
a.save()

tidx = 50021
img = a.dataset.train_data[tidx]
result = a.net(a.get_batch(tidx, 1)[0])
print(result)
plt.imshow(img)
plt.show()
