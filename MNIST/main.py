import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timeit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define LeNet 5 like architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.cnn = nn.Sequential(

            # Layer 1
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)

            )

        self.fc = nn.Sequential(

            # Layer 3
            nn.Linear(16*4*4, 120),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Linear(84, 10)

            )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

class Config():
    batch_size_train = 64
    batch_size_test = 1000
    training_epochs = 10

# Training written in a separate function
def train(epoch):
    global counter
    net.train()

    for i, data in enumerate(train_loader):
        input, target = data
        input, target = Variable(input).cuda(), Variable(target).cuda()


        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch {}: Loss: {}".format(epoch,loss.data[0]))
            loss_history.append(loss.data[0])
            iteration.append(counter)
            counter += 100

# Testing written in a separate function
def test():
    net.eval()

    correct = 0
    total = 0

    for i, data in enumerate(test_loader):
        input, target = data
        input, target = Variable(input).cuda(), Variable(target).cuda()


        output = net(input)
        _,predicted = torch.max(output.data,1)
        total += input.size()[0]
        #correct += (predicted == target).sum()
        correct += predicted.eq(target.data.view_as(predicted)).cpu().sum()


    print("Test accuracy: {}".format(correct/total * 100))

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_loader = DataLoader(train_dataset, batch_size = Config.batch_size_train,
                            shuffle = True, num_workers = 4)

test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

test_loader = DataLoader(test_dataset, batch_size = Config.batch_size_test,
                            shuffle = True, num_workers = 4)


# Net, Loss, Optimizer
net = LeNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

iteration = []
loss_history = []
counter = 0

# Training Loop
start_time = timeit.default_timer()
for epoch in range(0,Config.training_epochs):
    train(epoch)
    test()
elapsed = timeit.default_timer() - start_time
print("Time taken: {}".format(elapsed))


figure = plt.plot(iteration, loss_history)
plt.savefig("loss_curve.png")









