import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear = nn.Linear(10,100)

    def forward(self):
        pass

net = Net()

optimizer = optim.SGD(net.parameters(), lr = 0.01)

print(optimizer.state_dict()['param_groups'][0]['lr'])
print(optimizer.param_groups[0]['lr'])