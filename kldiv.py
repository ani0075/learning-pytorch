import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#AlexNet model. Borrowed from torchvision/models.
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x                                                #return a vector of size num_classes

class My_KLDivLoss(nn.Module):

    def __init__(self):
        super(My_KLDivLoss,self).__init__()


    def forward(self,input1,input2):

        #RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays).
        #Use .cpu() to move the tensor to host memory first.
        a = input1.cpu().data.numpy()
        b = input2.cpu().data.numpy()
        numpy_loss = np.sum(np.where(a != 0, a * np.log(a / b), 0))

        torch_loss = torch.from_numpy(np.array([numpy_loss]))
        return Variable(torch_loss, requires_grad=True)


model = AlexNet().cuda()
#model.eval()
input1 = Variable(torch.randn(2,3,224,224)).cuda()
input2 = Variable(torch.randn(2,3,224,224)).cuda()

print('Difference between two inputs', torch.sum(input1-input1))
output1 = model(input1)
output2 = model(input1)

print((output1 - output2).abs().max())

print('Difference between two outputs', torch.sum(output1-output2))
dist1 = F.softmax(output1,dim=1)
dist2 = F.softmax(output2,dim=1).detach()

print('Difference between two output sm', torch.sum(dist1-dist2))
criterion = nn.KLDivLoss()
loss = criterion(dist1.log(),dist2)

print(loss)

# dist1 = Variable(torch.Tensor([[0.25, 0.75, 0.0],[0.3, 0.2, 0.5]])).cuda() #F.softmax(output1,dim=1)
# dist2 = Variable(torch.Tensor([[0.3, 0.2, 0.5],[0.5, 0.4, 0.1]])).cuda() #F.softmax(output2,dim=1)
# print(dist1,dist2)
# criterion_new = My_KLDivLoss().cuda()
# loss = criterion_new(dist1,dist2)

# print(loss)