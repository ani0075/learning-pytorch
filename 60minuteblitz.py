#Follow this tutorial - http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

import torch                        #import the torch package
from torch.autograd import Variable #import Variable class
import torch.nn as nn               #module to create neural networks
import torch.nn.functional as F
import torch.optim as optim         #optimizer package
import torchvision                  #avoid writing boilerplate code
import torchvision.transforms as transforms     #transform image data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import timeit
from tensorboard_logger import configure, log_value

configure("runs/run-1234", flush_secs=5)

# # What is Pytorch?

# ''' Tensors
# Tensors are similar to numpy’s ndarrays, with the addition being that
# Tensors can also be used on a GPU to accelerate computing. '''


# # Tensor() by default creates FloatTensor()
# a = torch.Tensor(5,3)       #create a tensor with 5 rows and 3 columns
# print(a)


# # Create Tensor from random matrix
# a = torch.rand(5,3)
# print(a.size())             #size() prints size of the Tensor


# '''Operations
# Multiple syntaxes for operations'''

# b = torch.rand(5,3)
# print(a+b)                  # use operator symbol

# print(torch.add(a,b))       # use .add() torch function

# result = torch.Tensor(5,3)
# torch.add(a,b,out=result)   #use .add() and store result in new tensor
# print(result)

# b.add_(a)                   # in-place operations
# print(b)

# # any operation that mutates a tensor in-place is post-fixed with an _

# print(a[:,1])               # use standard numpy like indexing with all bells and whistles; all rows and second column

# '''Numpy bridge
# Changing torch Tensors to numpy arrays and vice versa
# Share underlying memory locations and changing one will change the other'''

# a = torch.ones(5,3)           # create tensor with all ones

# b = a.numpy()               # convert to numpy array
# print(b)

# a.add_(1)                   # mutate Tensor
# print(a)
# print(b)

# #Similarly conversion from numpy array to torch tensor is possible

# '''CUDA tensors
# Tensors can be moved onto the GPU using .cuda() function'''

# if torch.cuda.is_available():
#     a = a.cuda()
#     b = b.cuda()
#     a+b                             # this doesn't work???

# '''
# a = torch.Tensor(5,3)
# a = a.cuda()
# print(a)                            # doesn't work. Couldn't print cuda Tensor directly.
# '''


''' Read documentation of Tensors'''

''' Autograd
Automatic differentiation for all operations on Tensors
define-by-run framework - each iteration can be different '''

''' Autograd Variable
Raw tensor in .data
Gradient in .grad
Creator function in .grad_fn; grad_fn = None if Leaf Variable (created by user)
To calculate gradients use .backward();  Specify grad_output tensor of matching shape
'''

# a = Variable(torch.ones(2,2), requires_grad=True)
# #create Variable and set requires_grad = True; if not set to True all Variables created from this one will have requires_grad = False
# #error in back-propagation
# print(a)

# '''
# When might I want to use requires_grad=True on an input?
# e.g. when you are optimizing the input image (Neural Style Transfer); when you are optimizing that variable itself.
# https://discuss.pytorch.org/t/when-might-i-want-to-use-requires-grad-true-on-an-input/9624
# '''

# b = a + 2
# print(b)

# print(b.grad_fn)            # find out creator function

# c = b * b * 3
# out = c.mean()

# print(c,out)

# '''Gradients'''

# out.backward()              # back-propagation; equivalent to out.backward(torch.Tensor([1.0]))

# print(a.grad)               # print d(out)/da

# #Another example
# x = Variable(torch.rand(3), requires_grad=True)
# y = x * 2
# while y.data.norm()<1000:
#     y = y * 2

# print(x,y)

# gradients = torch.FloatTensor([0.1,1,0.0001])
# y.backward(gradients)
# print(x.grad)

''' Read documentation of Variable and Function '''

'''Neural Networks
Major steps:
1. Define the network which has some weights(learnable parameters)
2. Read inputs
3. Process the inputs - pass them through network
4. Calculate loss
5. Propagate gradients back
6. Update the weights
'''

'''
.view() function
https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch
'''

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 1 input image channel, 6 output channels, 5x5 square convolution kernel
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6 input image channel, 16 output channels, 5x5 square convolution kernel

#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)


#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))   # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)       # If the size is a square you can only specify a single number
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]           # all dimensions except the batch dimension; remember size() not subscriptable error
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

# net = Net()
# print(net)

# #define the forward function and the backward function is automatically defined by autograd

# #the learnable parameters of the net are returned by net.parameters()

# params = list(net.parameters())         # returns conv weights and bias
# print(len(params))                      # 5 * 2 = 10
# for i in range(len(params)):
#     print(params[i].size())             #size of weights of each layer


# # the input to the forward() is an autograd Variable and so is the output. Expected input size to this Net(LeNet) is 32x32

# input = Variable(torch.randn(1, 1, 32, 32))         # input Tensor
# output = net(input)                                 # pass through net
# print(output)

# #net.zero_grad()                                     # reset all gradients
# #output.backward(torch.randn(1,10))                  # back-propagate


# # loss function; takes (output,target) pair of inputs and returns loss

# target = Variable(torch.arange(1,11))               # dummy target variable
# criterion = nn.MSELoss()

# loss = criterion(output,target)
# print(loss)

# # backprop

# net.zero_grad()                                     # reset all gradients

# print('Conv1 bias grad before back-prop')
# print(net.conv1.bias.grad)
# loss.backward()                                     # back-propagate

# '''RuntimeError: Trying to backward through the graph a second time,
# but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.'''

# print('Conv1 bias grad after back-prop')
# print(net.conv1.bias.grad)

# # optimizer
# optimizer = optim.SGD(net.parameters(), lr = 0.01)

# #in the training loop
# optimizer.zero_grad()
# output = net(input)
# loss = criterion(output,target)
# loss.backward()
# optimizer.step()

'''
Training a classifier
'''
# torchvision package created specifically for vision which has dataloaders for standard datasets
# data transformers

#For this tutorial, we will use the CIFAR10 dataset.
#It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
#The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

'''
Training an image classifier has the following steps:
1. Load and normalize CIFAR-10 training and test data using torchvision
2. Define the CNN
3. Define the loss function
4. Train on training data
5. Test on training data
'''

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
Generating matplotlib graphs without a running X server
https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
'''

#function to save the image
def imshow(img):
    img = img/2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imsave('save.png',np.transpose(npimg, (1, 2, 0)))

#get next batch of images
dataiter = iter(trainloader)
images,labels = dataiter.next()

#save images
#imshow(torchvision.utils.make_grid(images))

# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#define the CNN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)   # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6 input image channel, 16 output channels, 5x5 square convolution kernel

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))   # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)       # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]           # all dimensions except the batch dimension; remember size() not subscriptable error
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net().cuda()

#define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

start_time = timeit.default_timer()
#train the network
for epoch in range(2):                                      # number of epochs
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):                 # use enumerate to loop through trainloader

        inputs, labels = data                               # extract one batch of training data

        inputs,labels = Variable(inputs.cuda()), Variable(labels.cuda())  # wrap in Variables to pass through net

        optimizer.zero_grad()                               # flush gradients
        outputs = net(inputs)                               # pass through network
        loss = criterion(outputs,labels)                    # calculate loss
        loss.backward()                                     # back-propagate gradients
        #comment this line in neural network training to not propagate and hence not update weights
        optimizer.step()                                    # update weights

        log_value('loss', loss, i)
        #log_value('v2', v2, step)
        running_loss += loss.data[0]

        if i % 2000 == 1999:
            print('[epoch: %d iteration: %5d] : loss %.3f' %(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
            #print(net.conv1.weight.data[0,0,0,0])
elapsed = timeit.default_timer() - start_time
print('Finished training. Time taken: %.3f seconds' % elapsed)


# #get next batch of test images
# dataiter = iter(testloader)
# images,labels = dataiter.next()

# #save images
# imshow(torchvision.utils.make_grid(images))

# # print labels
# print('Groundtruth',' '.join('%5s' % classes[labels[j]] for j in range(4)))

# # pass through net
# outputs = net(Variable(images.cuda()))

# #print(outputs)

# # find prediction indices
# _,predicted = torch.max(outputs.data,1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))


# # performance on full test data
# correct = 0
# total = 0
# for data in testloader:
#     inputs, labels = data
#     outputs = net(Variable(inputs.cuda()))
#     _,predicted = torch.max(outputs.data,1)
#     total += inputs.size(0)
#     correct += (predicted == labels).sum()

# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# #performance on each class
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
#     inputs, labels = data
#     outputs = net(Variable(inputs.cuda()))
#     _,predicted = torch.max(outputs.data,1)

#     c = (predicted == labels).squeeze()
#     for i in range(4):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1

# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))