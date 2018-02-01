import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader    # for Dataset and DataLoader class
import torch.optim
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image                               # Python Imaging Library Image class
import PIL.ImageOps                                 # for invert() function
import random
import matplotlib
matplotlib.use('Agg')                               # for server scripts without display
import matplotlib.pyplot as plt
import numpy as np
import timeit

# Define the Siamese Network
# The forward function takes an input pair and outputs a pair
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(

            #Layer 1
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=0.2),

            #Layer 2
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2),

            #Layer 3
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2),

            )

        self.fc = nn.Sequential(

            #Layer 4
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            #Layer 5
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            #Layer 6
            nn.Linear(500, 5)

            )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Define Contrastive Loss
# It differentiates between a pair of images
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance,2) +
            (label)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2))

        return loss_contrastive

# Define Siamese Network Dataset custom class which returns a tuple of (image1, image2, Y); Y = 0 or 1
# For a custom dataset class, define the __len__ and __getitem__ functions
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #we need to make sure that approx 50% images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break

        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")                    # 'L' is for conversion to grayscale
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)        # negate the image
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img0_tuple[1]!=img1_tuple[1])],dtype=np.float32))


    def __len__(self):
        return len(self.imageFolderDataset.imgs)

# Configuration class to manage some parameters of traning/testing
class Config():
    training_dir = "./orl_faces/train"
    testing_dir = "./orl_faces/test"
    batch_size = 64
    train_number_epochs = 100

def imshow(img,index,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imsave("test"+str(index)+".png",np.transpose(npimg, (1, 2, 0)))
    #plt.show()

def main():

    # # Create siamese dataset
    # # torchvision.datasets.ImageFolder takes in a folder where images are organized into class folders
    # # returns tuples of (image, class) in imgs
    # folder_dataset = datasets.ImageFolder(root=Config.training_dir)

    # # returns tuples of (image1, image2, Y)
    # siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform = transforms.Compose([
    #                                                                                 transforms.Resize((100, 100)),
    #                                                                                 transforms.ToTensor()]),
    #                                                                                 should_invert = False)

    # # the DataLoader object takes in the dataset object and other parameters, loads data during training/testing
    # train_dataloader = DataLoader(siamese_dataset, batch_size = Config.batch_size,
    #                                 shuffle=True, num_workers=4)

    # net = SiameseNetwork().cuda()                                     # instantiate network, pass to GPU
    # criterion = ContrastiveLoss()                                   # instantiate loss criterion
    # optimizer = torch.optim.Adam(net.parameters(), lr = 0.0005)     # instantiate optimizer

    # # for plotting
    # loss_history = []
    # iteration = []
    # counter = 0

    # start_time = timeit.default_timer()
    # # training loop
    # for epoch in range(0,Config.train_number_epochs):
    #     for i, data in enumerate(train_dataloader):
    #         input1, input2, Y = data
    #         input1, input2, Y = Variable(input1).cuda(), Variable(input2).cuda(), Variable(Y).cuda()

    #         optimizer.zero_grad()
    #         output1, output2 = net(input1, input2)
    #         loss = criterion(output1, output2, Y)
    #         loss.backward()
    #         optimizer.step()

    #         if i % 10 == 0:
    #             print('Epoch {} completed: Loss {}\n'.format(epoch, loss.data[0]))
    #             loss_history.append(loss.data[0])
    #             iteration.append(counter)
    #             counter += 10

    # elapsed = timeit.default_timer() - start_time
    # print('Finished training. Time taken: %.3f seconds' % elapsed)

    # # plot training curve
    # figure = plt.plot(iteration, loss_history)
    # plt.savefig("loss_curve.jpg")

    # # save model
    # print("=> saving the model")
    # state = {
    #     'epoch' : epoch + 1,
    #     'arch'  : 'custom_oneshotlearning',
    #     'state_dict' : net.state_dict(),
    #     #'optimizer' : optimizer.state_dict(),
    #     }
    # torch.save(state, "net.pth.tar")
    # print("=> model saved in net.pth.tar file")

    net = SiameseNetwork().cuda()

    # load a saved model
    print("=> loading saved model")
    checkpoint = torch.load('./net.pth.tar')
    arch = checkpoint['arch']
    epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    print("=> loaded saved network: architecture {} epoch {}".format(arch,epoch))

    folder_dataset_test = datasets.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                           ,should_invert=False)

    test_dataloader = DataLoader(siamese_dataset,num_workers=4,batch_size=1,shuffle=True)
    dataiter = iter(test_dataloader)
    x0,_,_ = next(dataiter)

    # testing loop
    for i in range(10):
        _,x1,label2 = next(dataiter)
        concatenated = torch.cat((x0,x1),0)

        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), i,'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))





if __name__ == '__main__':
    main()
