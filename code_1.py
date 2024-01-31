import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import matplotlib.pyplot as plt
import numpy as np

batch_size = 100

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# loading the training data and test data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # BACKBONE
        self.fc1 = nn.Linear(768,1) # first Linear layer that outputs "a" -> vector
        self.conv1 = nn.Conv2d(3,1,5, padding=2)
        self.pool1 = nn.AvgPool2d(2,None)
        
        self.fc2 = nn.Linear(256,1) # second Linear layer that outputs "a" -> vector
        self.conv2 = nn.Conv2d(1,1,5, padding=2)
        self.pool2 = nn.AvgPool2d(2,None)

        self.fc3 = nn.Linear(256,1) # third Linear layer that outputs "a" -> vector
        self.conv3 = nn.Conv2d(1,1,5, padding=2)
        self.pool3 = nn.AvgPool2d(2,None)
        
        # CLASSIFIER -> simple MLP model
        self.cpool1 = nn.AvgPool2d(2,None)
        self.cfc1 = nn.Linear(256, 73)
        self.cfc2 = nn.Linear(73, 45)
        self.cfc3 = nn.Linear(45, 10)
        
    def N(self, x, conv, fcN, pool): # function that creates every N block, takes the respective input, conv layer, linear layer and average pool
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #checks if you can run it on CUDA
        # device = 'cpu'
        
        O_1 = conv(x) # apply convolution to get "K" outputs 
        a = pool(x).reshape(inputs.shape[0] ,-1) # Get "a" with Linear layer and respective pool
        a = F.relu(fcN(a)) # activation function
        O_1_tmp = torch.zeros(O_1.shape[0],1,O_1.shape[2],O_1.shape[3], device=device) # Final "O" output, all the inputs will get added to here
        
        # loop to add the "K" outputs to produce "O"
        for batch_indx in range(O_1.shape[0]):
            for matrix_number in range(O_1.shape[1]):
                O_1[batch_indx][matrix_number]*=a[batch_indx] # multiply "K_i" with "a_i" 
                O_1_tmp[batch_indx][0]+=O_1[batch_indx][matrix_number] # add the matrices to have a single output
        
        return O_1_tmp
        
        
    def forward(self, x):

        m_1 = nn.BatchNorm2d(3).cuda()
        m_2 = nn.BatchNorm2d(1).cuda()
        m_3 = nn.BatchNorm2d(1).cuda()

        x = m_1(x)
        x = self.N(x, self.conv1, self.fc1, self.pool1) # apply first block with respective conv, pool, ...
        x = m_2(x)
        x = self.N(x, self.conv2, self.fc2, self.pool2) # apply second block with respective conv, pool, ...
        x = m_3(x)
        x = self.N(x, self.conv3, self.fc3, self.pool3)
        x = m_3(x)
        
        x = self.cpool1(x) # Classifier pool
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.cfc1(x)) # first MLP of the classifier
        x = F.relu(self.cfc2(x)) # second MLP of the classifier
        x = self.cfc3(x) # final MLP of the classifier

        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.0015
num_epochs = 11

model = Net()
model = model.to(device) # pass model to CUDA if available

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

print("training started")
start = time.time()


def my_plot(epochs, loss): # function to plot epochs and loss
    plt.plot(epochs, loss)

loss_vals=[] # array to save loss to plot it later

for epoch in range(num_epochs): #main training loop
    epoch_loss=[] # array to save loss of epoch to plot it later
    running_loss = 0.0 # to calculate the loss and how it evolves
    
    for batch_idx, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # forward
        scores = model(inputs)        
        loss = criterion(scores, labels)
        epoch_loss.append(loss.item())
        
        #backward
        optimizer.zero_grad()
        loss.backward()

        #adam step
        optimizer.step()

        running_loss += loss.item() # to calculate how the loss evolves 
        
        if batch_idx % 500 == 499:    # print every 500 mini-batches + how long it takes the batch
            end = time.time()
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 500:.3f}: {end-start:.2f}s')
            start = end
            running_loss = 0.0
            
    loss_vals.append(sum(epoch_loss)/len(epoch_loss)) # saving the loss values to plot graph
my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals) # plotting how the loss curve
 

print("Finish training")

def check_accuracy(loader, model):
    
    loss_vals=[] # array to save loss to plot it later
    
    if loader.dataset.train:
        print("Checking on training data")
    else:
        print("Checking on test data")
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        epoch_loss=[] # array to save loss of epoch to plot it later

        for x, y in loader:
            # x = x.reshape(x.shape[0],-1)
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            
            _, predictions = scores.max(1)
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        # loss_vals.append(sum(epoch_loss)/len(epoch_loss)) # saving the loss values to plot graph

    model.train()

check_accuracy(trainloader,model)
check_accuracy(testloader,model)