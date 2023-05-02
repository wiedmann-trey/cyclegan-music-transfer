import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
## DISCLAIMER: I haven't run this yet but it theoretically should work
# once we fix the todos
from datasets import get_classifier_data

# sources for classifier model inspiration:
# https://music-classification.github.io/tutorial/part3_supervised/tutorial.html
# https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/scripts/models.py


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling_size=2, stride=1, padding=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        #self.max_pool = nn.MaxUnpool2d(kernel_size=pooling_size, stride=stride, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=pooling_size)#, stride=stride, padding=0)

    def forward(self, x):
        print(x.shape)
        #x = x.expand(-1, -1, 391)
        x = self.conv_layer(x)
        print(27)
        print(x.shape)
        x = x[None, :]
        print(30)
        print(x.shape)
        x = torch.squeeze(x, dim=1)
        print(33)
        print(x.shape)
        #x.expand(-1, 32)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes # number of classes we are predicting

        self.conv1 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, pooling_size=2, stride=1, padding=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pooling_size=2, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pooling_size=4, stride=1, padding=1)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pooling_size=4, stride=1, padding=1)

        self.dense1 = nn.Linear(in_features=2048, out_features=1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=1024, out_features=32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.dense3 = nn.Linear(in_features=12, out_features=self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("input shape", x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.reshape(x, (-1, 2048))
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        print(77)
        print(x.shape)
        x = torch.transpose(x, dim0=0, dim1=1)
        x = self.dense3(x)
        print(x.shape)
        x = self.softmax(x)
        print(x.shape)
        return x
    
# train and test from pytorch documentation: 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def train(model):
    pop_jazz_train_loader, pop_jazz_test_loader = get_classifier_data()
    #loss_func = nn.CrossEntropyLoss()
    #TORCH.NN.FUNCTIONAL.CROSS_ENTROPY
    #loss_func = nn.functional.cross_entropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(pop_jazz_train_loader): # TODO: add our train data loader here
            # get the inputs; data is a list of [inputs, labels]
            # TODO: what format will the data be in??
            timeshift, timeshift_label = data['timeshift'], data['timeshift_label']
            timeshift = torch.nn.functional.one_hot(timeshift, num_classes=(391)).float()
            
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(timeshift)
            #outputs=torch.squeeze(outputs)
            print(outputs.shape)
            print(timeshift_label.shape)
            timeshift_label = torch.squeeze(timeshift_label)
            loss = nn.functional.cross_entropy(outputs, timeshift_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
            # print statistics
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

def test(model):
    pop_jazz_train_loader, pop_jazz_test_loader = get_classifier_data()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in pop_jazz_test_loader: # TODO: replace with test data loader
            timeshift, timeshift_label = data['timeshift'], data['timeshift_label']
            timeshift = torch.nn.functional.one_hot(timeshift, num_classes=(391)).float()
            outputs = model(timeshift)#[1]
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += timeshift_label.size(0)

            total += timeshift_label.size(0)
            correct += (predicted == timeshift).sum().item()

    print(f'Accuracy of the network on the {total} test inputs: {100 * correct // total} %')


if __name__ == '__main__':
    classifier = Classifier(num_classes=2) # TODO: 3 for pop, jazz, classical ??
    train(classifier)
    test(classifier)