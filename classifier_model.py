import torch 
import torch.nn as nn
from torch.nn import functional as F

## DISCLAIMER: I haven't run this yet but it theoretically should work
# once we fix the todos

# sources for classifier model inspiration:
# https://music-classification.github.io/tutorial/part3_supervised/tutorial.html
# https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/scripts/models.py


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling_size=2, stride=1, padding=1):
        super(self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxUnpool2d(kernel_size=pooling_size)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes # number of classes we are predicting

        self.conv1 = ConvBlock(in_channels=1, out_channels=64, kernel_size=3, pooling_size=2, stride=1, padding=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pooling_size=2, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pooling_size=4, stride=1, padding=1)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pooling_size=4, stride=1, padding=1)

        self.dense1 = nn.Linear(in_features=2048, out_features=1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(in_features=1024, out_features=256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.dense3 = nn.Linear(in_features=256, out_features=self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("input shape", x.shape)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.softmax(x)

        return x
    
# train and test from pytorch documentation: 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def train(model):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(pop_rock_trainloader, 0): # TODO: add our train data loader here
            # get the inputs; data is a list of [inputs, labels]
            # TODO: what format will the data be in??
            inputs, labels = data 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

def test(model):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader: # TODO: replace with test data loader
            inputs, labels = data
            outputs = model(inputs)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test inputs: {100 * correct // total} %')


if __name__ == '__main__':
    classifier = Classifier(num_classes=3) # TODO: 3 for pop, jazz, classical ??
    train(classifier)
    test(classifier)
