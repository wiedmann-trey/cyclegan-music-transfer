import torch 
import torch.nn as nn
from torch.nn import functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import copy
## DISCLAIMER: I haven't run this yet but it theoretically should work
# once we fix the todos
from datasets import get_classifier_data

# sources for classifier model inspiration:
# https://music-classification.github.io/tutorial/part3_supervised/tutorial.html
# https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/scripts/models.py
#https://github.com/yuchenlin/lstm_sentence_classifier/blob/master/LSTM_sentence_classifier.py 

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, dtype=torch.float)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = 0 #self.init_hidden(batch_size=32)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return [autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)), 
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))]

    def forward(self, sentence):
        # batch_size = sentence.size(0)
        # embeds = self.word_embeddings(sentence)
        # x = embeds.view(batch_size, -1, self.embedding_dim)
        # x = torch.reshape(x, (-1, 32, 50))
        # print(x.shape)
        # print(self.hidden)
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        # y  = self.hidden2label(lstm_out[-1])
        # log_probs = F.log_softmax(y)
        # return log_probs
        batch_size = sentence.size(0)
        embeds = self.word_embeddings(sentence)
        #print(embeds.shape)
        hidden = self.init_hidden(32)
        #print("here 49")
        lstm_out, self.hidden = self.lstm(embeds, hidden)
        #print("51")
        y = self.hidden2label(lstm_out[:, -1, :])
        log_probs = F.log_softmax(y, dim=1)
        #print(log_probs.shape)
        return log_probs


def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

# train and test from pytorch documentation: 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def train(model):
    
    best_dev_acc = 0.0
    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)

    pop_jazz_train_loader, pop_jazz_test_loader = get_classifier_data(batch_size=32)
    #loss_func = nn.CrossEntropyLoss()
    #loss_func = nn.functional.cross_entropy()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(pop_jazz_train_loader):
            timeshift, timeshift_label = data['timeshift'], data['timeshift_label']
        
            with torch.autograd.set_detect_anomaly(True):
                outputs = model(timeshift)
                #timeshift_label = torch.squeeze(timeshift_label)
                loss = nn.functional.cross_entropy(outputs, timeshift_label)
                loss.backward(retain_graph=True)
            
                optimizer.step()
            running_loss += loss.item()

            # print statistics
            if i % 100 == 1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:10d}] loss: {running_loss / 100:.10f}')
                running_loss = 0.0
                running_acc = 0.0
        
        # save every epoch
        torch.save(model.state_dict(), f'classifier_epoch{epoch}_model.pth')

    print('Finished Training')

def test(model):
    pop_jazz_train_loader, pop_jazz_test_loader = get_classifier_data()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in pop_jazz_test_loader:
            timeshift, timeshift_label = data['timeshift'], data['timeshift_label']
            #timeshift = torch.nn.functional.one_hot(timeshift, num_classes=(391)).float()
            
            outputs = model(timeshift)#[1]
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += timeshift_label.size(0)

            total += timeshift_label.size(0)
            correct += (predicted == timeshift_label).sum().item()

    print(f'Accuracy of the network on the {total} test inputs: {100 * correct // total} %')


if __name__ == '__main__':
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 100
    classifier = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=391,label_size=2)
    train(classifier)
    test(classifier)
