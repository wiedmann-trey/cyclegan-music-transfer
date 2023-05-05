import torch 
import torch.nn as nn
from torch.nn import functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import copy
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
        self.hidden = self.init_hidden(batch_size=32)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return [autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)), 
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))]

    def forward(self, sentence):
        
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        y = self.hidden2label(lstm_out[:, -1, :])
        probs = F.softmax(y, dim=1)

        return probs

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if torch.argmax(truth[i])==torch.argmax(pred[i]):
             right += 1.0
     return right/len(truth)

# train and test from pytorch documentation: 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def train(model, load=False, model_path='classifier_epoch6_model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LSTMClassifier(embedding_dim=256, hidden_dim=256, vocab_size=391, label_size=2)
    if load:
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    pop_jazz_train_loader, pop_jazz_test_loader = get_classifier_data(batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc=0.0
        b=0
        for i, data in enumerate(pop_jazz_train_loader):
            timeshift, timeshift_label = data['timeshift'], data['timeshift_label']
        
            with torch.autograd.set_detect_anomaly(True):
                model.hidden = model.init_hidden(batch_size=32)
                model.zero_grad()
                outputs = model(timeshift)
                loss = nn.functional.cross_entropy(outputs, timeshift_label)
                loss.backward()
                acc = get_accuracy(timeshift_label, outputs)
                running_acc+=acc
                b+=1
            optimizer.step()
            running_loss += loss.item()
            
            if i % 50 == 1: # print every 50 mini-batches
                print(f'[{epoch + 1}, {i + 1:10d}] loss: {running_loss / 50:.10f} accuracy: {running_acc / b}')
                running_loss = 0.0
        # save every epoch
        torch.save(model.state_dict(), f'pop_jazz_classifier_{epoch}.pth')
    print('Finished Training')

def test(model):
    pop_jazz_train_loader, pop_jazz_test_loader = get_classifier_data()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in pop_jazz_test_loader:
            timeshift, timeshift_label = data['timeshift'], data['timeshift_label']
            outputs = model(timeshift)
            _, predicted = torch.max(outputs.data, 1)
            total += timeshift_label.size(0)

            total += timeshift_label.size(0)
            correct += (predicted == timeshift).sum().item()

    print(f'Accuracy of the network on the {total} test inputs: {100 * correct // total} %')


if __name__ == '__main__':
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    EPOCH = 10
    classifier = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=391,label_size=2)
    train(classifier)
    test(classifier)
