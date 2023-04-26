import torch 
import torch.nn as nn
from torch.nn import functional as F

def cycle_loss(real_a, cycle_a, real_b, cycle_b):
    return F.cross_entropy(cycle_a, real_a, reduction='mean') + F.cross_entropy(cycle_b, real_b, reduction='mean')

#https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html

class Discriminator(nn.Module):

    def __init__(self, vocab_size=387, embedding_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        vocab_size=int(vocab_size)
        embedding_dim=int(vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, dtype=torch.int64)
        self.embedding.weight.requires_grad=False
        self.embedding.weight.requires_grad_=False
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim, 1)  
        )

    def forward(self, input):
        x = input
        x = x @ self.embedding.weight # embeddings for output of softmax
        _,x = self.gru(x) # we just want the last hidden state
        x = self.classify(x)
        x = torch.sigmoid(x) # want [0,1]
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size=387, embedding_dim=256, hidden_dim=256, dropout_rate=.1):
        super(Encoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        vocab_size=int(vocab_size)
        embedding_dim=int(vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, dtype=torch.int64)
        self.embedding.weight.requires_grad_=False
        self.embedding.weight.requires_grad=False
        self.gru = nn.GRU(embedding_dim, hidden_dim, dropout=dropout_rate, batch_first=True)     

    def forward(self, input):
        x = input
        #x = torch.reshape(x, ((int(x.size(dim=1)/self.embedding_dim)), self.embedding_dim))
        

        # convert indices to LongTensor
        x = torch.LongTensor(x)
        print(type(self.embedding.weight))
        print(type(x))
        x = x @ self.embedding.weight # embeddings for output of softmax

        _,hidden = self.gru(x)
        return hidden
    
class Decoder(nn.Module):
    def __init__(self, vocab_size=387, embedding_dim=256, hidden_dim=256, dropout_rate=.1):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, dtype=torch.int64)
        self.embedding.weight.requires_grad=False
        self.embedding.weight.requires_grad_=False
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout_rate)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, vocab_size, dtype=torch.float),
            nn.Softmax()
        )

    def forward(self, input, hidden):
        x = input # [batch_size]
        x = self.embedding(x) # [batch_size, embedding]
        x,hidden = self.gru(x, hidden)
        x = self.predict(x) # [batch_size, vocab_size]
        return x, hidden

class Generator(nn.Module):
    def __init__(self, vocab_size=387, embedding_dim=256, hidden_dim=256, dropout_rate=.1):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.decoder = Decoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    
    def forward(self, input, teacher_force_ratio=0.5):
        #input is [batch_size, sentence_len, vocab_size]
        max_len = input.shape[1]
        batch_size = input.shape[0]
        vocab_size = input.shape[2]

        hidden = self.encoder(input)

        outputs = torch.zeros(batch_size, max_len, vocab_size, requires_grad=False)
        max_output = torch.zeros(batch_size, max_len, requires_grad=False)

        decoder_input = torch.zeros(batch_size, requires_grad=False)
        max_output[:,0] = decoder_input

        for t in range(max_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs[:,t] = decoder_output

            argMax = decoder_output.max(1)[1]
            max_output[:,t] = argMax

            decoder_input = argMax

        return outputs, max_output
    

class CycleGAN(nn.Module):
        def __init__(self, mode='train', lamb=10):
            super(CycleGAN, self).__init__()
            assert mode in ["train", "A2B", "B2A"]
            self.G_A2B = Generator()
            self.G_B2A = Generator()
            self.D_A = Discriminator()
            self.D_B = Discriminator()
            self.l2loss = nn.MSELoss(reduction="mean")
            self.mode = mode
            self.lamb = lamb

        def forward(self, real_A, real_B):
            # blue line
            fake_B = self.G_A2B(real_A)
            cycle_A = self.G_B2A(fake_B)

            # red line
            fake_A = self.G_B2A(real_B)
            cycle_B = self.G_A2B(fake_A)

            if self.mode == 'train':

                DA_real = self.D_A(real_A)
                DB_real = self.D_B(real_B)

                DA_fake = self.D_A(fake_A)
                DB_fake = self.D_B(fake_B)

                # Cycle loss
                c_loss = self.lamb * cycle_loss(real_A, cycle_A, real_B, cycle_B)

                # Generator losses
                g_A2B_loss = self.l2loss(DB_fake, torch.ones_like(DB_fake, requires_grad=False)) + c_loss
                g_B2A_loss = self.l2loss(DA_fake, torch.ones_like(DA_fake, requires_grad=False)) + c_loss

                # Discriminator losses
                d_A_loss_real = self.l2loss(DA_real, torch.ones_like(DA_real, requires_grad=False))
                d_A_loss_fake = self.l2loss(DA_fake, torch.zeros_like(DA_fake, requires_grad=False))
                d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
                d_B_loss_real = self.l2loss(DB_real, torch.ones_like(DB_real, requires_grad=False))
                d_B_loss_fake = self.l2loss(DB_fake, torch.zeros_like(DB_fake, requires_grad=False))
                d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2

                # All losses
                '''
                d_A_all_loss_real = self.l2loss(DA_real, torch.ones_like(DA_real))
                d_A_all_loss_fake = self.l2loss(DA_fake, torch.zeros_like(DA_fake))
                d_A_all_loss = (d_A_all_loss_real + d_A_all_loss_fake) / 2
                d_B_all_loss_real = self.l2loss(DB_real, torch.ones_like(DB_real))
                d_B_all_loss_fake = self.l2loss(DB_fake, torch.zeros_like(DB_fake))
                d_B_all_loss = (d_B_all_loss_real + d_B_all_loss_fake) / 2
                '''
                return (c_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss)
                    #d_A_all_loss, d_B_all_loss)

            elif self.mode == 'A2B':
                  return fake_B, cycle_A
            elif self.mode == 'B2A':
                  return fake_A, cycle_B

                    #https://github.com/Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch/blob/main/model.py      