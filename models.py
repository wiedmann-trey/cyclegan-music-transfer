import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def cycle_loss(real_a, cycle_a, real_b, cycle_b, padding_index):
    return F.nll_loss(torch.log(cycle_a), real_a, reduction='mean') + F.nll_loss(torch.log(cycle_b), real_b, reduction='mean')

def acc(real_a, cycle_a, real_b, cycle_b, padding_index):
    acc_a = torch.sum(torch.logical_and(real_a != padding_index,  real_a==cycle_a).float())/torch.sum((real_a != padding_index).float())
    acc_b = torch.sum(torch.logical_and(real_b != padding_index,  real_b==cycle_b).float())/torch.sum((real_b != padding_index).float())
    return acc_a, acc_b
#https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html

class Discriminator(nn.Module):

    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)#, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim, 1)  
        )

    def forward(self, input):
        x = input
        x = x @ self.embedding.weight
        _,x = self.gru(x) # we just want the last hidden state
        x = self.classify(x)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=512):
        super(Encoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)#, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True)     
        
    def forward(self, input):
        x = input # [batch_size, max_len, vocab_size]
        x = x @ self.embedding.weight
        _,hidden = self.gru(x)
        return hidden
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=512):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)#, padding_idx=padding_idx) 
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True) 
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=.1),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, input, hidden):
        x = input # [batch_size]
        x = self.embedding(x) # [batch_size, embedding]
        x = torch.unsqueeze(x, dim=1) # [batch_size, 1, embedding]
        x,hidden = self.rnn(x, hidden) # [batch_size, 1, hidden_dim], # [1, batch_size, hidden_dim] 
        x = self.pred(x) # [batch_size, 1, vocab_size]
        return x, hidden

class Generator(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=512, pretrain=False):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(vocab_size, padding_idx, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(vocab_size, padding_idx, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pretrain = pretrain
    def forward(self, input, temp=1, teach_force_ratio=1):
        max_len = input.shape[1]
        batch_size = input.shape[0]
        vocab_size = input.shape[2]
        input_toks = input.max(-1)[1]
        hidden = self.encoder(input)

        if self.device == 'cuda':
            outputs = torch.zeros(batch_size, 0, vocab_size, requires_grad=True).cuda() 
            max_output = torch.zeros(batch_size, max_len).cuda() 

            decoder_input = 388*torch.ones(batch_size, dtype=torch.int64).cuda()
        else: 
            outputs = torch.zeros(batch_size, 0, vocab_size, requires_grad=True) 
            max_output = torch.zeros(batch_size, max_len)

            decoder_input = 388*torch.ones(batch_size, dtype=torch.int64) 

        max_output[:,0] = decoder_input

        start_token = torch.nn.functional.one_hot(decoder_input, num_classes=vocab_size).float().reshape((batch_size, 1, vocab_size))
        outputs = torch.cat([outputs, start_token], dim=1)
        
        for t in range(1,max_len):
            
            decoder_output, hidden = self.decoder(decoder_input, hidden) # [batch_size, 1, vocab_size], [1, batch_size, hidden_dim] 
            decoder_output = F.softmax(decoder_output/temp, dim=-1)
            
            outputs = torch.cat([outputs, decoder_output], dim=1)
            
            samples = torch.squeeze(decoder_output, dim=1)
            samples = torch.multinomial(samples, 1, True)
            samples = torch.squeeze(samples, dim=-1)
            decoder_input = samples 
            max_output[:, t] = samples

            if self.pretrain and np.random.uniform() < teach_force_ratio:
                decoder_input = input_toks[:,t]
            
        return outputs, max_output
    

class CycleGAN(nn.Module):
        def __init__(self, vocab_size, padding_idx, mode='train', lamb=10):
            super(CycleGAN, self).__init__()
            assert mode in ["train", "A2B", "B2A", "pretrain"]
            self.G_A2B = Generator(vocab_size, padding_idx, pretrain=(mode == "pretrain"))
            self.G_B2A = Generator(vocab_size, padding_idx, pretrain=(mode == "pretrain"))
            self.D_A = Discriminator(vocab_size, padding_idx)
            self.D_B = Discriminator(vocab_size, padding_idx)
            self.l2loss = nn.MSELoss(reduction="mean")
            self.mode = mode
            self.lamb = lamb
            self.padding_idx = padding_idx
            self.vocab_size = vocab_size

        def pretrain(self, real_A, real_B):
            real_A_int = real_A
            real_B_int = real_B
            
            real_A = torch.nn.functional.one_hot(real_A, num_classes=self.vocab_size).float()
            real_B = torch.nn.functional.one_hot(real_B, num_classes=self.vocab_size).float()

            fake_B, guesses_B = self.G_A2B(real_A)
            fake_A, guesses_A = self.G_B2A(real_B)
            acc_a, acc_b = acc(real_A_int, guesses_B, real_B_int, guesses_A, self.padding_idx)

            fake_A = torch.permute(fake_A, (0, 2, 1))
            fake_B = torch.permute(fake_B, (0, 2, 1))
            
            return cycle_loss(real_A_int, fake_B, real_B_int, fake_A, self.padding_idx), acc_a, acc_b

        def forward(self, real_A, real_B):
            
            real_A_int = real_A.clone().detach()
            real_B_int = real_B.clone().detach()
            
            real_A = torch.nn.functional.one_hot(real_A, num_classes=self.vocab_size).float()
            real_B = torch.nn.functional.one_hot(real_B, num_classes=self.vocab_size).float()

            fake_B, fake_B_toks = self.G_A2B(real_A)
            cycle_A, cycle_A_toks = self.G_B2A(fake_B)

            fake_A, fake_A_toks = self.G_B2A(real_B)
            cycle_B, cycle_B_toks = self.G_A2B(fake_A)
            
            if self.mode == 'train':

                DA_fake = self.D_A(fake_A)
                DB_fake = self.D_B(fake_B)

                # Cycle loss
                cycle_A = torch.permute(cycle_A, (0, 2, 1))
                cycle_B = torch.permute(cycle_B, (0, 2, 1))

                c_loss = self.lamb * cycle_loss(real_A_int, cycle_A, real_B_int, cycle_B, self.padding_idx)
                
                # Generator losses
                g_A2B_loss = self.l2loss(DB_fake, torch.ones_like(DB_fake)) + c_loss
                g_B2A_loss = self.l2loss(DA_fake, torch.ones_like(DA_fake)) + c_loss
                
                # Discriminator losses
                DA_real = self.D_A(real_A)
                DB_real = self.D_B(real_B)
                
                fake_A = fake_A.clone().detach()
                fake_B = fake_B.clone().detach()
                
                d_A_loss_real = self.l2loss(DA_real, torch.ones_like(DA_real))
                d_A_loss_fake = self.l2loss(DA_fake, torch.zeros_like(DA_fake))
                d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
                d_B_loss_real = self.l2loss(DB_real, torch.ones_like(DB_real))
                d_B_loss_fake = self.l2loss(DB_fake, torch.zeros_like(DB_fake))
                d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2

                return (c_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss)
                    
            elif self.mode == 'A2B':
                  return fake_B_toks, cycle_A_toks
            elif self.mode == 'B2A':
                  return fake_A_toks, cycle_B_toks

#https://github.com/Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch/blob/main/model.py      