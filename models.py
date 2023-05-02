import torch 
import torch.nn as nn
from torch.nn import functional as F

def cycle_loss(real_a, cycle_a, real_b, cycle_b, padding_index):
    return F.cross_entropy(cycle_a, real_a, ignore_index=padding_index,reduction='mean') + F.cross_entropy(cycle_b, real_b, ignore_index=padding_index, reduction='mean')

#https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html

class Discriminator(nn.Module):

    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx) #, dtype=torch.int64)#.requires_grad_(False)
        #self.embedding.weight.requires_grad = False
        #self.embedding.weight.requires_grad_(False)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim, 1)  
        )

    def forward(self, input):
        x = input
        x = x @ self.embedding.weight # embeddings for output of softmax
        _,x = self.gru(x) # we just want the last hidden state
        x = self.classify(x)
        #x = torch.sigmoid(x) # want [0,1] apparently we dont want sigmoid, because for LSGAN, it encourages our samples to be close to the decision boundary
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=256):
        super(Encoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx) #, dtype=torch.int64)#.requires_grad_(False)
        
        #self.embedding.weight.requires_grad = False
        #self.embedding.weight.requires_grad_(False)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)     
        
    def forward(self, input):
        x = input # [batch_size, max_len, vocab_size]
        x = x @ self.embedding.weight # embeddings for output of softmax
        _,hidden = self.gru(x)
        return hidden
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=256):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx) #, dtype=torch.int64)#.requires_grad_(False)
        #self.embedding.weight.requires_grad = False
        #self.embedding.weight.requires_grad_(False)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True) #changed batch first from True to False
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, input, hidden):
        x = input # [batch_size]
        x = self.embedding(x) # [batch_size, embedding]
        x = torch.unsqueeze(x, dim=1) # [batch_size, 1, embedding]
        x,hidden = self.rnn(x, hidden) # [batch_size, 1, hidden_dim], # [1, batch_size, hidden_dim] 
        x = self.pred(x) # [batch_size, 1, vocab_size]
        return x, hidden

class Generator(nn.Module):
    def __init__(self, vocab_size, padding_idx, embedding_dim=256, hidden_dim=256):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(vocab_size, padding_idx, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(vocab_size, padding_idx, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    
    def forward(self, input, teacher_force_ratio=0.5):
        #input is [batch_size, sentence_len, vocab_size]
        max_len = input.shape[1]
        batch_size = input.shape[0]
        vocab_size = input.shape[2]

        hidden = self.encoder(input)

        outputs = torch.zeros(batch_size, max_len, vocab_size).cuda() #, requires_grad=False)
        max_output = torch.zeros(batch_size, max_len).cuda() #, requires_grad=False)

        decoder_input = torch.zeros(batch_size, dtype=torch.int32).cuda() #, requires_grad=False)
        max_output[:,0] = decoder_input

        for t in range(max_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden) # [batch_size, 1, vocab_size], [1, batch_size, hidden_dim] 
            outputs[:,t:t+1] = decoder_output

            argMax = torch.squeeze(decoder_output.max(-1)[1], dim=-1) #[batch_size]
            max_output[:,t] = argMax

            decoder_input = argMax

        return outputs, max_output
    

class CycleGAN(nn.Module):
        def __init__(self, vocab_size, padding_idx, mode='train', lamb=10):
            super(CycleGAN, self).__init__()
            assert mode in ["train", "A2B", "B2A"]
            self.G_A2B = Generator(vocab_size, padding_idx)
            self.G_B2A = Generator(vocab_size, padding_idx)
            self.D_A = Discriminator(vocab_size, padding_idx)
            self.D_B = Discriminator(vocab_size, padding_idx)
            self.l2loss = nn.MSELoss(reduction="mean")
            self.mode = mode
            self.lamb = lamb
            self.padding_idx = padding_idx
            self.vocab_size = vocab_size

        def forward(self, real_A, real_B):
            # blue line
            real_A_int = real_A
            real_B_int = real_B
            print(real_A)
            print(real_B)
            
            real_A = torch.nn.functional.one_hot(real_A, num_classes=(self.vocab_size)).float().cuda()
            real_B = torch.nn.functional.one_hot(real_B, num_classes=(self.vocab_size)).float().cuda()

            fake_B, fake_B_toks = self.G_A2B(real_A)
            cycle_A, cycle_A_toks = self.G_B2A(fake_B)

            # red line
            fake_A, fake_A_toks = self.G_B2A(real_B)
            cycle_B, cycle_B_toks = self.G_A2B(fake_A)
            
            fake_B, fake_B_toks = fake_B.cuda(), fake_B_toks.cuda()
            cycle_A, cycle_A_toks = cycle_A.cuda(), cycle_A_toks.cuda()

            fake_A, fake_A_toks = fake_A.cuda(), fake_A_toks.cuda()
            cycle_B, cycle_B_toks = cycle_B.cuda(), cycle_B_toks.cuda()

            if self.mode == 'train':

                DA_real = self.D_A(real_A)
                DB_real = self.D_B(real_B)

                DA_real = DA_real.cuda()
                DB_real = DB_real.cuda()

                DA_fake = self.D_A(fake_A)
                DB_fake = self.D_B(fake_B)

                DA_fake = DA_fake.cuda()
                DB_fake = DB_fake.cuda()

                # Cycle loss
                c_loss = self.lamb * cycle_loss(real_A_int, cycle_A, real_B_int, cycle_B, self.padding_idx)

                # Generator losses
                g_A2B_loss = self.l2loss(DB_fake, torch.ones_like(DB_fake)) + c_loss
                g_B2A_loss = self.l2loss(DA_fake, torch.ones_like(DA_fake)) + c_loss

                # Discriminator losses
                d_A_loss_real = self.l2loss(DA_real, torch.ones_like(DA_real))
                d_A_loss_fake = self.l2loss(DA_fake, torch.zeros_like(DA_fake))
                d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
                d_B_loss_real = self.l2loss(DB_real, torch.ones_like(DB_real))
                d_B_loss_fake = self.l2loss(DB_fake, torch.zeros_like(DB_fake))
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
                  return fake_B_toks, cycle_A_toks
            elif self.mode == 'B2A':
                  return fake_A_toks, cycle_B_toks

                    #https://github.com/Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch/blob/main/model.py      