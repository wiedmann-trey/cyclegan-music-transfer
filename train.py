from models import CycleGAN
import torch
from datasets import get_data
import copy 

def pretrain(epochs=35, vocab_size=391, save=True, load=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pop_rock_train_loader, pop_rock_test_loader = get_data()
    model = CycleGAN(vocab_size, vocab_size-1, mode='pretrain')
    if load:
        model = model.to(device)
        model.load_state_dict(torch.load("1_pretrain_pop_jazz.pth", map_location=device))
    model = model.to(device)
    
    opt_G_A2B = torch.optim.Adam(model.G_A2B.parameters())#, weight_decay=1e-4)
    opt_G_B2A = torch.optim.Adam(model.G_B2A.parameters())#, weight_decay=1e-4)
    b = 2
    for epoch in range(epochs):
        model.train()
        print(f"pretrain epoch:{epoch}")
        total_loss = 0
        total_acc_a = 0
        total_acc_b = 0
        num_batch = 0
        for i, data in enumerate(pop_rock_train_loader):
            real_a, real_b = data['bar_a'], data['bar_b']
            
            real_a, real_b = real_a.to(device), real_b.to(device)
            opt_G_A2B.zero_grad()
            opt_G_B2A.zero_grad()

            cycle_loss, acc_a, acc_b = model.pretrain(real_a, real_b)
            
            cycle_loss.backward()

            torch.nn.utils.clip_grad.clip_grad_value_(model.G_A2B.parameters(), 100)
            torch.nn.utils.clip_grad.clip_grad_value_(model.G_A2B.parameters(), 100)
            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 100)
            #for p in model.parameters():
            #    p.register_hook(lambda grad: torch.clamp(grad, 0, 100))
            #for p in model.G_A2B.parameters():
            #    p.register_hook(lambda grad: torch.clamp(grad, 0, 100))
            #for p in model.G_B2A.parameters():
            #    p.register_hook(lambda grad: torch.clamp(grad, 0, 100))
            opt_G_A2B.step()
            opt_G_B2A.step()

            total_loss += float(cycle_loss)
            total_acc_a += float(acc_a)
            total_acc_b += float(acc_b)
            num_batch += 1
        print(f"loss:{total_loss/num_batch} acc_a:{total_acc_a/num_batch} acc_b:{total_acc_b/num_batch}")
        if save:
            x = str(b)
            path = "_pretrain_pop_jazz"
            final_path = x + path + ".pth"
            b+=1
            print("saving to " + final_path)
            torch.save(model.state_dict(), final_path)

def train(epochs=20, vocab_size=391, save=True, load=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pop_rock_train_loader, pop_rock_test_loader = get_data()
    model = CycleGAN(vocab_size, vocab_size-1)
    if load:
        model.load_state_dict(torch.load("pretrain_pop_jazz.pth", map_location=torch.device(device)))
    model = model.to(device)
    opt_G_A2B = torch.optim.Adam(model.G_A2B.parameters())
    opt_G_B2A = torch.optim.Adam(model.G_B2A.parameters())
    opt_D_A = torch.optim.Adam(model.D_A.parameters())
    opt_D_B = torch.optim.Adam(model.D_B.parameters())
    b=1
    for epoch in range(epochs):
        model.train()
        print(f"epoch:{epoch}")
        total_loss = 0
        num_batch = 0
        for i, data in enumerate(pop_rock_train_loader):
            real_a, real_b = data['bar_a'], data['bar_b']
            # we may want to feed in as not one_hots and convert to one hots in the model
            real_a, real_b = real_a.to(device), real_b.to(device)
            opt_G_A2B.zero_grad()
            opt_G_B2A.zero_grad()
            opt_D_A.zero_grad()
            opt_D_B.zero_grad()

            cycle_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss = model(real_a, real_b)
            
            g_A2B_loss.backward(retain_graph=True)
            g_B2A_loss.backward(retain_graph=True) #changed to true
            #torch.nn.utils.clip_grad.clip_grad_value_(model.G_A2B.parameters(), 500)
            #torch.nn.utils.clip_grad.clip_grad_value_(model.G_A2B.parameters(), 500)
            #torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 500)
            #for p in model.parameters():
            #    p.register_hook(lambda grad: torch.clamp(grad, 0, 100))
            #for p in model.G_A2B.parameters():
            #    p.register_hook(lambda grad: torch.clamp(grad, 0, 100))
            #for p in model.G_B2A.parameters():
            #    p.register_hook(lambda grad: torch.clamp(grad, 0, 100))

            opt_G_A2B.step()
            opt_G_B2A.step()
            print("84 train")
            with torch.autograd.set_detect_anomaly(True):
                d_A_loss = copy.copy(d_A_loss)
                d_A_loss.backward(retain_graph=True)
                print("86 train")
                d_B_loss = copy.copy(d_B_loss)
                d_B_loss.backward()

            opt_D_A.step()
            opt_D_B.step()
            total_loss += float(g_A2B_loss)
            num_batch += 1
        print(f"loss:{total_loss/num_batch}")
        if save:
            path = str(b) + "train_model" + ".pth"
            torch.save(model.state_dict(), path)
            b+=1

    if save:
        torch.save(model.state_dict(), 'modelPLS.pth')

if __name__=="__main__":
    pretrain()