from models import CycleGAN
import torch
from datasets import get_data

pop_jazz_train_loader, pop_jazz_test_loader = get_data()

def train(epochs=1, vocab_size=391, save=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pop_rock_train_loader, pop_rock_test_loader = get_data()
    model = CycleGAN(vocab_size, vocab_size-1)
    model = model.to(device)
    opt_G_A2B = torch.optim.Adam(model.G_A2B.parameters())
    opt_G_B2A = torch.optim.Adam(model.G_B2A.parameters())
    opt_D_A = torch.optim.Adam(model.D_A.parameters())
    opt_D_B = torch.optim.Adam(model.D_B.parameters())

    for epoch in range(epochs):
        model.train()
        print(f"epoch:{epoch}")
        total_loss = 0
        num_batch = 0
        for i, data in enumerate(pop_rock_train_loader):
            print(f"batch: {i}")
            real_a, real_b = data['bar_a'], data['bar_b']
            # we may want to feed in as not one_hots and convert to one hots in the model
            real_a, real_b = real_a.to(device), real_b.to(device)
            opt_G_A2B.zero_grad()
            opt_G_B2A.zero_grad()
            opt_D_A.zero_grad()
            opt_D_B.zero_grad()

            cycle_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss = model(real_a, real_b)
            
            g_A2B_loss.backward(retain_graph=True)
            g_B2A_loss.backward(retain_graph=True)

            d_A_loss.backward(retain_graph=True)
            d_B_loss.backward()

            opt_G_A2B.step()
            opt_G_B2A.step()
            opt_D_A.step()
            opt_D_B.step()
            total_loss += float(g_A2B_loss)
            num_batch += 1
        print(f"loss:{total_loss/num_batch}")
        if save:
            torch.save(model.state_dict(), 'model.pth')

    if save:
        torch.save(model.state_dict(), 'model.pth')

if __name__=="__main__":
    train()