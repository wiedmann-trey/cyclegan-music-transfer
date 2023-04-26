from models import CycleGAN
import torch

def train(epochs=10, save=True):
    model = CycleGAN()

    opt_G_A2B = torch.optim.Adam(model.G_A2B.parameters())
    opt_G_B2A = torch.optim.Adam(model.G_B2A.parameters())
    opt_D_A = torch.optim.Adam(model.D_A.parameters())
    opt_D_B = torch.optim.Adam(model.D_B.parameters())


    for epoch in range(epochs):
        model.train()
        for batch in batches: # TODO FIGURE OUR DATA LOADING / BATCHING
            opt_G_A2B.zero_grad()
            opt_G_B2A.zero_grad()
            opt_D_A.zero_grad()
            opt_D_B.zero_grad()

            cycle_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss = model(real_a, real_b, real_mixed)

            g_A2B_loss.backward(retain_graph=True)
            g_B2A_loss.backward(retain_graph=True)

            d_A_loss.backward(retain_graph=True)
            d_B_loss.backward(retain_graph=True)

            opt_G_A2B.step()
            opt_G_B2A.step()
            opt_D_A.step()
            opt_D_B.step()
        print(epoch)

    if save:
        torch.save(model.state_dict(), 'model.pth')

if __name__=="main":
    train()