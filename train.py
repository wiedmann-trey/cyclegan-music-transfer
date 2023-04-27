from models import CycleGAN
import torch
from datasets import get_data

def train(epochs=10, save=True):
    pop_rock_train_loader, pop_rock_test_loader = get_data()
    print("get data no issue")
    model = CycleGAN(389)
    print("this part the issue")
    opt_G_A2B = torch.optim.Adam(model.G_A2B.parameters())
    opt_G_B2A = torch.optim.Adam(model.G_B2A.parameters())
    opt_D_A = torch.optim.Adam(model.D_A.parameters())
    opt_D_B = torch.optim.Adam(model.D_B.parameters())
    ite=0
    for epoch in range(epochs):
        model.train()
        print("its model.train")
        #for i, batch in batches: # TODO FIGURE OUR DATA LOADING / BATCHING
        for i, data in enumerate(pop_rock_train_loader):
            print("starting batch!")
            ite = ite + 1
            real_a, real_b = data['bar_a'], data['bar_b']
            print(real_a)
            
            real_a = torch.nn.functional.one_hot(torch.tensor(real_a, dtype=torch.int64), num_classes=(387))#.float()
            real_b = torch.nn.functional.one_hot(torch.tensor(real_b, dtype=torch.int64), num_classes=(387))#.float()
            
            opt_G_A2B.zero_grad()
            opt_G_B2A.zero_grad()
            opt_D_A.zero_grad()
            opt_D_B.zero_grad()

            cycle_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss = model(real_a, real_b)
            
            g_A2B_loss.backward(retain_graph=True)
            g_B2A_loss.backward(retain_graph=True)

            d_A_loss.backward(retain_graph=True)
            d_B_loss.backward(retain_graph=True)

            opt_G_A2B.step()
            opt_G_B2A.step()
            opt_D_A.step()
            opt_D_B.step()
            print(cycle_loss)
        print(epoch)

    if save:
        torch.save(model.state_dict(), 'model.pth')

if __name__=="__main__":
    train()
