from models import CycleGAN
import torch

def eval():
    model = CycleGAN()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    model.mode = 'A2B'

    #TODO feed in some data and see what we get