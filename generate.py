from models import CycleGAN
import torch
from datasets import get_data
import numpy as np
import muspy
import mido
from torch.nn.utils.rnn import pad_sequence

def generate_song(model_path, input_song_path, output_song_path, genre='jazz', vocab_size=391):

    model = CycleGAN(vocab_size=vocab_size, padding_idx=vocab_size-1, mode='A2B')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))#, map_location=torch.device('cpu')))
    
    input_song = mido.MidiFile(input_song_path, clip=True)
    input_song = muspy.from_mido(input_song)
    input_song = muspy.to_event_representation(input_song)
    input_song = np.ndarray.flatten(input_song)
    testing_song = input_song[:550]
    testing_song = torch.tensor(testing_song, requires_grad=False)
    input_song = input_song[:500]
    start_token = np.array([388])
    end_token = np.array([389])
    input_song = np.concatenate([start_token, input_song, end_token])
    
    #this was to see if padding would make any difference
    input_song = torch.tensor(input_song, requires_grad=False)
    tensor_list = [testing_song, input_song]
    tensor_list = pad_sequence(tensor_list, batch_first=True, padding_value=390)
    input_song = tensor_list[1]
    input_song = torch.nn.functional.one_hot(input_song, num_classes=(vocab_size)).float()
    
    #input_song = input_song[None, :]
    input_song = torch.reshape(input_song, (1, 550, 391))
    # Generate a time-shift representation of the output song
    if genre == 'jazz':
        softmax_output, output_song = model.G_A2B(input_song)
    
    elif genre == 'classical':
        softmax_output, output_song = model.G_B2A(input_song)
    else:
        raise ValueError("Invalid genre specified")
    print(softmax_output)
    softmax_output = torch.squeeze(softmax_output)
    softmax_output = softmax_output.detach().cpu().numpy()
    np.savetxt('SADsoftmax.txt', softmax_output)
    output_song = output_song.detach().cpu().numpy()
    mask = np.logical_and(output_song != 388, output_song != 389, output_song != 390)
    output_song = output_song[mask]
    output_song = (np.round(output_song)).astype(int)
    output_song = np.ndarray.flatten(output_song)
    print(output_song)
    print(len(output_song))

    output_song = output_song.reshape(-1, 1)
    print(output_song)
    np.savetxt('SAD.txt', output_song)
    output_song = muspy.from_event_representation(output_song)
    
    with open(output_song_path, 'wb') as file:
        muspy.outputs.write_midi(output_song_path, output_song)

if __name__=="__main__":
    generate_song('model_5eps.pth', 
                  'ORIGINAL.midi', 
                  'TryingAgain.mid', 
                  genre='jazz', 
                  vocab_size=391)
    
#if __name__=="__main__":
#    jfc = muspy.Music('YAYYYYY.mid')
#    jfc = muspy.to_default_event_representation(jfc)
#    print(jfc)
    
