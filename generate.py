from models import CycleGAN
import torch
from datasets import get_data
import numpy as np
import muspy
import mido

def generate_song(input_song_path, output_song_path, genre='jazz', vocab_size=391):

    model = CycleGAN(vocab_size, vocab_size-1, mode='A2B')

    model.load_state_dict(torch.load('model.pth'))
    input_song = mido.MidiFile(input_song_path, clip=True)
    input_song = muspy.from_mido(input_song)
    input_song = muspy.to_event_representation(input_song)
    input_song = np.ndarray.flatten(input_song)
    
    # Generate a time-shift representation of the output song
    if genre == 'jazz':
        output_song = model.G_A2B(input_song)[1]
    elif genre == 'classical':
        output_song = model.G_B2A(input_song)[1]
    else:
        raise ValueError("Invalid genre specified")    
    mask = np.logical_and(output_song != 388, output_song != 389, output_song != 390)
    output_song = output_song = output_song[mask]
    output_song = muspy.from_event_representation(output_song)

    with open(output_song_path, 'wb') as file:
        muspy.outputs.write_midi(output_song_path, output_song)

if __name__=="__main__":
    generate_song('input_song.mid', 'output_song.mid', genre='jazz')