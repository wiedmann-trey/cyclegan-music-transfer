from models import CycleGAN
import torch
from datasets import get_data
import numpy as np
import muspy
import mido
from torch.nn.utils.rnn import pad_sequence
from mido import MidiFile, MidiTrack

def get_events(midi_path):
    """given a midi file path and a time interval, divides the midi file into sub-midi-files of that time interval.
    Returns a list of muspy music objects of that time interval"""
    original_mido = MidiFile(midi_path, clip=True) # clip velocities to prevent a lot of the LAKH errors
    mus_object = muspy.from_mido(original_mido)
    mus_object = muspy.to_event_representation(mus_object, encode_velocity=True)
    mus_object = np.ndarray.flatten(mus_object)
    timeshifts = []
    mus_length = len(mus_object)
    for i in range(mus_length // 400):
        if 400*(i+1) <= mus_length:
            ending_interval = 400*(i+1)
        else:
            ending_interval = mus_length
        split_mus = mus_object[i*400 : ending_interval]
        if len(split_mus) < 400:
            split_mus = np.pad(split_mus, (0, 400-len(split_mus)), 'constant', constant_values=(390, 390))
        start_token = np.array([388])
        end_token = np.array([389])
        timeshift = np.concatenate([start_token, split_mus, end_token])
        timeshift = torch.tensor(timeshift, requires_grad=False).long()
        timeshift = torch.nn.functional.one_hot(timeshift,num_classes=(391)).float()
        timeshift = torch.reshape(timeshift, (1, -1, 391))
        timeshifts.append(timeshift)

    return timeshifts



def generate_song(model_path, input_song_path, output_song_path, temp, genre='jazz', vocab_size=391, pretrain=False):

    song = input_song_path.split('/')[2].split('_')[0]
    print(f'generating a {genre} song with temperature {temp} from {song}')
    model = CycleGAN(vocab_size=vocab_size, padding_idx=vocab_size-1, mode='A2B')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    input_songs = get_events(input_song_path)

    getting_resolution = MidiFile(input_song_path)
    getting_resolution = muspy.from_mido(getting_resolution, duplicate_note_mode='fifo')
    resolution = getting_resolution.resolution

    final_song = []
    final_song = np.array(final_song)
    i=0
    for song in input_songs:
        i+=1
        # Generate a time-shift representation of the output song
        if genre == 'jazz':
            if pretrain:
                model.G_A2B.pretrain = True
            softmax_output, output_song = model.G_A2B(song, temp=temp)
            output_song = output_song.detach().cpu().numpy()
            output_song = np.array(output_song)
            output_song = np.ndarray.flatten(output_song)
            filtered_output = []
            for tok in output_song:
                if tok == 388 or tok == 390:
                    continue
                if tok == 389:
                    break
                filtered_output.append(tok)
            final_song = np.append(final_song, np.array(filtered_output))

        elif genre == 'pop':
            if pretrain:
                model.G_B2A.pretrain = True
            softmax_output, output_song = model.G_B2A(song, temp=temp)
            output_song = output_song.detach().cpu().numpy()
            output_song = np.array(output_song)
            output_song = np.ndarray.flatten(output_song)
            filtered_output = []
            for tok in output_song:
                if tok == 388 or tok == 390:
                    continue
                if tok == 389:
                    break
                filtered_output.append(tok)
            final_song = np.append(final_song, np.array(filtered_output))

        else:
            raise ValueError("Invalid genre specified")

    output_song = np.array(final_song)
    output_song = output_song.astype(int)
    print(output_song)
    output_song = muspy.from_event_representation(output_song, resolution=resolution)
    print("done")
    with open(output_song_path, 'wb') as file:
        muspy.outputs.write_midi(output_song_path, output_song)

if __name__=="__main__":
    generate_song('trained_classical_jazz/88CJ_train_model.pth',
                  'generating_songs/base_songs/ORIGINAL.midi', 
                  'generating_songs/outputted_songs/ORIGINAL_jazz_55.mid', 
                  temp=0.55,
                  genre='jazz', 
                  vocab_size=391, 
                  pretrain=False)
    