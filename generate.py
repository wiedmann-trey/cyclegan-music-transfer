from models import CycleGAN
import torch
from datasets import get_data
import numpy as np
import muspy
import mido
from torch.nn.utils.rnn import pad_sequence
from mido import MidiFile, MidiTrack

def split_midi(midi_path, time_interval, final_name):
    """given a midi file path and a time interval, divides the midi file into sub-midi-files of that time interval.
    Returns a list of muspy music objects of that time interval"""
    original_mido = MidiFile(midi_path, clip=True) # clip velocities to prevent a lot of the LAKH errors
    split_midi_files = [] 
    split_muspy_events = []
    split_song_labels = []
    total_seconds = original_mido.length #length of midos are given in seconds 
    if time_interval > total_seconds:
        time_interval = total_seconds
    ticks_per_beat = original_mido.ticks_per_beat 
    total_ticks = 0

    #calculate total number of ticks
    for msg in original_mido.tracks[1]: #have to start at index one since the first track is a header so it doesn't have any time ticks (apparently)
        if msg.time:
            total_ticks += msg.time

    total_beats = total_ticks / ticks_per_beat
    ticks_per_second = (total_beats / total_seconds) * ticks_per_beat

    scale = np.random.random()
    scaling_factor = 1

    if scale < 0.1:
        scaling_factor = 1.2
    elif scale > 0.9:
        scaling_factor = 0.8

    for i in range(int((original_mido.length * scaling_factor) // time_interval)):
        new_split_mido = MidiFile()
        lower_range = i * time_interval
        upper_range = (i + 1) * time_interval 

        for track in original_mido.tracks:
            split_track = MidiTrack()
            current_msg_time=0

            for msg in track:
                current_msg_time += (int(np.round(msg.time * scaling_factor))/ ticks_per_second)

                if lower_range <= current_msg_time < upper_range: 
                    msg.time = int(round(msg.time * scaling_factor))
                    if hasattr(msg, "instrument_name"):
                        if msg.instrument_name != "drums" and msg.instrument_name != "drum" and msg.instrument_name != "percussion":
                            split_track.append(msg)
                    else:
                        split_track.append(msg)

            new_split_mido.tracks.append(split_track)
        split_midi_files.append(new_split_mido)
    subinterval = 0
    
    for split_midi in split_midi_files:
        split_mus = muspy.from_mido(split_midi)
        new_mus = muspy.Music()
        for i in split_mus.tracks:
            if not i.is_drum:
                new_mus.tracks.append(i)
        split_mus = muspy.to_event_representation(new_mus, use_end_of_sequence_event=False, use_single_note_off_event=False, encode_velocity=True) # do we want encode velocity true?
        file_name = f"{final_name}_{subinterval}.npy" # create a unique filename based on header and subinterval
        array_mus = split_mus
        split_muspy_events.append(split_mus)
        subinterval+=1
        
    return split_muspy_events

def numpy_to_torch(input_song, complete_song=False, incomplete=True):
    """
    Takes in a numpy array and converts it to a tensor with a start and end token.

    Args:
    - input_song: numpy array, the array containing the event representation of the song

    Returns:
    - timeshift: pytorch tensor representation of the array, with start and end tokens
    """
    if complete_song:
        timeshifts = []
        for i in input_song:
            timeshift = np.ndarray.flatten(i)
            start_token = np.array([388])
            end_token = np.array([389])
            if incomplete: 
                if len(timeshift) > 400:   
                    timeshift = timeshift[:400]
            timeshift = np.concatenate([start_token, timeshift, end_token])
            timeshifts.append(timeshift)
        
        timeshifts = [torch.tensor(seq, requires_grad=False).long() for seq in timeshifts]
        timeshifts = [torch.nn.functional.one_hot(seq, num_classes=(391)).float() for seq in timeshifts]
        input_song = pad_sequence(timeshifts, padding_value=390, batch_first=True)
    else:
        timeshift = np.ndarray.flatten(input_song)
        start_token = np.array([388])
        end_token = np.array([389])

        if incomplete:
            if len(timeshift) > 400:   
                timeshift = timeshift[:400]

        timeshift = np.concatenate([start_token, timeshift, end_token])
        timeshift = torch.tensor(timeshift, requires_grad=False)
        input_song = torch.nn.functional.one_hot(timeshift.long(), num_classes=(391)).float()

        input_song = torch.reshape(input_song, (1, -1, 391))

    return input_song

def generate_song(model_path, input_song_path, output_song_path, genre='jazz', vocab_size=391):

    model = CycleGAN(vocab_size=vocab_size, padding_idx=vocab_size-1, mode='A2B')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    input_song = split_midi(input_song_path, 20, 'testing')[0]

    input_song = numpy_to_torch(input_song, complete_song=False, incomplete=False)

    getting_resolution = MidiFile(input_song_path)
    getting_resolution = muspy.from_mido(getting_resolution, duplicate_note_mode='fifo')
    resolution = getting_resolution.resolution

    
    # Generate a time-shift representation of the output song
    if genre == 'jazz':
        model.G_A2B.pretrain = True
        softmax_output, output_song = model.G_A2B(input_song, temp=0.7)
    
    elif genre == 'pop':
        model.G_B2A.pretrain = True
        softmax_output, output_song = model.G_B2A(input_song, temp=0.7)
    else:
        raise ValueError("Invalid genre specified")

    softmax_output = softmax_output.detach().cpu().numpy()

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

    output_song = np.array(filtered_output)
    output_song = output_song.astype(int)
    output_song = muspy.from_event_representation(output_song, resolution=resolution, use_single_note_off_event=False)
    print("done")
    with open(output_song_path, 'wb') as file:
        muspy.outputs.write_midi(output_song_path, output_song)

if __name__=="__main__":
    generate_song('pretrain not ignoring padding/35_pretrain_pop_jazz.pth',
                  'ORIGINAL.midi', 
                  '35_pretrain.midi', 
                  genre='jazz', 
                  vocab_size=391)
