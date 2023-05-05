from models import CycleGAN
import torch
from datasets import get_data
import numpy as np
import muspy
import mido
from torch.nn.utils.rnn import pad_sequence
from mido import MidiFile, MidiTrack
import os

def split_midi(midi_path, time_interval, final_name):
    """given a midi file path and a time interval, divides the midi file into sub-midi-files of that time interval.
    Returns a list of muspy music objects of that time interval"""
    original_mido = MidiFile(midi_path, clip=True) # a lot of the LAKH ones were erroring since some of the velocities were over 127 so im clipping them w mido
    split_midi_files = [] 
    split_muspy_events = []
    split_song_labels = []
    total_seconds = original_mido.length #length of midos are given in seconds 
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
    file_names = []
    for split_midi in split_midi_files:
        split_mus = muspy.from_mido(split_midi)
        new_mus = muspy.Music()
        for i in split_mus.tracks:
            if not i.is_drum:
                new_mus.tracks.append(i)
        low_mus = new_mus.adjust_resolution(84)
        weird_mus = new_mus.adjust_resolution(384)
        low_mus = muspy.to_event_representation(low_mus, use_end_of_sequence_event=False)
        weird_mus = muspy.to_event_representation(weird_mus, use_end_of_sequence_event=False)
        if np.array_equal(weird_mus,low_mus):
            print("yay they're equal!")
        split_mus = muspy.to_event_representation(new_mus, use_end_of_sequence_event=False)
        split_muspy_events.append(split_mus)
        subinterval+=1
    
    return split_muspy_events

def testing_numpy(numpy_paths, file_head):
    i=0
    for numpy_path in numpy_paths: 
        timeshift = np.load(numpy_path)
        music = muspy.from_event_representation(timeshift)
        file_name = f"{file_head}_{i}.mid"
        muspy.outputs.write_midi(file_name, music)
        i+=1

def test_numpy(path, file_head):
        i=0
        x = split_midi(path, 10, 'jeez')
        music = muspy.from_event_representation(x[1], resolution=48)
        file_name = f"{file_head}_{i}.mid"
        muspy.outputs.write_midi(file_name, music)

test_numpy("maestro-v3.0.0/2006/MIDI-Unprocessed_02_R1_2006_01-04_ORIG_MID--AUDIO_02_R1_2006_01_Track01_wav.midi",
           "yamaha_test")

def baseline(midi_path):
    x = MidiFile(midi_path)
    MidiFile.save(x, 'midoSAMPLE.mid')
    x = muspy.from_mido(x, duplicate_note_mode='lifo')
    x = muspy.adjust_resolution(x, 48)
    muspy.outputs.write_midi('BASELINE_MIDO_sample.mid', x)
    print(x.resolution)
    x = muspy.to_event_representation(x, max_time_shift=10, encode_velocity=True, force_velocity_event=False)
    
    x = muspy.from_event_representation(x, max_time_shift=10, resolution=48, duplicate_note_mode='lifo')
    muspy.outputs.write_midi('BASELINE_sample.mid', x)

baseline("ORIGINAL.midi")