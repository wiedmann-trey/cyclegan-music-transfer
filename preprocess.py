import numpy as np
import os
import pretty_midi
import shutil
import muspy 
from mido import MidiFile, MidiTrack, Message
import numpy as np
import torch as torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def create_song_genre_dict(root):
    """
    reads through "labels for songs" folder and maps each song ID ("TRKJDHSJK for instance") to the corresponding genre
    """
    song_genre_dict = {}
    for dirpath, dirs, filenames in os.walk(root):
        for file in filenames:

            absolute_file = os.path.join(root, file)

            with open(absolute_file, "r", encoding="utf-8", errors="ignore") as opening_file: 
                lines = opening_file.read().split()

                for i in lines: 
                    song_genre_dict[i] = file[:-4]
    return song_genre_dict

genre_dictionary = create_song_genre_dict("labels for songs")

def get_genre_path(root):
    path_songs = []
    dict_song_genre = {}
    for dirpath, dirs, filenames in os.walk(root):
        for dir in dirs:
            if len((dir)) >= 5:
                path_songs.append(os.path.join(dirpath, dir))
                try: 
                    song_genre = genre_dictionary[dir]
                except KeyError as e:
                    genre_dictionary[dir] = "unknown"
                song_genre_dict[os.path.join(dirpath, dir)] = dir
                
    
    return path_songs, dict_song_genre

lakh_song_folders, song_genre_dict = get_genre_path("lmd_matched")


def create_genre_number_dict(root):
    """creates a dictionary mapping all the string labels to a number which we can later use for one hot encoding"""
    genre_numerical_labels = {}
    for dirpath, dirs, filenames in os.walk(root):
        n = 0
        for file in filenames:
            genre_numerical_labels[file[:-4]] = n
            n += 1
    genre_numerical_labels["classical"] = len(genre_numerical_labels)

genre_number_dict = create_genre_number_dict("labels for songs")

def augment_pitch(muspy_object):
    if np.random.random() < 0.1:
        muspy_object = muspy_object.transpose(np.random.randint(-2, 2))
    return muspy_object

def split_midi(midi_path, time_interval, header):
    """given a midi file path and a time interval, divides the midi file into sub-midi-files of that time interval.
    Returns a list of muspy music objects of that time interval"""
    original_mido = MidiFile(midi_path, clip=True) # a lot of the LAKH ones were erroring since some of the velocities were over 127 so im clipping them w mido
    split_midi_files = [] 
    split_muspy_events = []
    split_song_labels = []
    split_testing_muspy = []
    total_seconds = original_mido.length #length of midos are given in seconds 
    ticks_per_beat = original_mido.ticks_per_beat 
    total_ticks = 0
    try: 
        just_song_key = song_genre_dict[header]
        song_genre = genre_dictionary[just_song_key]
        song_genre = genre_number_dict[song_genre]
    except Exception as e:
        song_genre = genre_number_dict["classical"]

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
                    #msg.time = msg.time * scaling_factor
                    if hasattr(msg, "instrument_name"):
                        "has instrument name"
                        if msg.instrument_name != "drums" and msg.instrument_name != "drum" and msg.instrument_name != "percussion":
                            split_track.append(msg)
                    else:
                        split_track.append(msg)

            new_split_mido.tracks.append(split_track)
        split_midi_files.append(new_split_mido)
    
    string1 = "/Users/carolinezhang/Downloads/cyclegan-music-transfer/lol test folder/subtle" 
    end = ".mid"
    new_file = "1"
    for split_midi in split_midi_files:
        #finalpath = string1 + new_file + end
        split_mus = muspy.from_mido(split_midi)
        new_mus = muspy.Music()
        for i in split_mus.tracks:
            if not i.is_drum:
                new_mus.tracks.append(i)
        #muspy.outputs.save(finalpath, split_mus)
        #muspy.outputs.write(path=finalpath, music=split_mus)
        #split_mus = augment_pitch(split_mus)
        split_mus = muspy.to_event_representation(new_mus, use_end_of_sequence_event=False)
        split_muspy_events.append(split_mus)
        split_song_labels.append(song_genre)
        new_file += "1"
        
    return split_muspy_events, split_song_labels

def get_event_representations(lakh_paths:list, yamaha_path:str, time_interval:int):
    '''converts midis in lakh dataset to muspy event representation'''
    timeshifts = []
    labels = []
    invalid_data = []
    for lakh_path in lakh_paths:
        for dirpath, dirs, midifiles in os.walk(lakh_path):
            for midifile in midifiles: 
                absolute_song_path = os.path.join(lakh_path, midifile)
                try: 
                    lakh_timeshifts, lakh_labels = split_midi(absolute_song_path, time_interval, lakh_path)
                    timeshifts.extend(lakh_timeshifts)
                    labels.extend(lakh_labels)
                    print("lakh not corrupted!")
                    print(labels)
                except Exception as e:
                    print(repr(absolute_song_path))
                    invalid_data.append(absolute_song_path)
                    continue
    for dirpath, dirs, midifiles in os.walk(yamaha_path):
            for midifile in midifiles: 
                absolute_song_path = os.path.join(yamaha_path, midifile)
                try: 
                    yamaha_timeshifts, yamaha_labels = split_midi(absolute_song_path, time_interval, midifile)
                    timeshifts.extend(yamaha_timeshifts)
                    labels.extend(yamaha_labels)
                    print("yamaha not corrupted!")
                    print(labels)
                except Exception as e:
                    print(repr(absolute_song_path))
                    invalid_data.append(absolute_song_path)
                    continue
    #just cuz I wanted to see what it looked like
    with open("muspy_labels.txt", "w") as output:
        output.write(str(labels))
    return timeshifts, labels
                
total_timeshifts, total_labels = get_event_representations(song_paths, "maestro-v3.0.0", 30)


def get_test_train_samples(all_timeshifts, all_labels, first_class, second_class, num_classes):
    first_genre_ts = []
    second_genre_ts = []
    for i in range(len(all_labels)):
        if all_labels[i] == first_class or all_labels[i] == second_class:
                first_genre_ts.append(all_timeshifts[i])
        if all_labels[i] == second_class:
              second_genre_ts.append(all_timeshifts[i])
    first_genre_ts = np.asarray(first_genre_ts)
    second_genre_ts = np.asarray(second_genre_ts)
    genre_labels = np.array(genre_labels)
    genre_labels = torch.from_numpy(genre_labels)
    genre_labels = torch.nn.functional.one_hot(genre_labels, num_classes)
    return first_genre_ts, second_genre_ts

class TimeShiftDataset(Dataset):
    def __init__(self, time_shifts):
        self.time_shifts = time_shifts

    def __len__(self):
        return len(self.time_shifts)

def get_data():
    song_paths, song_genre_dict = get_genre_path("lmd_matched")
    genre_number_dict = create_genre_number_dict("labels for songs")
    total_timeshifts, total_labels = get_event_representations(song_paths, "maestro-v3.0.0", 30)

    pop_rock_samples, classical_samples = get_test_train_samples(total_timeshifts, total_labels, 2, 13, 15)
    pop_rock_data = TimeShiftDataset(time_shifts=pop_rock_samples)
    classical_data = TimeShiftDataset(time_shifts=classical_samples)

    classical = DataLoader(dataset=classical_data, batch_size=32, shuffle=True)
    pop_rock = DataLoader(dataset=classical_data, batch_size=32, shuffle=True)
    return classical, pop_rock