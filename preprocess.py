import numpy as np
import os
import shutil
import muspy 
from mido import MidiFile, MidiTrack, Message
import numpy as np
import torch as torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

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
                    song_genre_dict[i] = file.split("_")[0]
    return song_genre_dict

def get_genre_path(root, genre_dict):
    '''creates a list of all the song filepaths in LMD_matched and also maps the song label to the genre'''
    path_songs = []
    dict_song_genre = {}
    faulty_file = 1
    for dirpath, dirs, filenames in os.walk(root):
        for dir in dirs:
            if len((dir)) >= 5:
                try: 
                    song_genre = genre_dict[dir]
                    dict_song_genre[os.path.join(dirpath, dir)] = dir
                    path_songs.append(os.path.join(dirpath, dir))
                except KeyError as e:
                    #error_dir = "/Users/carolinezhang/Downloads/cyclegan-music-transfer/incorrect_genre_midis/"
                    #error_string = str(faulty_file)
                    #error_path = error_dir + error_string
                    #shutil.move(os.path.join(dirpath, dir), error_path)
                    #faulty_file+=1
                    continue             
    return path_songs, dict_song_genre

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
    total_seconds = original_mido.length #length of midos are given in seconds 
    ticks_per_beat = original_mido.ticks_per_beat 
    total_ticks = 0
    genre_path = "/Users/carolinezhang/Downloads/cyclegan-music-transfer/classical_events/"
    try: 
        just_song_key = song_genre_dict[header]
        song_genre = genre_dictionary[just_song_key]
        song_genre = genre_number_dict[song_genre]
        if song_genre == 0: 
            genre_path = "/Users/carolinezhang/Downloads/cyclegan-music-transfer/jazz_events/"
        if song_genre == 1: 
            genre_path = "/Users/carolinezhang/Downloads/cyclegan-music-transfer/pop_events/"
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
                    msg.time = int(round(msg.time * scaling_factor))
                    if hasattr(msg, "instrument_name"):
                        if msg.instrument_name != "drums" and msg.instrument_name != "drum" and msg.instrument_name != "percussion":
                            split_track.append(msg)
                    else:
                        split_track.append(msg)

            new_split_mido.tracks.append(split_track)
        split_midi_files.append(new_split_mido)
    subinterval = 0
    new_path = midi_path.replace('/', '').replace('.', '')
    for split_midi in split_midi_files:
        split_mus = muspy.from_mido(split_midi)
        new_mus = muspy.Music()
        for i in split_mus.tracks:
            if not i.is_drum:
                new_mus.tracks.append(i)
        split_mus = muspy.to_event_representation(new_mus, use_end_of_sequence_event=False)
        file_name = f"{new_path}_{subinterval}.npy" # create a unique filename based on header and subinterval
        final_path = os.path.join(genre_path, file_name)
        array_mus = split_mus
        np.save(final_path, array_mus, allow_pickle=True)
        split_muspy_events.append(split_mus)
        split_song_labels.append(song_genre)
        subinterval+=1
        
    return split_muspy_events, split_song_labels

#like the main thing is that it would be better if the lakh_paths that we're inputting only correspond to the training genre(s) so that the preprocessing
#takes a lot longer
def get_event_representations(lakh_paths:list, yamaha_path:str, time_interval:int, lakh_first=False):
    '''converts midis in lakh dataset to muspy event representation'''
    timeshifts = []
    labels = []
    
    num_left = 14704
    faulty_file = 0
    if lakh_first: 
        for lakh_path in lakh_paths:
                for dirpath, dirs, midifiles in os.walk(lakh_path):
                    for midifile in midifiles: 
                        if midifile.endswith(".midi") or midifile.endswith(".mid"):
                            absolute_song_path = os.path.join(lakh_path, midifile)
                            try: 
                                lakh_timeshifts, lakh_labels = split_midi(absolute_song_path, time_interval, lakh_path)
                                timeshifts.extend(lakh_timeshifts)
                                labels.extend(lakh_labels)
                                print("lakh not corrupted!")
                                
                            except Exception as e:
                                print(repr(absolute_song_path))
                                print(e)
                                
                                continue
                            num_left-=1
                            print(num_left)
    #yam_songs = 0
    for dirpath, dirs, midifiles in os.walk(yamaha_path):
            #for dir in dirs:
                #if len(dir)==4:
                    for midifile in midifiles:
                        if midifile.endswith(".midi") or midifile.endswith(".mid"):
                        #print(midifile)
                        #if midifile[0]!=".":
                            absolute_song_path = os.path.join(dirpath, midifile)
                            #absolute_song_path = os.path.join(yamaha_path, rel_path)
                            print(midifile)
                            try: 
                                yamaha_timeshifts, yamaha_labels = split_midi(absolute_song_path, time_interval, midifile[:-5])
                                timeshifts.extend(yamaha_timeshifts)
                                labels.extend(yamaha_labels)
                                print("yamaha not corrupted!")
                                print(len(labels))
                                print(len(timeshifts))
                                #yam_songs += 1
                            except Exception as e:
                                print(repr(absolute_song_path))  
                                print(e)  
                                continue          
                            
    #timeshifts = [torch.tensor(seq) for seq in timeshifts]
    #timeshifts = pad_sequence(timeshifts, padding_value=0)
    #print(timeshifts.shape)

    return timeshifts, labels
               
def get_test_train_samples(all_timeshifts, all_labels, first_class, second_class, num_classes):
    """given lists of all the muspy event representations and the corresponding genre labels, extracts only the ones corresponding to the 
    two training genres"""
    first_genre_ts = []
    second_genre_ts = []
    for i in range(len(all_labels)): #like ideally we wouldn't have to do this 
        if all_labels[i] == first_class:
            first_genre_ts.append(all_timeshifts[i])
        if all_labels[i] == second_class:
            second_genre_ts.append(all_timeshifts[i])
    #first_genre_ts = np.ndarray(first_genre_ts)
    #second_genre_ts = np.ndarray(second_genre_ts)
    #genre_labels = np.array(genre_labels)
    #genre_labels = torch.from_numpy(genre_labels)
    #genre_labels = torch.nn.functional.one_hot(genre_labels, num_classes)
    #first_genre_ts = np.random.shuffle(first_genre_ts)
    #second_genre_ts = np.random.shuffle(second_genre_ts)
    return first_genre_ts, second_genre_ts


genre_dictionary = create_song_genre_dict("labels for songs") #maps song IDs to corresponding genre
song_paths, song_genre_dict = get_genre_path("lmd_matched", genre_dictionary)
    #song_paths = paths to LAKH songs corresponding to our genres, 
    #song_genre_dict = dict mapping LAKH folder paths to the corresponding song IDs
genre_number_dict = {"jazz": 0, "pop": 1, "classical": 2} #dict mapping song labes to numbers  #changes to two zeros just for small samples
print(len(song_paths))

def get_data():
    total_timeshifts, total_labels = get_event_representations(song_paths, "maestro-v3.0.0", 5, lakh_first=False)
    #pop_samples, classical_samples = get_test_train_samples(total_timeshifts, total_labels, 1, 2, 3)

get_data()

   
