import numpy as np
import os
import pretty_midi
import shutil
import muspy 
from mido import MidiFile
import numpy as np
import torch as torch
from genre_labels import genre_dictionary, lakh_song_folders, song_genre_dict, genre_number_dict

def add_augmentation(muspy_object):
    # tbh I'm not super sure what adjust resolution this is kinda a filler add augmentation function, but we might
    # have to augment midi instead of muspy object
    if np.random.random() < 0.1:
        return muspy_object.adjust_resolution(target=1, factor=0.1)
    return muspy_object

def create_muspy(lakh_song_folders, maestro_path):
    '''goes through all of the lakh song folder contents and creates muspy objects.
    also goes through the main maestro folder and creates muspy objects
    returns all of the muspy objects from both lakh and yamaha, and moves corrupt midi files to 
    a separate folder'''
    muspy_objects = []
    invalid_mids = []
    corresponding_labels = []
    os.mkdir("invalid_midis")
    for song_path in lakh_song_folders:
        for dirpath, dirs, midifiles in os.walk(song_path):
            for midifile in midifiles: 
                absolute_song_path = os.path.join(song_path, midifile)
                try: 
                    just_song_key = song_genre_dict[song_path]
                    song_genre = genre_dictionary[just_song_key]
                    if song_genre != "unknown":
                        muspy_obj = muspy.read(absolute_song_path)
                        muspy_obj = add_augmentation(muspy_obj)
                        muspy_obj.annotations = song_genre
                        muspy_obj = muspy.to_event_representation(music=muspy_obj, encode_velocity=True, max_time_shift=100, velocity_bins=32)
                        muspy_objects.append(muspy_obj)
                        numerical_label = genre_number_dict[song_genre]
                        corresponding_labels.append(numerical_label)
                    else:
                        continue
                except Exception as e:
                    print(repr(absolute_song_path))
                    invalid_mids.append(absolute_song_path)
                    shutil.move(absolute_song_path, "invalid_midis")
                    continue
    #maybe theres an easier way to do this but like we have to use folder names to ID midi files in LAKH whereas in
    #yamaha everything in classical so I just used os.walk on two paths instead of putting them all in one big folder. 
    for dirpath, dirs, filenames in os.walk(maestro_path):
        for filename in filenames: 
            if filename.endswith(".midi") or filename.endswith(".mid"):
                rel_song_path = os.path.join(dirpath, filename)
                try: 
                    new_yamaha = muspy.read(rel_song_path)
                    new_yamaha = add_augmentation(new_yamaha)
                    song_genre = "classical"
                    new_yamaha.annotations = song_genre
                    new_yamaha = muspy.to_event_representation(music=new_yamaha, encode_velocity=True, max_time_shift=100, velocity_bins=32)
                    muspy_objects.append(new_yamaha) 
                    numerical_label = genre_number_dict[song_genre]
                    corresponding_labels.append(numerical_label)
                except Exception as e:
                    print(repr(filename))
                    shutil.move(rel_song_path, "invalid_midis")
                    continue
    return muspy_objects, corresponding_labels


#create_muspy("lmd_matched", "maestro-v3.0.0")
            


