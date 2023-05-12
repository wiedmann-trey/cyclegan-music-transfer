from LSTM_classifier import LSTMClassifier
import torch 
# from full_generate import get_events
from mido import MidiFile, MidiTrack
import muspy
import numpy as np

def get_events_for_classifier(midi_path):
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
        timeshift = torch.nn.functional.one_hot(timeshift,num_classes=(391)).long() # changed float -> long
        # timeshift = torch.reshape(timeshift, (1, -1, 391)) # should be 2d for lstm
        timeshifts.append(timeshift)
    timeshifts = torch.cat(timeshifts)

    return timeshifts

def classify_song(model_path, input_song_path, requested_genre):

    song = input_song_path.split('/')[2].split('_')[0]
    print(f'classifying output of {song}, requested genre: {requested_genre}')

    # in get classifier data, this is how things were one-hot encoded, so i think this i the right mapping??
    # pop_labels = [torch.nn.functional.one_hot(torch.tensor(1), num_classes=(3)).float() for i in pop_samples]
    # jazz_labels = [torch.nn.functional.one_hot(torch.tensor(0), num_classes=(3)).float() for i in jazz_samples]
    # classical_labels = [torch.nn.functional.one_hot(torch.tensor(2), num_classes=(3)).float() for i in classical_samples]

    genre_to_index = {'jazz': 0, 'pop': 1, 'classical': 2}
    genre_index = genre_to_index[requested_genre]

    # setting up the model
    classifier = LSTMClassifier(embedding_dim=256,hidden_dim=256,
                           vocab_size=391,label_size=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    
    # loading the trained model
    classifier.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    # pasing in input
    input_songs = get_events_for_classifier(input_song_path) 
    #print(len(input_songs))
    input_songs = input_songs[:32]
    num_correct = 0
    total = 0
    counts = {0:0, 1:0, 2:0}
    song = input_songs
    song = torch.tensor(song, dtype=torch.long)
    #for song in input_songs:
    #print('song', song)
    #print(song.shape)
    classifier_output = classifier(song)
    print(classifier_output)
    max_i = torch.argmax(classifier_output)
    if max_i == genre_index:
        num_correct += 1
    total += 1
    counts[max_i.item()] += 1

    # return the number of timeshifts that were classified correctly
    # and then most common genre predicted by the classifier

    #percent_correct = num_correct / total

    #print(counts)
    most_common_i = max(counts, key=counts.get) 
    index_to_genre = {0: 'jazz', 1: 'pop', 2: 'classical'} 
    most_common_genre = index_to_genre[most_common_i]
    print(most_common_genre)
    return most_common_genre
    

if __name__=="__main__":
    classify_song(model_path='Classifiers/PJC_classifier_14.pth',
                  input_song_path='generating_songs/base_songs/IfIAintGotYou.mid',
                  requested_genre='jazz')
