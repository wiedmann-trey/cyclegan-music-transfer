from LSTM_classifier import LSTMClassifier
import torch 
from full_generate import get_events

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
    input_songs = get_events(input_song_path) 
    num_correct = 0
    total = 0
    counts = {0:0, 1:0, 2:0}

    for song in input_songs:
        print('song', song)
        classifier_output = classifier(song)
        print(classifier_output)
        max_i = torch.argmax(classifier_output)
        if max_i == genre_index:
            num_correct += 1
        total += 1
        counts[max_i] += 1

    # return the number of timeshifts that were classified correctly
    # and then most common genre predicted by the classifier

    percent_correct = num_correct / total

    print(counts)
    most_common_i = max(counts, key=counts.get) 
    index_to_genre = {0: 'jazz', 1: 'pop', 2: 'classical'} 
    most_common_genre = index_to_genre[most_common_i]

    return percent_correct, most_common_genre
    

if __name__=="__main__":
    classify_song(model_path='Classifiers/PJC_classifier_14.pth',
                  input_song_path='generating_songs/outputted_songs/PianoMan_jazz_55.mid',
                  requested_genre='jazz')
