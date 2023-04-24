import os

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

genre_dictionary = create_song_genre_dict("preprocessing/labels_for_songs_(our_genres)")


# didn't change these two below, just moved into a separate file

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

genre_number_dict = create_genre_number_dict("preprocessing/labels_for_songs_(our_genres)")
