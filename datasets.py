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
from process_numpy import numpy_to_torch


class TimeShiftDataset(Dataset):
    def __init__(self, A_time_shifts, B_time_shifts, genre_a, genre_b):
        self.A_time_shifts = A_time_shifts
        self.B_time_shifts = B_time_shifts
        self.genre_a = genre_a 
        self.genre_b = genre_b

    def __len__(self):
        return len(self.A_time_shifts)

    def __getitem__(self, index):
        bar_a = self.A_time_shifts[index]
        bar_b = self.B_time_shifts[index]
        bar_a = torch.squeeze(bar_a)
        bar_b = torch.squeeze(bar_b)
        baridx = np.array([index])
        sample = {'baridx': baridx, 'bar_a': bar_a, 'bar_a_label': self.genre_a,
                  'bar_b': bar_b, 'bar_b_label': self.genre_b}
        return sample
    
class ClassifierDataset(Dataset):
    def __init__(self, all_genres, all_labels):
        self.all_genres = all_genres
        self.all_labels = all_labels

    def __len__(self):
        return len(self.all_genres)

    def __getitem__(self, index):
        timeshift = self.all_genres[index]
        timeshift = torch.squeeze(timeshift)
        timeshift_label = self.all_labels[index]
        sample = {'timeshift': timeshift, 'timeshift_label': timeshift_label}
        return sample

def get_data():
    pop_samples = numpy_to_torch("UPDATED_POP")
    jazz_samples = numpy_to_torch("UPDATED_JAZZ")

    num_samples = min(len(pop_samples), len(jazz_samples))
    num_samples_round = num_samples // 32
    pop_samples = pop_samples[:num_samples_round*32]
    jazz_samples = jazz_samples[:num_samples_round*32]
    pop_jazz_set = TimeShiftDataset(A_time_shifts=pop_samples, B_time_shifts=jazz_samples, genre_a=1, genre_b=2)

    pop_jazz_train, pop_jazz_test = data.random_split(pop_jazz_set, [int(round(len(pop_samples)*0.8)), int(round(len(pop_samples)*0.2))])
    
    pop_jazz_train_loader = DataLoader(dataset=pop_jazz_train, batch_size=32, shuffle=True)
    pop_jazz_test_loader = DataLoader(dataset=pop_jazz_test, batch_size=32, shuffle=False)

    return pop_jazz_train_loader, pop_jazz_test_loader

def get_classifier_data(batch_size):
    pop_samples = numpy_to_torch("UPDATED_POP")
    jazz_samples = numpy_to_torch("UPDATED_JAZZ")
    num_samples = min(len(pop_samples), len(jazz_samples))
    mod = num_samples // batch_size
    
    pop_samples = pop_samples[:mod*batch_size]
    jazz_samples = jazz_samples[:mod*batch_size]
    
    pop_labels = [torch.nn.functional.one_hot(torch.tensor(1), num_classes=(2)).float() for i in pop_samples]
    jazz_labels = [torch.nn.functional.one_hot(torch.tensor(0), num_classes=(2)).float() for i in jazz_samples]

    all_samples = torch.cat((pop_samples, jazz_samples))
    
    pop_labels.extend(jazz_labels)
    all_labels=pop_labels
    print(len(all_samples))
    
    pop_jazz_set = ClassifierDataset(all_genres = all_samples, all_labels = all_labels)
    pop_jazz_train, pop_jazz_test = data.random_split(pop_jazz_set, [int(round(len(all_labels)*0.5)), int(round(len(all_labels)*0.5))])
    
    pop_jazz_train_loader = DataLoader(dataset=pop_jazz_train, batch_size=32, shuffle=True)
    pop_jazz_test_loader = DataLoader(dataset=pop_jazz_test, batch_size=32, shuffle=False)

    return pop_jazz_train_loader, pop_jazz_test_loader
