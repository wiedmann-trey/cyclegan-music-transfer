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
import pickle

def numpy_to_torch(folder_path):
    """
    Loads numpy arrays in a folder and returns them as a list of numpy arrays.

    Args:
    - folder_path: string, the path to the folder containing the numpy arrays to be unpickled.

    Returns:
    - data: list of numpy arrays, contains all unpickled numpy arrays from the folder.
    """
    timeshifts = []
    for filename in os.listdir(folder_path):
    # check if the file is a .npy file
        if filename.endswith(".npy"):
            # load the array from the file
            timeshift = np.load(os.path.join(folder_path, filename))
            start_token = np.array([388])
            end_token = np.array([389])
            timeshift = np.ndarray.flatten(timeshift)
            if len(timeshift) > 400:
                timeshift = timeshift[:400]
            timeshift = np.concatenate([start_token, timeshift, end_token])
            timeshifts.append(timeshift)
    timeshifts = [torch.tensor(seq, requires_grad=False) for seq in timeshifts]
    timeshifts = pad_sequence(timeshifts, padding_value=390, batch_first=True)
    print("loaded!") 
    return timeshifts
