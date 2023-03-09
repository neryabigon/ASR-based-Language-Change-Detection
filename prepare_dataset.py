import pandas as pd
from datasets import load_dataset, load_metric, Audio, ClassLabel, load_from_disk, Features, Value, concatenate_datasets
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import numpy as np
from IPython.display import display, HTML
import re
import json
from torch.utils.tensorboard import SummaryWriter
import pyarabic.araby as araby
from unidecode import unidecode

'''
This script prepares the dataset for training.
this means that it will take from the dataset 40000 random pairs of sentences with a cosine similarity of 0.56 and up,
 in the first stage.
then it will concatenate the pairs in the following way:
    1. the audio of the first sentence of the pair will be concatenated 
    with the audio of the second sentence of the pair with a pause of 0.5 seconds in between.
    2. the sentence of the first sentence of the pair will be concatenated to the sentence of the 
    second sentence of the pair with a space in between.
'''
PATH_1 = 'arabic'
PATH_2 = 'portuguese'

print("----------------- Checking if cuda is available... -----------------")
print(f'Cuda Available = {torch.cuda.is_available()}\n\n')

# load the dataset
print("Loading pairs...")
df = pd.read_pickle('pickles/cosine_similarity/arabic_portuguese_train.pickle')

# get the sentences with a cosine similarity of 0.56 and up
print("Filtering dataset...")
df = df[df['cos_sim'] >= 0.56]

# get 40000 random pairs
print("Sampling dataset at random...")
df = df.sample(n=10, random_state=1)
# df = df.sample(n=40000, random_state=1)

print("----------------- Loading Datasets... -----------------")
features = Features(
    {
        "client_id": Value("string"),
        "path": Value("string"),
        "audio": Audio(sampling_rate=48_000),
        "sentence": Value("string"),
        "up_votes": Value("int64"),
        "down_votes": Value("int64"),
        "age": Value("string"),
        "gender": Value("string"),
        "accents": Value("string"),
        "locale": Value("string"),
        "segment": Value("string"),
    }
)

# load the dataset
dataset_1 = load_dataset('csv', data_files={'train': 'train.csv', },
                         data_dir='/home/or/Desktop/language-and-speaker-change-detection-based-on-automatic-speech'
                                  '-recognition-methods-/datasets csv/arabic')
dataset_2 = load_dataset('csv', data_files={'train': 'train.csv', },
                         data_dir='/home/or/Desktop/language-and-speaker-change-detection-based-on-automatic-speech'
                                  '-recognition-methods-/datasets csv/portuguese')

# cast the features
dataset_1 = dataset_1.cast(features)
dataset_2 = dataset_2.cast(features)

# remove the columns
dataset_1 = dataset_1.remove_columns(
    ["accents", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
dataset_2 = dataset_2.remove_columns(
    ["accents", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

dataset_1 = dataset_1['train']
dataset_2 = dataset_2['train']
print("\n----------------- Loading Datasets complete. -----------------")

# take only the lines that are in the 40000 random pairs from the first stage
print("Loading chosen pairs from the dataset...")
dataset_1 = dataset_1.filter(lambda example: example['path'] in df[PATH_1].values)
dataset_2 = dataset_2.filter(lambda example: example['path'] in df[PATH_2].values)

# loop through df and print the line that wasn't found in the dataset
for index, row in df.iterrows():
    if row[PATH_1] not in dataset_1['path']:
        print(row[PATH_1])
    if row[PATH_2] not in dataset_2['path']:
        print(row[PATH_2])

# Rename the columns in dataset_2 to avoid duplicates
dataset_2 = dataset_2.rename_column("path", "path_2")
dataset_2 = dataset_2.rename_column("audio", "audio_2")
dataset_2 = dataset_2.rename_column("sentence", "sentence_2")

# Concatenate the datasets
dataset = concatenate_datasets([dataset_1, dataset_2], axis=1)


# merge the two datasets into one dataset according to the way we stated in the beginning of the script: audio +
# pause + audio, sentence + space + sentence

def merge(sample):
    # concatenate the audio
    sample['audio'] = torch.cat((torch.from_numpy(sample['audio']['array']), torch.zeros(24000),
                                 torch.from_numpy(sample['audio_2']['array'])), dim=0)
    # concatenate the sentence
    sample['sentence'] = sample['sentence'] + ' ' + sample['sentence_2']
    return sample


# merge the two datasets
dataset = dataset.map(merge)

# remove the columns that we don't need
dataset = dataset.remove_columns(["path_2", "audio_2", "sentence_2"])

# save the dataset
print("Saving dataset to disk...")
dataset.save_to_disk('pickles/merged_dataset/arabic_portuguese')
print("Done!")
