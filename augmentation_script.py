import pickle

import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeMask, PitchShift, BandStopFilter
import numpy as np
from datasets import load_metric, load_from_disk
from audiomentations.augmentations.mp3_compression import Mp3Compression
import soundfile as sf  # save data as wav file
import pydub  # convert wav format to mp3
import os
import glob
from pathlib import Path
import audio2numpy as a2n

'''
the goal is to add to each train set of 10000 samples 2187 samples of augmented data such that the 2187 samples are divided equally 
between the 3 augmentations.
each augmented sample should have only one augmentation applied to it.
please change the code to achieve this goal.
the augmented samples should be saved in the same directory as the original samples, and a new line should be added to the dataframe.
the dataframe should be saved as a pickle file.
'''

# load the dataframe
df = pd.read_pickle('pickles/cosine_similarity/spanish_portuguese_train.pickle')


def append_path(example):
    example['spanish'] = '/home/or/Desktop/spanish/chosen_train/' + example['spanish']
    # example['russian'] = '/home/or/Desktop/russian/train/' + example['russian']
    example['portuguese'] = '/home/or/Desktop/portu_dataset/augmentations/train/' + example['portuguese']
    return example

print('----------- Appending paths... ----------------')
df = df.apply(append_path, axis=1)
print('----------- Done appending paths... ----------------')
# choose the samples with the correct cosine similarity
df_high = df[df['cos_sim'] >= 0.58]

# choose 2187 samples from the dataframe
df = df.sample(n=2187, random_state=1)

# make a list of the mp3 files to augment
# list_to_augment_1 = df['russian'].tolist()
list_to_augment_2 = df['portuguese'].tolist()
list_to_augment_1 = df['spanish'].tolist()

new_path = '/home/or/Desktop/spanish_portuguese_augmentations/high'

augment_gaussian = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=1.0),
])

augment_pitch = Compose([
    PitchShift(min_semitones=-6, max_semitones=8, p=1.0),
])

augment_band = Compose([
    BandStopFilter(min_center_freq=60, max_center_freq=2500, min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.4,
                   p=1.0)
])

print('----------- Augmenting... ----------------')
# since we are augmenting 2187 (so it will divide by 3) samples,
# we need to divide the 2187 samples equally between the 3 augmentations
# the first 729 samples will be augmented with gaussian noise,
# the next 729 samples will be augmented with pitch shift
# and the last 729 samples will be augmented with band stop filter

for i in range(0, 729):
    filename_1 = Path(list_to_augment_1[i]).stem
    filename_2 = Path(list_to_augment_2[i]).stem
    # fullpath_1 = '/home/or/Desktop/russian/train/' + filename_1 + '.mp3'
    fullpath_2 = '/home/or/Desktop/portu_dataset/augmentations/train/' + filename_2 + '.mp3'
    fullpath_1 = '/home/or/Desktop/spanish/chosen_train/' + filename_2 + '.mp3'
    x_1, sr_1 = a2n.audio_from_file(list_to_augment_1[i])
    x_2, sr_2 = a2n.audio_from_file(list_to_augment_2[i])
    augmented_samples_1 = augment_gaussian(samples=x_1, sample_rate=48000)
    augmented_samples_2 = augment_gaussian(samples=x_2, sample_rate=48000)
    ru_new_path = new_path + '/' + filename_1 + '.wav'
    ru_new_path_mp3 = new_path + '/' + filename_1 + '.mp3'
    pt_new_path = new_path + '/' + filename_2 + '.wav'
    pt_new_path_mp3 = new_path + '/' + filename_2 + '.mp3'
    sf.write(ru_new_path, augmented_samples_1, 48000)
    sf.write(pt_new_path, augmented_samples_2, 48000)
    # df_high = df_high.append({'russian': ru_new_path_mp3, 'portuguese': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)
    # df_high = df_high.append({'russian': ru_new_path_mp3, 'spanish': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)
    df_high = df_high.append({'spanish': ru_new_path_mp3, 'portuguese': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)

for i in range(729, 1458):
    filename_1 = Path(list_to_augment_1[i]).stem
    filename_2 = Path(list_to_augment_2[i]).stem
    # fullpath_1 = '/home/or/Desktop/russian/train/' + filename_1 + '.mp3'
    fullpath_2 = '/home/or/Desktop/portu_dataset/augmentations/train/' + filename_2 + '.mp3'
    fullpath_1 = '/home/or/Desktop/spanish/chosen_train/' + filename_2 + '.mp3'
    x_1, sr_1 = a2n.audio_from_file(list_to_augment_1[i])
    x_2, sr_2 = a2n.audio_from_file(list_to_augment_2[i])
    augmented_samples_1 = augment_pitch(samples=x_1, sample_rate=48000)
    augmented_samples_2 = augment_pitch(samples=x_2, sample_rate=48000)
    ru_new_path = new_path + '/' + filename_1 + '.wav'
    ru_new_path_mp3 = new_path + '/' + filename_1 + '.mp3'
    pt_new_path = new_path + '/' + filename_2 + '.wav'
    pt_new_path_mp3 = new_path + '/' + filename_2 + '.mp3'
    sf.write(ru_new_path, augmented_samples_1, 48000)
    sf.write(pt_new_path, augmented_samples_2, 48000)
    # df_high = df_high.append({'russian': ru_new_path_mp3, 'portuguese': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)
    # df_high = df_high.append({'russian': ru_new_path_mp3, 'spanish': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)
    df_high = df_high.append({'spanish': ru_new_path_mp3, 'portuguese': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)

for i in range(1458, 2187):
    filename_1 = Path(list_to_augment_1[i]).stem
    filename_2 = Path(list_to_augment_2[i]).stem
    # fullpath_1 = '/home/or/Desktop/russian/train/' + filename_1 + '.mp3'
    fullpath_2 = '/home/or/Desktop/portu_dataset/augmentations/train/' + filename_2 + '.mp3'
    fullpath_1 = '/home/or/Desktop/spanish/chosen_train/' + filename_2 + '.mp3'
    x_1, sr_1 = a2n.audio_from_file(list_to_augment_1[i])
    x_2, sr_2 = a2n.audio_from_file(list_to_augment_2[i])
    augmented_samples_1 = augment_band(samples=x_1, sample_rate=48000)
    augmented_samples_2 = augment_band(samples=x_2, sample_rate=48000)
    ru_new_path = new_path + '/' + filename_1 + '.wav'
    ru_new_path_mp3 = new_path + '/' + filename_1 + '.mp3'
    pt_new_path = new_path + '/' + filename_2 + '.wav'
    pt_new_path_mp3 = new_path + '/' + filename_2 + '.mp3'
    sf.write(ru_new_path, augmented_samples_1, 48000)
    sf.write(pt_new_path, augmented_samples_2, 48000)
    # df_high = df_high.append({'russian': ru_new_path_mp3, 'portuguese': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)
    # df_high = df_high.append({'russian': ru_new_path_mp3, 'spanish': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)
    df_high = df_high.append({'spanish': ru_new_path_mp3, 'portuguese': pt_new_path_mp3, 'cos_sim': 0.999}, ignore_index=True)

print('----------- Augmenting complete. ----------\n\n')

print('----------- exporting to mp3... ----------------')
wav_files = glob.glob(new_path + '/*wav')
for wav in wav_files:
    # print(wav)
    mp3_file = os.path.splitext(wav)[0] + '.mp3'
    sound_2 = pydub.AudioSegment.from_wav(wav)
    sound_2.export(mp3_file, format="mp3")
    os.remove(wav)
print('----------- exporting to mp3 complete. ----------\n\n')

# save the new dataframe to pickle
print('-------------- writing to pickle... ------------------')
with open('pickles/cosine_similarity/spanish_portuguese_train_augmented.pickle', 'wb') as f:
    pickle.dump(df_high, f)
print('-------------- writing to pickle complete. ------------------\n')
