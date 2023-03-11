import pandas as pd
from datasets import Audio, Features, Value, Dataset
import torch
import torchaudio
import numpy as np
import re
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
print(f'Cuda Available = {torch.cuda.is_available()}\n')

# load the dataset
print("Loading pairs...")
df = pd.read_pickle('pickles/cosine_similarity/arabic_portuguese_train.pickle')

# get the sentences with a cosine similarity of 0.56 and up (in the sequence it will be -0.16 and down)
print("Filtering dataset...")
df = df[df['cos_sim'] <= -0.16]

# get 40000 random pairs
print("Sampling dataset at random...")
# df = df.sample(n=100, random_state=1)
df = df.sample(n=40000, random_state=1)

# load the sentences
df_ar_sentences = pd.read_pickle('pickles/embedding/arabic/arabic_train.pickle')
df_pt_sentences = pd.read_pickle('pickles/embedding/portuguese/portuguese_train.pickle')

# create two extra columns in the dataset, one for the arabic sentence and one for the portuguese sentence
print("Adding sentences to dataset...")
# add Arabic sentences to dataset
arabic_sentences_dict = dict(zip(df_ar_sentences['path'].values, df_ar_sentences['sentence'].values))
df['arabic_sentence'] = df[PATH_1].map(arabic_sentences_dict)

# add Portuguese sentences to dataset
portuguese_sentences_dict = dict(zip(df_pt_sentences['path'].values, df_pt_sentences['sentence'].values))
df['portuguese_sentence'] = df[PATH_2].map(portuguese_sentences_dict)

dataset = Dataset.from_pandas(df)

# append to each element in 'arabic' '/home/or/Desktop/arabic_new_dataset/train/',
# and to each element in 'portuguese' '/home/or/Desktop/portu_dataset/augmentations/train/'
print("Appending paths...")


def append_path(example):
    example['arabic'] = '/home/or/Desktop/arabic_new_dataset/train/' + example['arabic']
    example['portuguese'] = '/home/or/Desktop/portu_dataset/augmentations/train/' + example['portuguese']
    return example


dataset = dataset.map(append_path)

# drop the column 'cos_sim'
print("Dropping column 'cos_sim'...")
dataset = dataset.remove_columns(['cos_sim'])
features = Features(
    {
        "arabic": Audio(sampling_rate=48_000),
        "portuguese": Audio(sampling_rate=48_000),
        "arabic_sentence": Value("string"),
        "portuguese_sentence": Value("string"),
        "__index_level_0__": Value("string"),
    }
)

dataset = dataset.cast(features)

# divide the dataset into four datasets, so it will be easier to work with
part_1 = dataset.select(range(0, 10000))
part_2 = dataset.select(range(10000, 20000))
part_3 = dataset.select(range(20000, 30000))
part_4 = dataset.select(range(30000, 40000))


# merge the two datasets into one dataset according to the way we stated in the beginning of the script: audio +
# pause + audio, sentence + space + sentence
def merge(sample):
    # concatenate the audio
    audio_array = np.asarray(sample['arabic']['array'])
    audio_array_2 = np.asarray(sample['portuguese']['array'])
    sample['arabic'] = torch.cat((torch.from_numpy(audio_array), torch.zeros(24000),
                                  torch.from_numpy(audio_array_2)), dim=0)
    # concatenate the sentence
    sample['arabic_sentence'] = sample['arabic_sentence'] + ' ' + sample['portuguese_sentence']
    return sample


# merge
print("Merging datasets...")
part_1 = part_1.map(merge)
# save the dataset
print("Saving dataset to disk...")
part_1.save_to_disk('pickles/merged_dataset/arabic_portuguese/part_1')
del part_1

part_2 = part_2.map(merge)
# save the dataset
print("Saving dataset to disk...")
part_2.save_to_disk('pickles/merged_dataset/arabic_portuguese/part_2')
del part_2
part_3 = part_3.map(merge)
# save the dataset
print("Saving dataset to disk...")
part_3.save_to_disk('pickles/merged_dataset/arabic_portuguese/part_3')
del part_3
part_4 = part_4.map(merge)
# save the dataset
print("Saving dataset to disk...")
part_4.save_to_disk('pickles/merged_dataset/arabic_portuguese/part_4')
del part_4
# remove the columns that we don't need
# dataset = dataset.remove_columns(["portuguese", "portuguese_sentence", "__index_level_0__"])
#
# # rename the columns
# dataset = dataset.rename_column("arabic", "audio")
# dataset = dataset.rename_column("arabic_sentence", "sentence")
#
# # save the dataset
# print("Saving dataset to disk...")
# dataset.save_to_disk('pickles/merged_dataset/arabic_portuguese/low_cos_sim')
# print("Done!")
