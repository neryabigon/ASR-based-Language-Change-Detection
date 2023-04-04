import pandas as pd
from datasets import Audio, Features, Value, Dataset
import torch
import torchaudio
import numpy as np
import re
import pyarabic.araby as araby
from unidecode import unidecode
from pathlib import Path
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
PATH_1 = 'russian'
PATH_2 = 'portuguese'
# PATH_1 = 'spanish'
# PATH_2 = 'spanish'


# load the dataset
print("Loading pairs...")
# df = pd.read_pickle('pickles/cosine_similarity/spanish_portuguese_train_augmented.pickle')
# df_validation = pd.read_pickle('pickles/cosine_similarity/russian_portuguese_validation.pickle')
df_test = pd.read_pickle('pickles/cosine_similarity/russian_portuguese_test.pickle')

'''
russian_portuguese:
get the sentences with a cosine similarity of 0.58 and up (the low will be -0.07 and down)

russian_spanish:
get the sentences with a cosine similarity of 0.56 and up (the low will be -0.07 and down)
'''
# print("Filtering dataset...")
# high
# df = df[df['cos_sim'] >= 0.58]
# df_validation = df_validation[df_validation['cos_sim'] >= 0.58]
df_test = df_test[df_test['cos_sim'] >= 0.58]


# low
# df = df[df['cos_sim'] <= -0.07]
# df_validation = df_validation[df_validation['cos_sim'] <= -0.07]
# df_test = df_test[df_test['cos_sim'] <= -0.07]

# sample the dataset at random
'''
russian_portuguese high & low: train = 10000, validation = 2000, test = 2000
russian_spanish high & low: train = 10000, validation = 1500, test = 1500
spanish_portuguese high & low: train = 10000, validation = 2000, test = 2000
'''
print("Sampling dataset at random...")
# df_1 = df.sample(n=10000, random_state=1)
# df_1 = df_1.append(df[df['cos_sim'] >= 0.999])
# df = df_1
# df_validation = df_validation.sample(n=2000, random_state=1)
df_test = df_test.sample(n=2000, random_state=1)

# load the sentences
# df_ru_sentences = pd.read_pickle('pickles/embedding/russian/russian_train.pickle')
# df_ru_sentences_validation = pd.read_pickle('pickles/embedding/russian/russian_validation.pickle')
df_ru_sentences_test = pd.read_pickle('pickles/embedding/russian/russian_test.pickle')

# df_pt_sentences = pd.read_pickle('pickles/embedding/portuguese/portuguese_train.pickle')
# df_pt_sentences_validation = pd.read_pickle('pickles/embedding/portuguese/portuguese_validation.pickle')
df_pt_sentences_test = pd.read_pickle('pickles/embedding/portuguese/portuguese_test.pickle')

# df_sp_sentences = pd.read_pickle('pickles/embedding/spanish/spanish_train.pickle')
# df_sp_sentences_validation = pd.read_pickle('pickles/embedding/spanish/spanish_validation_5000.pickle')
# df_sp_sentences_test = pd.read_pickle('pickles/embedding/spanish/spanish_test_5000.pickle')

# chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\]'
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�؟]'


# the differnce between this function and the next one is
# that this one is for the test set and doesn't add a space at the end of the sentence
def remove_special_characters_uni_test(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    batch["sentence"] = unidecode(batch["sentence"])  # remove pronunciation signs in portuguese
    return batch
def remove_special_characters_uni(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    batch["sentence"] = unidecode(batch["sentence"])  # remove pronunciation signs in portuguese
    return batch


# remove special characters from the sentences in df_ar_sentences, and from the sentences in df_pt_sentences
print("Removing special characters from sentences...")
# remove special characters from the sentences in df_ar_sentences

# df_ru_sentences = df_ru_sentences.apply(remove_special_characters_uni, axis=1)
# df_ru_sentences_validation = df_ru_sentences_validation.apply(remove_special_characters_uni, axis=1)
df_ru_sentences_test = df_ru_sentences_test.apply(remove_special_characters_uni_test, axis=1)

# df_pt_sentences = df_pt_sentences.apply(remove_special_characters_uni, axis=1)
# df_pt_sentences_validation = df_pt_sentences_validation.apply(remove_special_characters_uni, axis=1)
df_pt_sentences_test = df_pt_sentences_test.apply(remove_special_characters_uni_test, axis=1)

# df_sp_sentences = df_sp_sentences.apply(remove_special_characters_uni, axis=1)
# df_sp_sentences_validation = df_sp_sentences_validation.apply(remove_special_characters_uni, axis=1)
# df_sp_sentences_test = df_sp_sentences_test.apply(remove_special_characters_uni_test, axis=1)
# create two extra columns in the dataset, one for the arabic sentence and one for the portuguese sentence
print("Adding sentences to dataset...")

# add russian sentences to dataset
# train
# the first element in the following zip should be: Path(df_ru_sentences['path'].values[0]).stem + ".mp3" but instead of 0 it should be appl
# add a column to df 'name' which is the name of the file without the full path by using Path(x).stem + 'mp3'
# df['ru_name'] = df[PATH_1].apply(lambda x: Path(x).stem + ".mp3")
# russian_sentences_dict = dict(zip(df_ru_sentences['path'].values, df_ru_sentences['sentence'].values))
# df['russian_sentence'] = df['ru_name'].map(russian_sentences_dict)
# # validation
# russian_sentences_dict_validation = dict(zip(df_ru_sentences_validation['path'].values, df_ru_sentences_validation['sentence'].values))
# df_validation['russian_sentence'] = df_validation[PATH_1].map(russian_sentences_dict_validation)
# # test
russian_sentences_dict_test = dict(zip(df_ru_sentences_test['path'].values, df_ru_sentences_test['sentence'].values))
df_test['russian_sentence'] = df_test[PATH_1].map(russian_sentences_dict_test)

# add Portuguese sentences to dataset
# train


# add a column to df 'name' which is the name of the file without the full path by using Path(x).stem + 'mp3'
# df['pt_name'] = df[PATH_2].apply(lambda x: Path(x).stem + ".mp3")
# portuguese_sentences_dict = dict(zip(df_pt_sentences['path'].values, df_pt_sentences['sentence'].values))
# df['portuguese_sentence'] = df['pt_name'].map(portuguese_sentences_dict)
# validation
# portuguese_sentences_dict_validation = dict(zip(df_pt_sentences_validation['path'].values, df_pt_sentences_validation['sentence'].values))
# df_validation['portuguese_sentence'] = df_validation[PATH_2].map(portuguese_sentences_dict_validation)
# test
portuguese_sentences_dict_test = dict(zip(df_pt_sentences_test['path'].values, df_pt_sentences_test['sentence'].values))
df_test['portuguese_sentence'] = df_test[PATH_2].map(portuguese_sentences_dict_test)

# add Spanish sentences to dataset
# train
# df['sp_name'] = df[PATH_1].apply(lambda x: Path(x).stem + ".mp3")
# spanish_sentences_dict = dict(zip(df_sp_sentences['path'].values, df_sp_sentences['sentence'].values))
# df['spanish_sentence'] = df['sp_name'].map(spanish_sentences_dict)
# validation
# spanish_sentences_dict_validation = dict(zip(df_sp_sentences_validation['path'].values, df_sp_sentences_validation['sentence'].values))
# df_validation['spanish_sentence'] = df_validation[PATH_1].map(spanish_sentences_dict_validation)
# test
# spanish_sentences_dict_test = dict(zip(df_sp_sentences_test['path'].values, df_sp_sentences_test['sentence'].values))
# df_test['spanish_sentence'] = df_test[PATH_1].map(spanish_sentences_dict_test)
# dataset = Dataset.from_pandas(df)
# dataset = dataset.remove_columns(["sp_name", "pt_name", "cos_sim"])
# dataset_validation = Dataset.from_pandas(df_validation)
dataset_test = Dataset.from_pandas(df_test)

# append to each element in 'spanish' '/home/or/Desktop/spanish/chosen_train/'
# and to each element in 'portuguese' '/home/or/Desktop/portu_dataset/augmentations/train/'
# and to each element in 'russian' '/home/or/Desktop/russian/train/'
print("Appending paths...")


def append_path(example):
    # example['spanish'] = '/home/or/Desktop/spanish/chosen_train/' + example['spanish']
    example['russian'] = '/home/or/Desktop/russian/train/' + example['russian']
    example['portuguese'] = '/home/or/Desktop/portu_dataset/augmentations/train/' + example['portuguese']
    return example

def append_path_validation(example):
    # example['spanish'] = '/home/or/Desktop/spanish/dev/' + example['spanish']
    example['russian'] = '/home/or/Desktop/russian/dev/' + example['russian']
    example['portuguese'] = '/home/or/Desktop/portu_dataset/augmentations/validation/' + example['portuguese']
    return example

def append_path_test(example):
    example['spanish'] = '/home/or/Desktop/spanish/test/' + example['spanish']
    # example['russian'] = '/home/or/Desktop/russian/test/' + example['russian']
    # example['portuguese'] = '/home/or/Desktop/portu_dataset/augmentations/test/' + example['portuguese']
    return example

# dataset = dataset.map(append_path)
# dataset_validation = dataset_validation.map(append_path_validation)
# dataset_test = dataset_test.map(append_path_test)

# drop the column 'cos_sim'
print("Dropping column 'cos_sim'...")
# dataset = dataset.remove_columns(['cos_sim'])
# dataset_validation = dataset_validation.remove_columns(['cos_sim'])
dataset_test = dataset_test.remove_columns(['cos_sim'])

features = Features(
    {
        # "spanish": Audio(sampling_rate=48_000),
        "russian": Audio(sampling_rate=48_000),
        "portuguese": Audio(sampling_rate=48_000),
        # "spanish_sentence": Value("string"),
        "russian_sentence": Value("string"),
        "portuguese_sentence": Value("string"),
        "__index_level_0__": Value("string"),
    }
)

# dataset = dataset.cast(features)
# dataset_validation = dataset_validation.cast(features)
dataset_test = dataset_test.cast(features)


print("Casting columns to 16khz...")
# dataset = dataset.cast_column("russian", Audio(sampling_rate=16_000))
# dataset = dataset.cast_column("portuguese", Audio(sampling_rate=16_000))
# dataset = dataset.cast_column("spanish", Audio(sampling_rate=16_000))

# dataset_validation = dataset_validation.cast_column("russian", Audio(sampling_rate=16_000))
# dataset_validation = dataset_validation.cast_column("portuguese", Audio(sampling_rate=16_000))
# dataset_validation = dataset_validation.cast_column("spanish", Audio(sampling_rate=16_000))

dataset_test = dataset_test.cast_column("russian", Audio(sampling_rate=16_000))
dataset_test = dataset_test.cast_column("portuguese", Audio(sampling_rate=16_000))
# dataset_test = dataset_test.cast_column("spanish", Audio(sampling_rate=16_000))



# merge the two datasets into one dataset according to the way we stated in the beginning of the script: audio +
# pause + audio, sentence + space + sentence
def merge(sample):
    # concatenate the audio
    # audio_array = np.asarray(sample['spanish']['array'])
    audio_array = np.asarray(sample['russian']['array'])
    audio_array_2 = np.asarray(sample['portuguese']['array'])
    # sample['spanish'] = torch.cat((torch.from_numpy(audio_array), torch.zeros(24000), torch.from_numpy(audio_array_2)),dim=0)
    sample['russian'] = torch.cat((torch.from_numpy(audio_array), torch.zeros(24000), torch.from_numpy(audio_array_2)), dim=0)
    # concatenate the sentence
    # sample['spanish_sentence'] = sample['spanish_sentence'] + ' ' + sample['portuguese_sentence']
    sample['russian_sentence'] = sample['russian_sentence'] + ' ' + sample['portuguese_sentence']
    # sample['russian_sentence'] = sample['russian_sentence'] + ' ' + sample['spanish_sentence']
    return sample


print("Merging datasets...")
# dataset = dataset.map(merge)
# dataset_validation = dataset_validation.map(merge)
dataset_test = dataset_test.map(merge)

# remove the columns that we don't need
# dataset = dataset.remove_columns(["portuguese", "portuguese_sentence", "__index_level_0__"])
# dataset_validation = dataset_validation.remove_columns(["portuguese", "portuguese_sentence", "__index_level_0__"])
dataset_test = dataset_test.remove_columns(["portuguese", "portuguese_sentence", "__index_level_0__"])
# dataset = dataset.remove_columns(["spanish", "spanish_sentence", "__index_level_0__"])
# dataset_validation = dataset_validation.remove_columns(["spanish", "spanish_sentence", "__index_level_0__"])
# dataset_test = dataset_test.remove_columns(["spanish", "spanish_sentence"])
# rename the columns
# dataset = dataset.rename_column("spanish", "audio")
# dataset = dataset.rename_column("spanish_sentence", "sentence")
# dataset = dataset.rename_column("russian", "audio")
# dataset = dataset.rename_column("russian_sentence", "sentence")

# dataset_validation = dataset_validation.rename_column("russian", "audio")
# dataset_validation = dataset_validation.rename_column("russian_sentence", "sentence")
dataset_test = dataset_test.rename_column("russian", "audio")
dataset_test = dataset_test.rename_column("russian_sentence", "sentence")
# dataset_test = dataset_test.rename_column("spanish", "audio")
# dataset_test = dataset_test.rename_column("spanish_sentence", "sentence")

print("Saving dataset to disk...")
# dataset.save_to_disk('pickles/merged_dataset/spanish_portuguese/high/augmented/train')
# dataset_validation.save_to_disk('pickles/merged_dataset/russian_portuguese/high/validation')
dataset_test.save_to_disk('pickles/merged_dataset/russian_portuguese/high/test')



