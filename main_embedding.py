import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle

print('---------------- loading data from csv... ------------------')
df = pd.read_csv("datasets csv/spanish/chosen_train.csv")
print('---------------- loading data from csv complete. ------------------\n')

print('---------------- removing columns... ------------------')
del df['client_id']
# del df['audio']  # russian csv doesn't contains the 'audio' column
del df['up_votes']
del df['down_votes']
del df['age']
del df['gender']
del df['accents']
del df['locale']
del df['segment']
print('---------------- removing columns complete. ------------------\n')


def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds / norms


print('---------------- loading tokenizer and model... ------------------')
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
print('---------------- loading tokenizer and model complete. ------------------\n')

print('\n-------------- preparing dataframe... ------------------\n')
ls = []

for i in range(len(df.index)):
    cur = [0, 0, 0]
    cur[0] = df['path'][i]
    cur[1] = df['sentence'][i]
    try:
        outputs = np.array(normalization(encoder(preprocessor(tf.constant([df['sentence'][i]])))["default"])[0])
    except:
        print(f'Error on line: {i}')
        outputs = 'ERROR'
    cur[2] = outputs
    ls.append(cur)

df_res = pd.DataFrame(ls, columns=['path', 'sentence', 'embedding'])
print(df_res[:5])
print('-------------- preparing dataframe complete. ------------------\n')

print('-------------- writing to pickle... ------------------')
with open('pickles/embedding/spanish/spanish_train.pickle', 'wb') as f:
    pickle.dump(df_res, f)
print('-------------- writing to pickle complete. ------------------\n')
