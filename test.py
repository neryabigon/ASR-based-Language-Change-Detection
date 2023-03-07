import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle

print('---------------- loading data from csv... ------------------')
df = pd.read_csv("datasets csv/portuguese/dev.csv")
print('---------------- loading data from csv complete. ------------------\n')

print('---------------- removing columns... ------------------')
del df['client_id']
del df['audio']
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


# english_sentences = tf.constant(["good morning, what would you like to eat?"])
#
# italian_sentences = tf.constant(['buenos dias, que te gustaria comer?'])

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
print('\n-------------- preparing dataframe... ------------------\n')
ls = []
for i in range(len(df.index)):
    cur = [0, 0, 0]
    cur[0] = df['path'][i]
    cur[1] = df['sentence'][i]
    inputs = np.array(normalization(encoder(preprocessor(tf.constant([df['sentence'][i]])))["default"])[0])
    cur[2] = inputs
    ls.append(cur)
print(ls[:3])
df_res = pd.DataFrame(ls, columns=['path', 'sentence', 'embedding'])
print('-------------- preparing dataframe complete. ------------------\n')

print('-------------- writing to pickle... ------------------')
with open('portuguese_validation.pickle', 'wb') as f:
    pickle.dump(df_res, f)
print('-------------- writing to pickle complete. ------------------\n')

# english_embeds = encoder(preprocessor(english_sentences))["default"]
#
# italian_embeds = encoder(preprocessor(italian_sentences))["default"]

# For semantic similarity tasks, apply l2 normalization to embeddings
# english_embeds = normalization(english_embeds)
# italian_embeds = normalization(italian_embeds)

# english_embeds = np.array(english_embeds[0])
# italian_embeds = np.array(italian_embeds[0])

# cos_sim = dot(english_embeds, italian_embeds) / (norm(english_embeds) * norm(italian_embeds))
# print(cos_sim)
