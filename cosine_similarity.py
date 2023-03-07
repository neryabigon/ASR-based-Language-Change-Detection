import os

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
import time
from tqdm import tqdm

# change the languages before running the script
LANG_1 = 'russian'
LANG_2 = 'portuguese'

print('---------------- loading data from pickle... ------------------')
df_lang_1 = pd.read_pickle('pickles/embedding/russian/russian_train.pickle')
df_lang_2 = pd.read_pickle('pickles/embedding/portuguese/portuguese_train.pickle')
print('\n---------------- loading data from pickle complete. ------------------\n')

start_time = time.time()  # start timer

print(f'df_1_len: {len(df_lang_1.index)}')
print(f'df_2_len: {len(df_lang_2.index)}')

print('\n-------------- preparing list for dataframe... ------------------\n')

# Define the filename for the progress file
progress_file = 'progress.pkl'

# Check if a progress file exists, and load it if it does
if os.path.isfile(progress_file):
    with open(progress_file, 'rb') as f:
        i, j, ls = pickle.load(f)
else:
    i = 0
    j = 0
    ls = []

max = max(len(df_lang_1.index), len(df_lang_2.index))
min = min(len(df_lang_1.index), len(df_lang_2.index))

print(f'Resuming from crash point: i = {i}, j = {j}\n\n')

# for i in tqdm(range(10000)):
for i in tqdm(range(i, max)):
    inputs_lang_1 = df_lang_1['embedding'][i]
    norm_inputs_1 = norm(inputs_lang_1)
    # for j in range(10000):
    for j in range(j, min):
        cur = [0, 0, 0]

        cur[0] = df_lang_1['path'][i]
        cur[1] = df_lang_2['path'][j]
        inputs_lang_2 = df_lang_2['embedding'][j]

        if np.array_equal(inputs_lang_1, "ERROR") or np.array_equal(inputs_lang_2, "ERROR"):
            cur[2] = "ERROR"
        cur[1] = df_lang_2['path'][j]
        inputs_lang_2 = df_lang_2['embedding'][j]

        if np.array_equal(inputs_lang_1, "ERROR") or np.array_equal(inputs_lang_2, "ERROR"):
            cur[2] = "ERROR"
        else:
            cos_sim = dot(inputs_lang_1, inputs_lang_2) / (norm_inputs_1 * norm(inputs_lang_2))
            cur[2] = cos_sim
        if cur[2] <= -0.07 or cur[2] >= 0.58:
            ls.append(cur)

        # Save the progress to a file every 200,000,000 iterations
        if len(ls) % 200000000 == 0:
            with open(progress_file, 'wb') as f:
                pickle.dump((i, j, ls), f)
    j = 0

print('-------------- preparing list for dataframe complete. ------------------\n')

print('-------------- making final dataframe... ------------------')
df_res = pd.DataFrame(ls, columns=[LANG_1, LANG_2, 'cos_sim'])
print('-------------- making final dataframe complete. ------------------\n')

print(df_res[:3])

end_time = time.time()
print('Time taken:', end_time - start_time)

print('-------------- writing to pickle... ------------------')
with open('pickles/cosine_similarity/russian_portuguese_train.pickle', 'wb') as f:
    pickle.dump(df_res, f)
print('-------------- writing to pickle complete. ------------------\n')
