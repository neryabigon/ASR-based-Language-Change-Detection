import torch
from transformers import BertModel, BertTokenizerFast
import numpy as np
import pandas as pd
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

print('---------------- loading tokenizer and model... ------------------')
tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()
print('---------------- loading tokenizer and model complete. ------------------\n')
# english_sentences = [
#     "dog",
#     "Puppies are nice.",
#     "I enjoy taking long walks along the beach with my dog.",
# ]
#
#
# english_inputs = tokenizer(english_sentences, return_tensors="pt", padding=True)
# arabic_inputs = tokenizer(df['sentence'], return_tensors="pt", padding=True)

# df = pd.DataFrame([[],[],[]] ,columns=['A', 'B', 'C', 'D'])
# ls = [['common_voice_ar_19222711.mp3', 'عمي هو أخو أبي.', [[],[],[]]]]

print('\n-------------- preparing dataframe... ------------------\n')

ls = []
for i in range(len(df.index)):
    cur = [0, 0, 0]
    cur[0] = df['path'][i]
    cur[1] = df['sentence'][i]
    inputs = tokenizer(df['sentence'][i], return_tensors="pt", padding=True)
    with torch.no_grad():
        try:
            outputs = model(**inputs)
        except:
            print(f'Error on line: {i}')
    cur[2] = outputs
    ls.append(cur)

df_res = pd.DataFrame(ls, columns=['path', 'sentence', 'embedding'])
# print(df_res['embedding'])
print('-------------- preparing dataframe complete. ------------------\n')

print('-------------- writing to pickle... ------------------')
with open('portuguese_validation.pickle', 'wb') as f:
    pickle.dump(df_res, f)
print('-------------- writing to pickle complete. ------------------\n')
