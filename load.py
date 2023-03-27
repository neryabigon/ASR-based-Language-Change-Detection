import pandas as pd
import matplotlib.pyplot as plt
import os
from datasets import load_from_disk, concatenate_datasets
#
# print("----------------- Loading Datasets... -----------------")
# dataset = load_from_disk('pickles/merged_dataset/arabic_portuguese/try/ar_pt-high_1000')
# # data1 = load_from_disk('/media/or/Extreme SSD/wav2vec2/temp/part_1')
# # data2 = load_from_disk('/media/or/Extreme SSD/wav2vec2/temp/part_2')
# # data3 = load_from_disk('/media/or/Extreme SSD/wav2vec2/temp/part_3')
# # data4 = load_from_disk('/media/or/Extreme SSD/wav2vec2/temp/part_4')
# print("----------------- Loading Datasets complete. -----------------\n\n")
#
# print("----------------- Merging Datasets... -----------------")
# # dataset = concatenate_datasets([data1, data2, data3, data4])
# # remove the columns that we don't need
# dataset = dataset.remove_columns(["portuguese", "portuguese_sentence", "__index_level_0__"])
#
# # rename the columns
# dataset = dataset.rename_column("arabic", "audio")
# dataset = dataset.rename_column("arabic_sentence", "sentence")
#
# # save the dataset
# print("Saving dataset to disk...")
# dataset.save_to_disk('pickles/merged_dataset/arabic_portuguese/try/ar_pt-high_1000_ready')
# # dataset.save_to_disk('pickles/merged_dataset/arabic_portuguese/low_cos_sim')
# print("Done!")

df = pd.read_pickle('pickles/cosine_similarity/russian_spanish_test.pickle')
ls = df['cos_sim']

count75 = 0
count6 = 0
count5 = 0
count4 = 0
count_2 = 0
count_1 = 0
count = 0
for i in ls:
    # if i >= 0.75:
    #     count75 += 1
    # if i >= 0.6:
    #     count6 += 1
    if i >= 0.58:
        count5 += 1
    # if i >= 0.4:
    #     count4 += 1
    # if i <= -0.2:
    #     count_2 += 1
    if i <= -0.07:
        count_1 += 1

# print(f'0.75 and up: {count75}')
# print(f'0.6 and up: {count6}')
print(f'0.56 and up: {count5}')
# print(f'0.4 and up: {count4}')
print(f'-0.16 and down: {count_1}')
# print(f'-0.2 and down: {count_2}')
# print(f'0.75 and up: {count75}')


# # Create a histogram of the values
# plt.hist(ls, bins=1000)
#
# # Add labels and a title to the plot
# plt.xlabel('Cosine similarity')
# plt.ylabel('Frequency')
# plt.title('Distribution of cosine similarity values')
#
# # Show the plot
# plt.show()
