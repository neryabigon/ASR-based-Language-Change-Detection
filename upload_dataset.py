from datasets import load_from_disk

TOKEN = 'token'

dataset = load_from_disk('pickles/merged_dataset/spanish_portuguese/high/train')

dataset.push_to_hub("nerya/spanish_portuguese_high_similarity_train", private=True)