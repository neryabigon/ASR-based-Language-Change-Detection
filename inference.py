import json
import torch
import torchaudio
from datasets import load_dataset, Audio, load_from_disk, Features, Value, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from datasets import load_metric
from unidecode import unidecode

print("----------------- Checking if cuda is available... -----------------")
print(f'Cuda Available = {torch.cuda.is_available()}\n\n')

DEVICE = "cuda"
wer = load_metric("wer")
cer = load_metric("cer")

print('---------------- Loading Data... ---------------------')
test = load_from_disk('pickles/merged_dataset/spanish_portuguese/high/test')
print('---------------- Loading Data complete. ---------------------\n\n')
# small dataset for testing purposes only (10 samples)
# test_dataset = test.select(range(10))
test_dataset = test

print('---------------- Loading processor and model... ---------------------')

tokenizer = Wav2Vec2CTCTokenizer("./vocab/spanish_portuguese_high_augmented.json", unk_token="[UNK]",
                                 pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2ForCTC.from_pretrained('spanish_portuguese_high_augmented/checkpoint-7610')
model.to('cuda')
print('---------------- Loading processor and model complete. ---------------------\n\n')


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["speech"] = processor(audio, sampling_rate=16000).input_values[0]
    # print(batch["speech"])
    return batch


print('---------------- Processing the data... ---------------------')

test_dataset = test_dataset.map(prepare_dataset)
print('---------------- Processing the data complete. ---------------------\n\n')
# print('---------------- Saving dataset to disk... ---------------------')
# test_dataset.save_to_disk('/home/or/Desktop/rus/current/test')
# print('---------------- Saving dataset to disk complete. ---------------------\n\n')

# print('---------------- Loading Data from disk... ---------------------')
# test_dataset = load_from_disk('/home/or/Desktop/rus/current/test')
# print('---------------- Loading Data from disk complete. ---------------------\n\n')

# trying to apply the inference to the entire split
print('\n---------------- Inferencing... ---------------------')


# def predict_batch(batch):
#     inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model(inputs.input_values.to('cuda'), attention_mask=inputs.attention_mask.to('cuda')).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     predicted_batch = processor.batch_decode(predicted_ids)
#     return {'predicted_transcripts': predicted_batch}
#
#
# temp = Dataset.from_dict(test_dataset[:10000])
# print('\n---------------- Inferencing... ---------------------')
# predicted_sentences = temp.map(predict_batch, batched=False, remove_columns=temp.column_names)
#
# print('---------------- Inferencing complete. ---------------------\n\n')
# predicted_sentences = [prediction.get('predicted_transcripts')[0] for prediction in predicted_sentences]


def evaluate(batch):
    inputs = processor(batch["audio"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)

    return batch


# temp = Dataset.from_dict(test_dataset[:1000])

print("----------------- evaluating... -----------------")
result = test_dataset.map(evaluate, batched=False)
print("----------------- evaluating complete. -----------------\n\n")

predicted_sentences = [x[0] for x in result["pred_strings"]]
references = [x for x in result["sentence"]]


# predicted_sentences = predicted_sentences[:-1]  # only for arabic because the samples are not aligned
# references = references[1:]  # only for arabic because the samples are not aligned


# print('\n---------------- Inferencing... ---------------------')
# inputs = processor(test_dataset["speech"][:100], sampling_rate=16_000, return_tensors="pt", padding=True)
# with torch.no_grad():
#     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
#
# predicted_ids = torch.argmax(logits, dim=-1)
# predicted_sentences = processor.batch_decode(predicted_ids)
# print('---------------- Inferencing complete. ---------------------\n\n')


def same_length(references, predictions):
    # Zip the references and predictions, and filter by the condition
    # where the lengths are the same
    same_lengths = [(x, y) for x, y in zip(references, predictions) if len(x.split()) == len(y.split())]

    # Unzip the same_lengths list into two separate lists
    references_same_length, predictions_same_length = zip(*same_lengths)

    return list(references_same_length), list(predictions_same_length)


# print('---------------- extracting same length samples... ---------------------')
# references, predictions = same_length(predicted_sentences[:-1], test_dataset['sentence'][1:])
# print('\n---------------- extracting same length samples complete... -------------\n\n')

results_wer = []
results_cer = []

# for i, predicted_sentence in enumerate(predictions):
#     print(str(i) + "-" * 100)
#     print("Reference:", references[i])
#     print("Prediction:", predicted_sentence)
#
#     result_cer = cer.compute(predictions=[predicted_sentence], references=[references[i]])
#     result_wer = wer.compute(predictions=[predicted_sentence], references=[references[i]])
#     print("WER: " + str(result_wer))
#     print("CER: " + str(result_cer))
#     results_wer.append(result_wer)
#     results_cer.append(result_cer)

for i, predicted_sentence in enumerate(predicted_sentences):
    print(str(i) + "-" * 100)
    # print("Reference:", test_dataset[i + 1]["sentence"])
    print("Reference:", references[i])
    print("Prediction:", predicted_sentence)

    result_cer = cer.compute(predictions=[predicted_sentence], references=[references[i]])
    result_wer = wer.compute(predictions=[predicted_sentence], references=[references[i]])
    print("WER: " + str(result_wer))
    print("CER: " + str(result_cer))
    results_cer.append(result_cer)
    results_wer.append(result_wer)

num_of_bad_sample_wer = 0
num_of_sample_wer = 2000
num_of_bad_sample_cer = 0
num_of_sample_cer = 2000

for i in results_cer:
    if i >= 1.0:
        results_cer.remove(i)
        num_of_bad_sample_cer += 1

for i in results_wer:
    if i >= 1.0:
        results_wer.remove(i)
        num_of_bad_sample_wer += 1

num_of_sample_cer -= num_of_bad_sample_cer
num_of_sample_wer -= num_of_bad_sample_wer
print("Final WER: " + str(sum(results_wer) / num_of_sample_wer))
print("Final CER: " + str(sum(results_cer) / num_of_sample_cer))
