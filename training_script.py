from datasets import load_dataset, load_metric, Audio, ClassLabel, load_from_disk, Features, Value
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, \
    TrainingArguments, Trainer
import torch
import torchaudio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import re
import json
from torch.utils.tensorboard import SummaryWriter
from unidecode import unidecode

# the writer is responsible for tensorboard logging
writer = SummaryWriter(comment="_spanish_portuguese_high_augmented")

print("----------------- Checking if cuda is available... -----------------")
print(f'Cuda Available = {torch.cuda.is_available()}\n\n')

print("----------------- Loading Datasets complete. -----------------")
train = load_from_disk('pickles/merged_dataset/spanish_portuguese/high/augmented/train')
validation = load_from_disk('pickles/merged_dataset/spanish_portuguese/high/validation')
print("----------------- Loading Datasets complete. -----------------\n")


# small dataset for testing purposes only (10 samples)
# train = train.select(range(100))
# validation = validation.select(range(10))

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


print("\n----------------- Extracting all characters... -----------------")
vocab_train = train.map(extract_all_chars, batched=True, batch_size=128, keep_in_memory=True,
                        remove_columns=train.column_names)
vocab_test = validation.map(extract_all_chars, batched=True, batch_size=128, keep_in_memory=True,
                            remove_columns=validation.column_names)
print("----------------- Extracting all characters complete. -----------------\n\n")

# ----------------------------------- VOCAB -----------------------------------#

print("----------------- Preparing vocab... -----------------")
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

print(f'Vocab_dict: {vocab_dict}')

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(f'Vocab_len: {len(vocab_dict)}')

print("----------------- Preparing vocab complete. -----------------\n\n")

print("----------------- Saving vocab to jason... -----------------")
with open('vocab/spanish_portuguese_high_augmented.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

print("----------------- Saving vocab to jason complete. -----------------\n\n")

tokenizer = Wav2Vec2CTCTokenizer("./vocab/spanish_portuguese_high_augmented.json", unk_token="[UNK]", pad_token="[PAD]",
                                 word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


print("----------------- Preparing datasets... -----------------")
train = train.map(prepare_dataset, remove_columns=train.column_names, num_proc=4)  # maybe we'll have to reduce to 1
validation = validation.map(prepare_dataset, remove_columns=validation.column_names, num_proc=4)
print("\n\n----------------- Preparing datasets complete. -----------------\n\n")


# print("----------------- saving datasets... -----------------")
# train.save_to_disk('pickles/merged_dataset/spanish_portuguese/ready/low/train')
# validation.save_to_disk('pickles/merged_dataset/russian_portuguese/ready/low/validation')
# print("----------------- saving datasets complete. -----------\n\n")


# #  Loading from file
# print("----------------- Loading Datasets... -----------------")
# train = load_from_disk('pickles/merged_dataset/russian_spanish/ready/high/train')
# validation = load_from_disk('pickles/merged_dataset/russian_portuguese/ready/high/validation')
# print("----------------- Loading Datasets complete. ----------\n\n")
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# ----------------------------------- Loading Metrics -----------------------------------#
print("----------------- Loading Metrics... -----------------")
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")
print("----------------- Loading Metrics complete. -----------------\n\n")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


# ----------------------------------- Loading Model -----------------------------------#

print("----------------- Loading Model... -----------------")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
print("----------------- Loading Model complete. -----------------\n\n")

model.freeze_feature_encoder()

model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="spanish_portuguese_high_augmented",
    group_by_length=True,
    per_device_train_batch_size=8,  # if cuda is out of memory try decreasing batch size by half
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=761,  # change to num_of_samples / batch size in order to save on epoch
    eval_steps=100,
    logging_steps=10,
    learning_rate=3e-4,
    warmup_steps=200,
    save_total_limit=10,
    report_to='tensorboard'
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train,
    eval_dataset=validation,
    tokenizer=processor.feature_extractor,

)
print("----------------- Training... -----------------")
trainer.train()
# trainer.train(resume_from_checkpoint=True)  # to continue training from the last checkpoint
# trainer.train(resume_from_checkpoint=<path/to/checkpoint>)  # to continue training from specific checkpoint
print("----------------- Training complete. -----------------\n\n")

writer.close()
# To run the tensorboard run the following command in the folder containing the checkpoints: tensorboard --logdir=runs
