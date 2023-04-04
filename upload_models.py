from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, AutoModel

MODEL = 'russian_portuguese_high_augmented/checkpoint-7610'
VOCAB = "vocab/russian_portu_high_augmented.json"
save_directory = "./models_ready_to_upload/russian_portuguese_high_similarity_augmented"
TOKEN = 'token'

tokenizer = Wav2Vec2CTCTokenizer(VOCAB, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2ForCTC.from_pretrained(MODEL)

model.save_pretrained(save_directory, push_to_hub=True, use_auth_token=TOKEN)
processor.save_pretrained(save_directory, push_to_hub=True, use_auth_token=TOKEN)


