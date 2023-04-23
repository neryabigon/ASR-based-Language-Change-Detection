# Crossing Language Identification: Multilingual ASR Framework
Based on Semantic Dataset Creation & Wav2Vec 2.0

This study proposes an innovative methodology to enhance the performance of multilingual Automatic Speech Recognition
(ASR) systems by capitalizing on the high semantic similarity between sentences across different languages and eliminating the
requirement for Language Identification (LID). To achieve this, special bilingual datasets were created from the Mozzila Common
Voices datasets in Spanish, Russian, and Portuguese. The process involves computing sentence embeddings using Language agnostic BERT and selecting sentence pairs based on high and low cosine similarity. Subsequently, we train the Wav2vec 2.0
XLSR53 model on these datasets and assess its performance utilizing Character Error Rate (CER) and Word Error Rate (WER)
metrics. The experimental results indicate that models trained on high-similarity samples consistently surpass their low-similarity
counterparts, emphasizing the significance of high semantic similarity data selection for precise and dependable ASR performance.
Furthermore, the elimination of LID contributes to a simplified system with reduced computational costs and the capacity for real time text output. The findings of this research offer valuable insights for the development of more efficient and accurate multilingual
ASR systems, particularly in real-time and on-device applications.


# Framework
![image](https://user-images.githubusercontent.com/66886354/233838229-dc2516f5-ab72-449e-811c-454be788000e.png)





## Citation - TBD

If you use this code for your research, please cite our paper:

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={},
  organization={}
}
```
