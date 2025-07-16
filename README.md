# NLLB_200_600m_hin2kang_v2

This is a **distilled, fine-tuned version of Meta’s NLLB-200 (600M)** for low-resource **Hindi → Kangri** machine translation. The model achieves a **BLEU score of 26**, making it one of the best-performing open-source Kangri translators.

##  Model Summary

| Attribute         | Value                          |
|------------------|--------------------------------|
| Base Model       | facebook/nllb-200-distilled-600M |
| Language Pair    | Hindi (hin_Deva) → Kangri (kang_Deva) |
| BLEU Score       | 27.04 (on test set of 1k samples) |
| Parameters       | ~615M                          |
| Tokenizer        | Custom with Kangri vocab       |
| Architecture     | Encoder-Decoder (Transformer)  |

## Installation & Setup

## Install the required packages:
```bash
pip install transformers sentencepiece accelerate
```

##Usage (with Hugging Face Transformers)
```
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "cloghost/nllb-200-distilled-600M-hin-kang-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="hin_Deva", tgt_lang="kang_Deva")

# Translate a Hindi sentence
result = translator("यह एक उदाहरण वाक्य है।")
print(result[0]['translation_text'])

```
## Dataset

<https://www.kaggle.com/datasets/ikarun/hindi2kangri-v2>

## Training Details
Dataset: 51,000 aligned Hindi-Kangri sentence pairs.

Tokenizer: Extended SentencePiece tokenizer to include Kangri tokens (trained on ~250k-line corpus).

##Training:

Epochs: 3

Batch Size: 8 (fp16)

Optimizer: AdamW (lr=3e-5, weight_decay=0.01)

Dropout: 0.1

Label Smoothing: ε = 0.1

Beam Search: num_beams=4, no_repeat_ngram=3, repetition_penalty=1.2

LR Scheduler: Cosine

## Evaluation Metrics
Metric	Score
BLEU	27.04
ROUGE-L	4.75%
METEOR	43.6%
ChrF	53.93
BERTScore-F1	93.38%


## Why This Model?
First high-performing open-source Hindi ⇨ Kangri translation model.

Based on NLLB (No Language Left Behind), optimized for low-resource languages.

Built and fine-tuned with attention to fluency and faithfulness to Kangri syntax.

## Limitations
Currently unidirectional (Hindi → Kangri only).

May generate incorrect spellings for unseen Kangri entities.

Doesn't handle code-mixed Hindi-Kangri inputs.

Files in this Repository
pytorch_model.bin: Trained model weights

tokenizer.json: Extended tokenizer with Kangri vocabulary

special_tokens_map.json, tokenizer_config.json: Tokenizer configs


## Citation
If you use this model in research or production, please cite:

@misc{nllb-kangri-2025,
  title={NLLB-Kangri: Fine-Tuned NLLB-200 for Hindi to Kangri Translation},
  author={Karun Sharma},
  year={2025},
  url={https://huggingface.co/cloghost/NLLB-KANG-2.4_BLEU27}
}
## Contributing
Found an issue or improvement? Raise an issue or pull request at github.com/cloghost.
