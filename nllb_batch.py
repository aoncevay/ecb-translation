import os
#os.environ['HF_HOME'] = "~/air/models/arturo/huggingface/hub"

from utils import languages, not_cleaned_langs
from transformers import NllbTokenizer
from tqdm.auto import tqdm, trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#transformers==4.33?
import gc
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from read import load_dataset
from datasets import Dataset

import sys

# Define the translate function
def translate_batch(texts, model, tokenizer, src_lang='eng_Latn', tgt_lang='spa_Latn', max_input_length=1024, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    inputs = inputs.to(model.device)
    result = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def run(model_name = 'facebook/nllb-200-distilled-600M', num_sample=0, batch_size=16):
    os.makedirs("results.2023", exist_ok=True)
    prefix = f"results.2023/{model_name.split('/')[-1]}"

    dataset = load_dataset(filename_prefix="data_2023/ECB", sample=num_sample, verbose=False)
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Translate in batches
    def batch_translate(batch, lang, src_lang, tgt_lang):
        translations = translate_batch(batch[lang], model, tokenizer, src_lang, tgt_lang)
        return {"translations": translations}

    #cleanup()
    model.cuda();

    results = {}
    #full_dataset = Dataset.from_dict(dataset)
    eng_dataset = Dataset.from_dict({"en": dataset["en"]})
    for lang, langcode in languages:
        if lang in not_cleaned_langs or os.path.exists(f"{prefix}.{lang}2en.txt"):
            print(lang, "skipped")
            continue
        print(lang)       
        results[f"en2{lang}"] = eng_dataset.map(
            lambda batch: batch_translate(batch, "en", 'eng_Latn', langcode),
            batched=True, batch_size=batch_size
            )["translations"]
        with open(f"{prefix}.en2{lang}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(results[f"en2{lang}"]))
        lang_dataset = Dataset.from_dict({lang: dataset[lang]})
        results[f"{lang}2en"] = lang_dataset.map(
            lambda batch: batch_translate(batch, lang, langcode, 'eng_Latn'),
            batched=True, batch_size=batch_size)["translations"]
        with open(f"{prefix}.{lang}2en.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(results[f"{lang}2en"]))


if __name__ == "__main__":
    argv = sys.argv
    model_name = argv[1] if len(argv) > 1 else 'facebook/nllb-200-3.3B' #'facebook/nllb-200-distilled-600M'
    run(model_name, num_sample=0)