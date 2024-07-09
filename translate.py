import os
os.environ['HF_HOME'] = "~/air/models/arturo/huggingface/hub"

from utils import languages, not_cleaned_langs
from transformers import NllbTokenizer
from tqdm.auto import tqdm, trange
from transformers import AutoModelForSeq2SeqLM
#transformers==4.33?
import gc
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from read import load_dataset

import sys



def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def translate(text, model, tokenizer, src_lang='eng_Latn', tgt_lang='spa_Latn', a=16, b=1.5, max_input_length=1024, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        **kwargs
    )
    #print(inputs.input_ids.shape[1], result.shape[1])
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def run(model_name = 'facebook/nllb-200-distilled-600M', num_sample=51):
    os.makedirs("results.2023", exist_ok=True)
    prefix = f"results.2023/{model_name.split('/')[-1]}"

    dataset = load_dataset(filename_prefix="data_2023/ECB", sample=num_sample, verbose=False)
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    cleanup()
    model.cuda();

    results = {}
    for lang, langcode in languages:
        if lang in not_cleaned_langs:
            continue
        print(lang)
        results[f"en2{lang}"] = [translate(t, model, tokenizer, 'eng_Latn', langcode)[0] for t in tqdm(dataset["en"])]
        with open(f"{prefix}.en2{lang}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(results[f"en2{lang}"]))
        results[f"{lang}2en"] = [translate(t, model, tokenizer, langcode, 'eng_Latn')[0] for t in tqdm(dataset[lang])]
        with open(f"{prefix}.{lang}2en.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(results[f"{lang}2en"]))


if __name__ == "__main__":
    argv = sys.argv
    model_name = argv[1] if len(argv) > 1 else 'facebook/nllb-200-3.3B' #'facebook/nllb-200-distilled-600M'
    run(model_name, num_sample=51)