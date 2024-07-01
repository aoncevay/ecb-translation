import transformers
import torch
from read import load_dataset
from utils import languages_names
from tqdm.auto import tqdm
import os 
os.environ['TRANSFORMERS_CACHE'] = "~/air/models/arturo"

def llama_translate(pipeline, src_lang_name, tgt_lang_name, src_text):
    message = f"{src_lang_name}: {src_text}\n{tgt_lang_name}: "
    output = pipeline(message)
    return output

model_id = "meta-llama/Meta-Llama-3-8B"
token = "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"

pipeline = transformers.pipeline(
    "text-generation", 
    token = token,
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="cuda"
)

message = "English: Hello World!\nSpanish: "
print(pipeline(message))


os.makedirs("results.sample50", exist_ok=True)
prefix = "results.sample50/" + model_id.split("/")[-1]
dataset = load_dataset(sample=50)
results = {}

for lang, lang_name, _ in languages_names:
    print(lang, "en2xx")
    results[f"en2{lang}"] = [pipeline(f"English: {t}\n{lang_name}: ").strip() for t in tqdm(dataset["en"])]
    with open(f"{prefix}.en2{lang}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results[f"en2{lang}"]))
    print(lang, "xx2en")
    results[f"{lang}2en"] = [pipeline(f"{lang_name}: {t}\nEnglish: ").strip() for t in tqdm(dataset[lang])]
    with open(f"{prefix}.{lang}2en.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results[f"{lang}2en"]))
        