import transformers
import torch
from read import load_dataset
from utils import languages_names
from tqdm.auto import tqdm
import os 
os.environ['TRANSFORMERS_CACHE'] = "~/air/models/arturo"

def llama_translate(pipeline, src_lang_name, tgt_lang_name, src_text):
    messages = [
        {"role": "system", "content": "You are a professional translator in the banking and finance domain. Provide the required translation only."},
        {"role": "user", "content": f"{src_lang_name}: {src_text}\n{tgt_lang_name}: "},
    ]
    output = pipeline(
        messages, 
        max_new_tokens=256,
    )
    txt_output = output[0]["generated_text"][-1]["content"]
    if "\n" in txt_output:
        lines_txt_output = txt_output.strip("\n")
        max_len = 0
        for line in lines_txt_output:
            if len(line.strip()) > max_len:
                txt_output = line
                max_len = len(line)
    txt_output = txt_output.strip()
    return txt_output

def llama_translate_w_template(pipeline, messages_template, src_lang_name, tgt_lang_name, src_text):
    #messages = [
    #    {"role": "system", "content": "You are a professional translator in the banking and finance domain. Provide the required translation only."},
    #    {"role": "user", "content": f"{src_lang_name}: {src_text}\n{tgt_lang_name}: "},
    #]
    messages_template.append({"role": "user", "content": f"{src_lang_name}: {src_text}\n{tgt_lang_name}: "})
    output = pipeline(
        messages, 
        max_new_tokens=256,
    )
    txt_output = output[0]["generated_text"][-1]["content"]
    return txt_output



model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
token = "<TOKEN>"

pipeline = transformers.pipeline(
    "text-generation", 
    token = token,
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="cuda",
)

dataset_examples = load_dataset(sample=51, verbose=False)
messages = [
    {"role": "system", "content": "You are a professional translator in the banking and finance domain. Provide the required translation only."},
    {"role": "user", "content": f"English: {dataset_examples['en'][0]}\nSpanish: {dataset_examples['es'][0]}"},
    {"role": "user", "content": "English: The cross-check of the outcome of the economic analysis with that of the monetary analysis clearly confirms that annual inflation rates are likely to remain well above levels consistent with price stability for some time and , when taking into account the weakening of demand , that upside risks to price stability have diminished somewhat , but they have not disappeared .\nSpanish: "},
]
output = pipeline(
    messages, 
    max_new_tokens=256,
)
print(output[0]["generated_text"])
print(output[0]["generated_text"][-1])


os.makedirs("results.sample50", exist_ok=True)
prefix = "results.sample50/" + model_id.split("/")[-1]

dataset = load_dataset(sample=5, verbose=False)
results = {}

for lang, lang_name, _ in languages_names:
    print(lang, "en2xx")
    messages = [
        {"role": "system", "content": "You are a professional translator in the banking and finance domain. Provide the required translation only."},
        {"role": "user", "content": f"English: {dataset_examples['en'][0]}\n{lang_name}: {dataset_examples[lang][0]}"},
    ]
    results[f"en2{lang}"] = [llama_translate_w_template(pipeline, messages, "English", lang, t) for t in tqdm(dataset["en"])]
    with open(f"{prefix}.en2{lang}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results[f"en2{lang}"]))
    
    print(lang, "xx2en")
    messages = [
        {"role": "system", "content": "You are a professional translator in the banking and finance domain. Provide the required translation only."},
        {"role": "user", "content": f"{lang_name}: {dataset_examples[lang][0]}\nEnglish: {dataset_examples['en'][0]}"},
    ]
    results[f"{lang}2en"] = [llama_translate(pipeline, lang, "English", t) for t in tqdm(dataset[lang])]
    with open(f"{prefix}.{lang}2en.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results[f"{lang}2en"]))
        
