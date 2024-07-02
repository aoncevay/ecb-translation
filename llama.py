import transformers
import torch
from read import load_dataset
from utils import languages_names
from tqdm.auto import tqdm
import os 
from translate import cleanup

os.environ['TRANSFORMERS_CACHE'] = "~/air/models/arturo"
token = "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"

def translate(pipeline, src_lang_name, tgt_lang_name, src_text):
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
        lines_txt_output = txt_output.split("\n")
        max_len = 0
        for line in lines_txt_output:
            if len(line.strip()) > max_len:
                txt_output = line
                max_len = len(line)
    txt_output = txt_output.strip()
    return txt_output

def translate_w_template(pipeline, messages_template, src_lang_name, tgt_lang_name, src_text):
    #messages = [
    #    {"role": "system", "content": "You are a professional translator in the banking and finance domain. Provide the required translation only."},
    #    {"role": "user", "content": f"{src_lang_name}: {src_text}\n{tgt_lang_name}: "},
    #]
    messages_template.append({"role": "user", "content": f"{src_lang_name}: {src_text}\n{tgt_lang_name}: "})
    output = pipeline(
        messages_template, 
        max_new_tokens=256,
    )
    txt_output = output[0]["generated_text"][-1]["content"].strip()
    if "\n" in txt_output:
        lines_txt_output = txt_output.split("\n")
        max_len = 0
        for line in lines_txt_output:
            if len(line.strip()) > max_len:
                txt_output = line
                max_len = len(line)
    txt_output = txt_output.strip()
    return txt_output


def run_llm_translate(model_id = "meta-llama/Meta-Llama-3-8B-Instruct", num_shot=1, num_sample=0, results_dir="results"):

    pipeline = transformers.pipeline(
        "text-generation", 
        token = token,
        model=model_id, 
        model_kwargs={"torch_dtype": torch.bfloat16}, 
        device_map="cuda"
    )

    dataset_examples = load_dataset(sample=num_sample+num_shot, verbose=False)
    #messages = [
    #    {"role": "system", "content": "You are a professional translator in the banking and finance domain. Provide the required translation only."},
    #    {"role": "user", "content": f"English: {dataset_examples['en'][0]}\nSpanish: {dataset_examples['es'][0]}"},
    #    {"role": "user", "content": "English: The cross-check of the outcome of the economic analysis with that of the monetary analysis clearly confirms that annual inflation rates are likely to remain well above levels consistent with price stability for some time and , when taking into account the weakening of demand , that upside risks to price stability have diminished somewhat , but they have not disappeared .\nSpanish: "},
    #]
    #output = pipeline(
    #    messages, 
    #    max_new_tokens=256,
    #)
    #print(output[0]["generated_text"])
    #print(output[0]["generated_text"][-1])

    os.makedirs(f"{results_dir}", exist_ok=True)
    prefix = f"{results_dir}/" + model_id.split("/")[-1] + f"{num_shot}shot"

    dataset = load_dataset(sample=num_sample, verbose=False)
    results = {}

    for lang, lang_name, _ in languages_names:
        print(lang, "en2xx")
        messages = [
            {"role": "system", "content": "You are a professional translator in the banking and finance domain."},
        ]
        for i in range(num_shot):
            messages.append({"role": "user", "content": f"Translate the following text from English into {lang_name}.\nEnglish: {dataset_examples['en'][i]}\n{lang_name}: {dataset_examples[lang][i]}"})
        results[f"en2{lang}"] = [translate_w_template(pipeline, messages, "English", lang, t) for t in tqdm(dataset["en"])]
        with open(f"{prefix}.en2{lang}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(results[f"en2{lang}"]))
        
        print(lang, "xx2en")
        messages = [
            {"role": "system", "content": "You are a professional translator in the banking and finance domain."},
        ]
        for i in range(num_shot):
            messages.append({"role": "user", "content": f"Translate the following text from {lang_name} into English.\n{lang_name}: {dataset_examples[lang][i]}\nEnglish: {dataset_examples['en'][i]}"})
        results[f"{lang}2en"] = [translate_w_template(pipeline, messages, lang, "English", t) for t in tqdm(dataset[lang])]
        with open(f"{prefix}.{lang}2en.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(results[f"{lang}2en"]))
            

if __name__ == "__main__":
    list_models = ["meta-llama/Meta-Llama-3-8B-Instruct", "Unbabel/TowerInstruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "CohereForAI/aya-23-8B", "bigscience/bloomz-7b1"] #, "mistralai/Mistral-7B-Instruct-v0.3"]
    list_num_shots = [1, 5, 10]
    for model_id in list_models:
        print("MODEL:", model_id)
        for num_shot in list_num_shots:
            print("  num_shot:", num_shot)
            run_llm_translate(model_id=model_id, num_shot=num_shot, num_sample=50,results_dir="results.smpl50")
            cleanup()
            print()