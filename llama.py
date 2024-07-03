import os
os.environ['HF_HOME'] = "~/air/models/arturo/huggingface/hub"
token = "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"

import transformers
import torch
from read import load_dataset
from utils import languages_names
from tqdm.auto import tqdm
from translate import cleanup

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


def run_llm_translate(model_id = "meta-llama/Meta-Llama-3-8B-Instruct", system_model=True, list_num_shots=[1,5,10], num_sample=0, results_dir="results"):

    pipeline = transformers.pipeline(
        "text-generation", 
        token = token,
        model=model_id, 
        model_kwargs={"torch_dtype": torch.bfloat16}, 
        device_map="cuda"
    )

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
    dataset = load_dataset(sample=num_sample, verbose=False)

    for num_shot in list_num_shots:
        dataset_examples = load_dataset(sample=num_sample+num_shot, verbose=False)
        prefix = f"{results_dir}/" + model_id.split("/")[-1] + f".{num_shot}shot"
        results = {}

        for lang, lang_name, _ in languages_names:
            print(lang, "en2xx")
            messages = []
            if system_model:
                messages.append({"role": "system", "content": "You are a professional translator in the banking and finance domain."})
            for i in range(num_shot):
                messages.append({"role": "user", "content": f"Translate the following text from English into {lang_name}: {dataset_examples['en'][i]}"}) 
                #messages.append({"role": "user", "content": f"Translate the following text from English into {lang_name}.\nEnglish: {dataset_examples['en'][i]}\n{lang_name}: "}) 
                messages.append({"role": "assistant", "content": f"{dataset_examples[lang][i]}"})

            results[f"en2{lang}"] = [translate_w_template(pipeline, messages, "English", lang, t) for t in tqdm(dataset["en"])]
            with open(f"{prefix}.en2{lang}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(results[f"en2{lang}"]))
            
            print(lang, "xx2en")
            messages = []
            if system_model:
                messages.append({"role": "system", "content": "You are a professional translator in the banking and finance domain."})
            for i in range(num_shot):
                messages.append({"role": "user", "content": f"Translate the following text from {lang_name} into English: {dataset_examples[lang][i]}"})
                #messages.append({"role": "user", "content": f"Translate the following text from {lang_name} into English.\n{lang_name}: {dataset_examples[lang][i]}\nEnglish: "})
                messages.append({"role": "assistant", "content": f"{dataset_examples['en'][i]}"})
            results[f"{lang}2en"] = [translate_w_template(pipeline, messages, lang, "English", t) for t in tqdm(dataset[lang])]
            with open(f"{prefix}.{lang}2en.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(results[f"{lang}2en"]))
            

if __name__ == "__main__":
    list_models = ["meta-llama/Meta-Llama-3-8B-Instruct", "CohereForAI/aya-23-8B", "bigscience/bloomz-7b1", "mistralai/Mixtral-8x7B-Instruct-v0.1"] #, "Unbabel/TowerInstruct-v0.2", , "mistralai/Mistral-7B-Instruct-v0.3"]
    list_system_model = [True, False, True, False]
    list_num_shots = [1, 5, 10]
    for model_id, system_model in zip(list_models, list_system_model):
        print("MODEL:", model_id)
        run_llm_translate(model_id=model_id, system_model= system_model, list_num_shots=list_num_shots, num_sample=50,results_dir="results.smpl50")
        cleanup()
        break

# ValueError: Could not load model mistralai/Mixtral-8x7B-Instruct-v0.1 with any of the following classes: (<class 'transformers.models.auto.modeling_auto.AutoModelForCausalLM'>, <class 'transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM'>).
# Aya: jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant/...
# You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
# BloomZ: No chat template is set for this tokenizer, falling back to a default class-level template. This is very error-prone, because models are often trained with templates different from the class default! Default chat templates are a legacy feature and will be removed in Transformers v4.43, at which point any code depending on them will stop working. We recommend setting a valid chat template before then to ensure that this model continues working without issues.
            