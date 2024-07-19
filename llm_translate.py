import os
os.environ['HF_HOME'] = "~/air/models/arturo/huggingface/hub"
os.environ['HF_TOKEN']= "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"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from utils import languages_names, not_cleaned_langs
from read import load_dataset
from translate import cleanup

def generate_pipeline(pipeline, tokenizer, messages_template):
    output = pipeline(
        messages_template, 
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
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


def generate(
        messages,
        model,
        tokenizer,
        temperature=0.3,
        top_p=0.75,
        top_k=0,
        max_new_tokens=1024,
        ):
    
    #text_chat=tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #print(text_chat)
    input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        )
    input_ids = input_ids.to(model.device)
    prompt_padded_len = len(input_ids[0])

    #attention_mask = input_ids['input_ids'].get('attention_mask', None)
    # If attention_mask is not provided by the tokenizer, generate it manually
    #if attention_mask is None:
    #    attention_mask = (input_ids != tokenizer.pad_token_id).long()


    gen_tokens = model.generate(
            input_ids,
            #attention_mask=attention_mask,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            #pad_token_id=tokenizer.eos_token_id
        )

    # get only generated tokens
    gen_tokens = [
        gt[prompt_padded_len:] for gt in gen_tokens
        ]

    gen_text_list = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    gen_text = gen_text_list[0].strip()
    if "\n" in gen_text:
        all_gen_text = gen_text.split("\n")
        max_len = 0
        for i_gen_text in all_gen_text:
            if len(i_gen_text) > max_len:
                gen_text = i_gen_text
                max_len = len(i_gen_text)
        if max_len == 0:
            print("error in generation:", all_gen_text)
        gen_text_list = [gen_text]
    return gen_text_list


"""
def get_message_format_system_few_shot(prompts):
    system_prompt = {"role": "system", "content": "You are a professional translator in the banking and finance domain."}
    messages = []

    for p in prompts:
        messages.append(
            [system_prompt, {"role": "user", "content": p}]
        )

    return messages
"""
def get_message_format_few_shot(prompts, src_name, tgt_name, src_examples, tgt_examples, system_prompt=False):
    #few_shot_prompt = [
    #    {"role": "user", "content": "Translate from English to Spanish: The cross-check of the outcome of the economic analysis with that of the monetary analysis clearly confirms that annual inflation rates are likely to remain well above levels"},
    #    {"role": "assistant", "content": "El contraste de los resultados del análisis económico con los del análisis monetario confirma claramente que es probable que las tasas de inflación interanual se mantengan durante algún tiempo muy por encima de los niveles compatibles con la estabilidad de precios y que , teniendo en cuenta el debilitamiento de la demanda , los riesgos al alza para la estabilidad de precios se han reducido ligeramente , aunque no han desaparecido ."}
    #]
    few_shot_prompt = []
    for src_e, tgt_e in zip(src_examples, tgt_examples):
        few_shot_prompt.append({"role": "user", "content": f'Translate from {src_name} to {tgt_name}: "{src_e}"'})
        few_shot_prompt.append({"role": "assistant", "content": f'"{tgt_e}"'})

    messages = []

    for p in prompts:
        if not system_prompt:
            messages.append(
                few_shot_prompt + [{"role": "user", "content": p}]
            )
        else:
            messages.append(
                [{"role": "system", "content": "You are a professional translator in the banking and finance domain."}] 
                + few_shot_prompt + [{"role": "user", "content": p}]
            )

    return messages


def run(model_id, list_num_shots=[1,5], num_sample=0, results_dir="results"):
    # Load Model
    quantization_config = None
    attn_implementation = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eos_token_id = tokenizer.eos_token_id
    if "llama" in model_id:
        llm_pipeline = pipeline(
            "text-generation", 
            model=model_id, 
            model_kwargs={"torch_dtype": torch.bfloat16, "pad_token_id": eos_token_id}, 
            device_map="cuda"
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                )
        
    system_prompt = True
    if "mistral" in model_id:
        # Option 1: Use EOS token as padding token
        tokenizer.pad_token = tokenizer.eos_token
        system_prompt = False

    os.makedirs(f"{results_dir}", exist_ok=True)
    dataset = load_dataset(filename_prefix="data_2023/ECB", sample=num_sample, verbose=False)


    for num_shot in list_num_shots:
        print("  num_shot", num_shot)
        dataset_examples = load_dataset(filename_prefix="data_2023/ECB", sample=num_sample+num_shot, verbose=False)
        prefix = f"{results_dir}/" + model_id.split("/")[-1] + f".{num_shot}shot"
        results = {}

        for lang, lang_name, _ in languages_names:
            if lang in not_cleaned_langs or os.path.exists(f"{prefix}.{lang}2en.txt"):
                print(lang, "skipped")
                continue
            print(lang, "en2xx")
            prompts = [f"Translate the following text from English into {lang_name}: {text}" for text in dataset[lang]]
            messages = get_message_format_few_shot(prompts, "English", lang_name, dataset_examples["en"][:num_shot], dataset_examples[lang][:num_shot], system_prompt=system_prompt)
            try:
                #results[f"en2{lang}"] = generate(messages, model, tokenizer)
                results[f"en2{lang}"] = []
                for m in messages:
                    if "llama" in model_id:
                        results[f"en2{lang}"].extend(generate_pipeline(llm_pipeline, tokenizer, m))
                    else:
                        results[f"en2{lang}"].extend(generate([m], model, tokenizer))
                with open(f"{prefix}.en2{lang}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(results[f"en2{lang}"]))
            except Exception as e:
                print(e)
            
            print(lang, "xx2en")
            prompts = [f"Translate the following text from {lang_name} into English: {text}" for text in dataset["en"]]
            messages = get_message_format_few_shot(prompts, lang_name, "English", dataset_examples[lang][:num_shot], dataset_examples["en"][:num_shot], system_prompt=system_prompt)
            try:
                #results[f"{lang}2en"] = generate(messages, model, tokenizer)
                results[f"{lang}2en"] = []
                for m in messages:
                    if "llama" in model_id:
                        results[f"en2{lang}"].extend(generate_pipeline(llm_pipeline, tokenizer, m))
                    else:
                        results[f"{lang}2en"].extend(generate([m], model, tokenizer))
                with open(f"{prefix}.{lang}2en.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(results[f"{lang}2en"]))
            except Exception as e:
                print(e)
            

if __name__ == "__main__":

    for model_name in ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]: #, "CohereForAI/aya-23-8B",
        print("MODEL:", model_name)
        run(model_name, list_num_shots=[1], num_sample=51, results_dir="results.2023")
        cleanup()