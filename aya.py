MODEL_NAME = "CohereForAI/aya-23-8b"
import os
os.environ['HF_HOME'] = "~/air/models/arturo/huggingface/hub"
os.environ['HF_TOKEN']= "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"

from transformers import AutoModelForCausalLM, AutoTokenizer#, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
#from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import torch
#import bitsandbytes as bnb
#from datasets import load_dataset
#from trl import SFTTrainer
#from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import re
#import wandb

# Load Model
quantization_config = None
attn_implementation = None

model = AutoModelForCausalLM.from_pretrained(
          MODEL_NAME,
          quantization_config=quantization_config,
          attn_implementation=attn_implementation,
          torch_dtype=torch.bfloat16,
          device_map="cuda",
        )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_message_format(prompts):
  messages = []

  for p in prompts:
    messages.append(
        [{"role": "user", "content": p}]
      )

  return messages

def get_message_format_system(prompts):
  system_prompt = {"role": "system", "content": "You are a professional translator in the banking and finance domain."}
  messages = []

  for p in prompts:
    messages.append(
        [system_prompt, {"role": "user", "content": p}]
      )

  return messages

def get_message_format_few_shot(prompts):
  few_shot_prompt = [
    {"role": "user", "content": "Translate from English to Spanish: The cross-check of the outcome of the economic analysis with that of the monetary analysis clearly confirms that annual inflation rates are likely to remain well above levels"},
    {"role": "assistant", "content": "El contraste de los resultados del análisis económico con los del análisis monetario confirma claramente que es probable que las tasas de inflación interanual se mantengan durante algún tiempo muy por encima de los niveles compatibles con la estabilidad de precios y que , teniendo en cuenta el debilitamiento de la demanda , los riesgos al alza para la estabilidad de precios se han reducido ligeramente , aunque no han desaparecido ."}
  ]          
  messages = []

  for p in prompts:
    messages.append(
        few_shot_prompt + [{"role": "user", "content": p}]
      )

  return messages


def generate_aya_23(
      prompts,
      model,
      temperature=0.3,
      top_p=0.75,
      top_k=0,
      max_new_tokens=1024,
      prompt_type=0
    ):

  
  if prompt_type == 2:
    messages = get_message_format_system(prompts)
  elif prompt_type == 3:
    messages = get_message_format_few_shot(prompts)
  else:
    messages = get_message_format(prompts)
  
  text_chat=tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
  print(text_chat)
  input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
      )
  input_ids = input_ids.to(model.device)
  prompt_padded_len = len(input_ids[0])

  gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
      )

  # get only generated tokens
  gen_tokens = [
      gt[prompt_padded_len:] for gt in gen_tokens
    ]

  gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
  return gen_text


prompts = [
  'Translate from English to Spanish: "Rates are competitive, almost always the best in the market"',
  'Translate from English to Spanish: "As a consequence , the annual growth rate of M3 probably overstates the underlying pace of monetary expansion ."',
  'Translate from English to Spanish: "It is imperative to avoid broad-based second-round effects in price and wage-setting ."',
]

print("...example...")
generations = generate_aya_23(prompts, model, prompt_type=1)
for p, g in zip(prompts, generations):
  print(
      "PROMPT", p ,"RESPONSE", g, "\n", sep="\n"
    )

print("...system prompt example...")
generations_system = generate_aya_23(prompts, model, prompt_type=2)
for p, g in zip(prompts, generations_system):
  print(
      "PROMPT", p ,"RESPONSE", g, "\n", sep="\n"
    )


print("...few shot example...")
generations_few_shot = generate_aya_23(prompts, model, prompt_type=3)
for p, g in zip(prompts, generations_few_shot):
  print(
      "PROMPT", p ,"RESPONSE", g, "\n", sep="\n"
    )