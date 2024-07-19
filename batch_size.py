import os
#os.environ['HF_HOME'] = "~/air/models/arturo/huggingface/hub"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

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

# Load your model and tokenizer
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Example data
data = {
    "en": [
        'We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.',
        "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.",
        "Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.",
        "On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.",
        'Danius said, "Right now we are doing nothing. I have called and sent emails to his closest collaborator and received very friendly replies. For now, that is certainly enough."',
        "Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in his garage."
    ] * 100  # Replicate to simulate a large dataset
}

# Load your dataset
dataset = Dataset.from_dict(data)

# Function to find optimal batch size
def find_optimal_batch_size(model, tokenizer, dataset, src_lang='eng_Latn', tgt_lang='spa_Latn'):
    batch_size = 16
    max_batch_size = 16
    while True:
        try:
            print(f"Testing batch size: {batch_size}")
            def batch_translate(batch):
                return {"translations": translate_batch(batch["en"], model, tokenizer, src_lang, tgt_lang)}
            translated_dataset = dataset.map(batch_translate, batched=True, batch_size=batch_size)
            print(len(translated_dataset))
            print(translated_dataset.keys())
            print(len(translated_dataset["translations"]))
            max_batch_size = batch_size
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                break
            else:
                raise e
    return max_batch_size

optimal_batch_size = find_optimal_batch_size(model, tokenizer, dataset)
print(f"Optimal batch size: {optimal_batch_size}")