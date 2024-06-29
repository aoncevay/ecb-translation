import sacrebleu
from comet import download_model, load_from_checkpoint
from utils import languages
from read import load_dataset
import sys
import os
import json


def evaluate(results_prefix = "results/nllb-200-distilled-600M"):
    bleu_calc = sacrebleu.BLEU()
    chrf2_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

    # Choose your model from Hugging Face Hub
    # model_path = download_model("Unbabel/XCOMET-XL")
    # or for example:
    model_path = download_model("Unbabel/wmt22-comet-da")
    # Load the model checkpoint:
    comet_model = load_from_checkpoint(model_path)

    # Load data and results from files
    dataset = load_dataset()
    results = {}
    for lang, _ in languages:
        print(lang)
        with open(f"{results_prefix}.en2{lang}.txt", "r") as f:
            results[f'en2{lang}'] = f.readlines()
        with open(f"{results_prefix}.{lang}2en.txt", "r") as f:
            results[f'{lang}2en'] = f.readlines()

    scores = {}

    for lang, _ in languages:
        scores[f'{lang}2en'] = {}
        scores[f'en2{lang}'] = {}
        #print(f"en-2-{lang}")
        en2xx_bleu = bleu_calc.corpus_score(results[f'en2{lang}'], [dataset[f'{lang}']])
        en2xx_chrf = chrf2_calc.corpus_score(results[f'en2{lang}'], [dataset[f'{lang}']])
        #print(f"{lang}-2-en")
        xx2en_bleu = bleu_calc.corpus_score(results[f'{lang}2en'], [dataset[f'en']])
        xx2en_chrf = chrf2_calc.corpus_score(results[f'{lang}2en'], [dataset[f'en']])

        scores[f'en2{lang}']['bleu'] = en2xx_bleu.score
        scores[f'en2{lang}']['chrf'] = en2xx_chrf.score
        scores[f'{lang}2en']['bleu'] = xx2en_bleu.score
        scores[f'{lang}2en']['chrf'] = xx2en_chrf.score

        # COMET data format
        data_en2xx = []
        data_xx2en = []
        for src, mt, ref in zip(dataset['en'], results[f'en2{lang}'], dataset[lang]):
            data_en2xx.append({"src": src, "mt": mt, "ref": ref})
        for src, mt, ref in zip(dataset[lang], results[f'{lang}2en'], dataset['en']):
            data_xx2en.append({"src": src, "mt": mt, "ref": ref})
        # Call predict method:
        comet_en2xx_out = comet_model.predict(data_en2xx, batch_size=8, gpus=1)
        comet_xx2en_out = comet_model.predict(data_xx2en, batch_size=8, gpus=1)
        scores[f'en2{lang}']['comet'] = comet_en2xx_out.system_score
        scores[f'{lang}2en']['comet'] = comet_xx2en_out.system_score
        #print(lang, comet_en2xx_out.system_score, comet_xx2en_out.system_score)

        print(f"{lang},en2{lang},{en2xx_bleu.score},{en2xx_chrf.score},{comet_en2xx_out.system_score},{lang}2en,{xx2en_bleu.score},{xx2en_chrf.score},{comet_xx2en_out.system_score}")

    for lang, _ in languages:
        print(f"{lang},en2{lang},{scores[f'en2{lang}']['bleu']},{scores[f'en2{lang}']['chrf']},{scores[f'en2{lang}']['comet']},{lang}2en,{scores[f'{lang}2en']['bleu']},{scores[f'{lang}2en']['chrf']},{scores[f'{lang}2en']['comet']}")

    os.makedirs("scores", exist_ok=True)
    with open(f"scores/{results_prefix.split('/')[-1].json}", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    
if __name__ == "__main__":
    argv = sys.argv
    model_name = argv[1] if len(argv) > 1 else 'facebook/nllb-200-distilled-600M'
    
    evaluate(f"results/{model_name}")