from utils import languages

# loading datasets
def load_dataset(
        filename_prefix : str ="./data_multi/multi.ECB",
        verbose : bool =True,
        sample: int = 0) -> dict :
    
    dataset = {}
    for lang, _ in languages + [("en", "eng_Latn")]:
        filename = f'{filename_prefix}.{lang}.txt'
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        dataset[f"{lang}"] = [l.strip() for l in lines][-sample:]
    if verbose:
        print(lang, len(dataset[lang]))

    return dataset