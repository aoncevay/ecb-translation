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


def save_sample(
        filename_prefix : str ="./data_multi/multi.ECB",
        outdir_prefix : str = "./data_smpl50/multi.ECB",
        sample: int = 50) -> None:
    
    for lang, _ in languages + [("en", "eng_Latn")]:
        filename = f'{filename_prefix}.{lang}.txt'
        filename_out = f"{outdir_prefix}.{lang}.txt"
        with open(filename, "r", encoding="utf-8") as f, open(filename_out, "w", encoding="utf-8") as f_out:
            lines = f.readlines()
            sample_lines = [l.strip() for l in lines][-sample:]
            f_out.write("\n".join(sample_lines))
    return