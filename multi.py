
import re
from collections import defaultdict

def is_clean_line(line, min_word_count = 4):
    # Check if the line starts with "http:" or "Navigation path"
    if line.startswith("http:") or line.startswith("Navigation Path :") or line.startswith("Languages :") or "ppt (" in line:
        return False
    
    # Check for more non-alpha tokens than alpha-tokens
    alpha_count = sum(1 for char in line if char.isalpha())
    # Count non-alpha characters excluding whitespaces
    non_alpha_count = sum(1 for char in line if not char.isalpha() and not char.isspace())
    if non_alpha_count > alpha_count:
        return False
    
    # Check for minimum number of words
    words = [word for word in line.split() if word.isalpha()]
    if len(words) < min_word_count:
        return False
    
    return True

def load_parallel_data(filename_en, filename_other):
    with open(filename_en, 'r', encoding='utf-8') as file_en:
        english_sentences = file_en.readlines()
    with open(filename_other, 'r', encoding='utf-8') as file_other:
        other_sentences = file_other.readlines()
    
    # Ensure they are parallel
    assert len(english_sentences) == len(other_sentences)
    # Clean data by removing pairs where the source and target are identical
    #cleaned_data = [(eng.strip(), other.strip()) for eng, other in zip(english_sentences, other_sentences) if eng.strip() != other.strip()]
    # Clean data by removing pairs where the source and target are identical or don't meet cleaning rules
    cleaned_data = [
        (eng.strip(), other.strip())
        for eng, other in zip(english_sentences, other_sentences)
        if eng.strip() != other.strip() and is_clean_line(eng) and is_clean_line(other)
    ]
    return cleaned_data
    #return list(zip(english_sentences, other_sentences))


def build_multi_parallel_corpus(lang_pairs):
    corpus = defaultdict(lambda: defaultdict(str))
    
    for lang_pair in lang_pairs:
        data = load_parallel_data(f'data.en-{lang_pair}.en', f'data.en-{lang_pair}.{lang_pair}')
        for eng, other in data:
            corpus[eng.strip()][lang_pair] = other.strip()
    
    return corpus

def filter_full_translations(corpus, languages):
    filtered_corpus = {}
    for eng, translations in corpus.items():
        if all(lang in translations for lang in languages):
            filtered_corpus[eng] = translations
    return filtered_corpus


def analyze_impact_on_corpus_size(lang_pairs):
    languages = []
    sizes = []
    # we create a dict corpus with inner dicts for each language's translations 
    corpus = defaultdict(lambda: defaultdict(str))
    
    for lang_pair in lang_pairs:
        languages.append(lang_pair)
        data = load_parallel_data(f'data.en-{lang_pair}.en', f'data.en-{lang_pair}.{lang_pair}')
        for eng, other in data:
            corpus[eng.strip()][lang_pair] = other.strip()
        
        filtered_corpus = filter_full_translations(corpus, languages)
        sizes.append(len(filtered_corpus))
        print(f'After adding {lang_pair}: {len(filtered_corpus)} sentences remain')
    
    return sizes


def sort_langs_by_size(languages):
    lang_and_size = []
    for lang in languages:
        filename_en = f"data_raw/en-{lang}.txt/ECB.en-{lang}.en"
        with open(filename_en, 'r', encoding='utf-8') as file_en:
            english_sentences = file_en.readlines()
            lang_and_size.append((lang, len(english_sentences)))
    lang_and_size = sorted(lang_and_size, key=lambda x: x[1], reverse=True)
    sorted_langs = [lang for lang, _ in lang_and_size]
    return sorted_langs

def save_corpus(filtered_corpus, languages):
    # Open a file for each language, including English
    files = {lang: open(f"data_multi/multi.ECB.{lang}.txt", "w", encoding='utf-8') for lang in languages + ['en']}

    # Write each translation to the appropriate file
    for eng_sentence, translations in filtered_corpus.items():
        # Write the English sentence to its file
        files['en'].write(eng_sentence + '\n')
        # Write each translated sentence to its respective language file
        for lang in languages:
            files[lang].write(translations[lang] + '\n')

    # Close all files
    for file in files.values():
        file.close()


if __name__ == "__main__":
    # sort langs list by size (desc)
    languages = ["it", "fi", "es", "el", "lt", "cs", "da", "de", "et", "fr", "hu", "lv", "mt", "nl", "pl", "pt", "sk", "sl"]
    languages = sort_langs_by_size(languages)
    print(languages)

    langs_visited = []  
    sizes = []
    corpus = defaultdict(lambda: defaultdict(str))

    for lang in languages:
        langs_visited.append(lang)
        prefix = f"data_raw/en-{lang}.txt/ECB.en-{lang}"
        file_en = f"{prefix}.en"
        file_lang = f"{prefix}.{lang}"
        # clean each parallel file
        # non translated data
        # non alpha, single words
        lang_corpus = load_parallel_data(file_en, file_lang)
        
        for eng, other in lang_corpus:
            corpus[eng.strip()][lang] = other.strip()
        print(lang, len(lang_corpus), len(corpus))
        filtered_corpus = filter_full_translations(corpus, langs_visited)
        sizes.append(len(filtered_corpus))
        print(f'After adding {lang}: {len(filtered_corpus)} sentences remain')
    
    save_corpus(filtered_corpus, languages)
    