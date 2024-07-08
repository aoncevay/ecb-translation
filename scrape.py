import requests
from bs4 import BeautifulSoup
from utils import languages_names
from multi import is_clean_line
import os

not_cleaned_langs = ["mt", "de", "et", "ga", "pt"]

def scrape():
    for lang, _, _ in languages_names + [("en", "", "")]:
        if lang in not_cleaned_langs:
            continue
    #lang="en"
        # URL of the ECB Annual Report 2023
        url = f"https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2023~d033c21ac2.{lang}.html"

        # Send a GET request to the URL
        response = requests.get(url)

        # Parse the content of the request with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')


        # Find the specific section using the full path
        #main_div = soup.find('div', attrs={'id': 'main-wrapper'})  # Assuming main-wrapper is the main container
        main_wrapper = soup.find('div', id='main-wrapper')
        if main_wrapper:
            main = main_wrapper.find('main')
            divs = main.find_all('div', recursive=False)
            if len(divs) > 1:
                section_div = divs[1]
            else:
                section_div = None
        else:
            section_div = None

        #print(main_div)
        # List to store the extracted text
        text_parts = []

        if section_div:
            # Extract headers and paragraphs within the section
            for tag in section_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                text_parts.append(tag.get_text())

            # Extract quotes within the section
            for quote in section_div.find_all('div', class_='quote'):
                text_parts.append(quote.get_text())

        # Join all text parts into a single string
        text_content = '\n'.join([t.strip() for t in text_parts if len(t.strip())>0][:833]) # after this there are footnotes only

        # Print the extracted text
        #print(text_content)
        with open(f"2023report/{lang}.txt", "w") as f:
            f.write(text_content)
            #print(lang, len(text_content))

def load_and_clean():
    raw_prefix = "2023report"
    reference_file = f"{raw_prefix}/en.txt"
    with open(reference_file, 'r', encoding='utf-8') as ref_file:
        lines = ref_file.readlines()
        clean_lines_indices = [i for i, line in enumerate(lines) if is_clean_line(line, min_word_count=10, max_word_count=102)]
        print(len(lines), len(clean_lines_indices))
    #for lang, _, _ in languages_names:
    language_files = [f"{raw_prefix}/{lang}.txt" for lang,_,_ in languages_names + [("en","","")] if lang not in not_cleaned_langs]
    output_dir = "data_2023"
    # Process each language file based on the filtered indices
    for lang_file in language_files:
        with open(lang_file, 'r', encoding='utf-8') as lf:
            lang_lines = lf.readlines()

        # Select lines corresponding to the clean indices
        filtered_lines = [lang_lines[i] for i in clean_lines_indices]

        # Write the filtered lines to a new file
        output_file = os.path.join(output_dir, "ECB." +  os.path.basename(lang_file))
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.writelines(filtered_lines)


if __name__ == "__main__":
    #scrape()
    load_and_clean()