import json
import os
import re
from pprint import pprint
from urllib.parse import urlparse

import nltk
from seoanalyzer import analyze
import pandas as pd


def get_filename(url, suffix=''):
    domain = urlparse(url).netloc
    filename = f"{domain.replace('.', '_')}_{suffix}.json"
    return filename


def site_analysis(url):
    print(f'Analysing {url}')
    result = analyze(url)
    filename = get_filename(url=url, suffix='site_analysis')
    with open(filename, 'w+') as f:
        f.write(json.dumps(result, ensure_ascii=False))


def get_seo_stats(url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = get_filename(url, suffix='site_analysis')
    file_path = os.path.join(base_dir, filename)
    keywords = []
    bigrams = []
    trigrams = []
    warnings = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            page_data = data.get('pages')[0]
            keyword_records = page_data.get('keywords')
            for record in keyword_records:
                item = {
                    'keyword': record[1],
                    'count': record[0]
                }
                keywords.append(item)

            bigram_records = page_data.get('bigrams')
            for k, v in bigram_records.items():
                item = {
                    'bigram': k,
                    'count': v
                }
                bigrams.append(item)

            trigram_records = page_data.get('trigrams')
            for k, v in trigram_records.items():
                item = {
                    'trigram': k,
                    'count': v
                }
                trigrams.append(item)

            warning_records = list(set(page_data.get('warnings')))
            print("Error Warns", len(warning_records), len(page_data.get('warnings')))
            for record in warning_records:
                record_list = record.split(':')
                item = {}
                warning_message = record_list[0]
                if warning_message.startswith('Anchor missing title tag'):
                    pattern = r'Anchor missing title tag: (.+)'

                    match = re.search(pattern, record)
                    url = match.group(1) if match else None

                    item = {
                        'message': warning_message,
                        'value': url,

                    }
                elif warning_records == 'Image missing alt tag':
                    # todo implement image to text
                    pattern = r'src="([^"]+)"'
                    match = re.search(pattern, record)

                    # If a match is found, extract the image URL
                    image_url = match.group(1) if match else None
                    item = {
                        'message': warning_message,
                        'value': image_url
                    }
                else:
                    item = {
                        'message': warning_message,
                        'value': record_list[1].strip()
                    }
                warnings.append(item)

    else:
        site_analysis(url)
        return

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    bi_finder = nltk.BigramCollocationFinder.from_words(
        [item['bigram'] for item in bigrams]
    )
    tri_finder = nltk.TrigramCollocationFinder.from_words(
        [item['trigram'] for item in trigrams]
    )

    stats = {
        'warnings': warnings,
        'keywords': keywords,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'top_10_bigrams': bi_finder.nbest(bigram_measures.pmi, 10),
        'top_10_trigrams': tri_finder.nbest(trigram_measures.pmi, 10),

    }
    return stats


if __name__ == '__main__':
    website_url = 'https://macaws.ai'
    stats = get_seo_stats(url=website_url)
    pprint(stats)