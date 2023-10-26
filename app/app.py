import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import nltk
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.schema import Document
from playwright.sync_api import sync_playwright
from seoanalyzer import analyze
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words

OPEN_AI_KEY = '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'


def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


def get_filename(url, suffix=''):
    domain = urlparse(url).netloc
    filename = f"{domain.replace('.', '_')}_{suffix}.json"
    return filename


def scrape_urls(url):
    print(f'started scrapping urls from: {url}')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Navigate to the website
        page.goto(url)

        # Extract all unique non-empty, non-mailto, non-tel, and same-domain URLs on the page
        urls = page.evaluate(f'''(mainUrl) => {{
            const anchors = document.querySelectorAll('a');
            const uniqueUrls = new Set();
            const mainDomain = new URL(mainUrl).hostname;

            anchors.forEach(anchor => {{
                const href = anchor.href.trim();
                if (
                    href !== ''
                    && !href.startsWith('mailto:')
                    && !href.startsWith('tel:')
                    && new URL(href).hostname === mainDomain
                ) {{
                    uniqueUrls.add(href);
                }}
            }});

            return Array.from(uniqueUrls);
        }}''', url)

        filename = get_filename(url=url, suffix='urls')
        with open(filename, 'w+') as f:
            f.write(json.dumps(urls, ensure_ascii=False))
        # Close the browser
        browser.close()
        print(f'Scraped {len(urls)} URLs')
        return urls


def get_redirects(url):
    redirect_urls = []
    try:
        response = requests.head(url, allow_redirects=True)
        print(f"Redirects for {url}:")

        for redirect in response.history:
            print(f" - {redirect.url} ({redirect.status_code})")
            redirect_urls.append(redirect.url)

    except requests.RequestException as e:
        print(f"Error: {e}")
    return {
        'url': url,
        'redirects': redirect_urls
    }


def get_page_content(url):
    print(f'getting content for {url}')
    loader = AsyncChromiumLoader([url])
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        html,
        tags_to_extract=["p", "span", 'li', 'div'],
        unwanted_tags=["style", "script", "a"]
    )
    if len(docs_transformed) > 0:
        print(f'found: {docs_transformed[0].page_content}')
        return docs_transformed[0].page_content

    else:
        print('Did not find any content on the above page')
        return ''


def save_pages(website_url, urls):
    filename = get_filename(website_url, suffix='page_data')

    base_dir = os.path.dirname(os.path.realpath(__file__))

    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        pages_data = []
        for url in urls:
            item = {
                'url': url,
                'content': get_page_content(url)
            }
            print(f'processed {item}')
            pages_data.append(item)
        with open(filename, 'w+') as f:
            f.write(json.dumps(pages_data, ensure_ascii=False))


def summarize_page_content(website_url):
    # use langchain and open ai to summarize the content on a page.
    filename = get_filename(website_url, suffix='page_data')
    if not os.path.exists(filename) and not os.path.exists(get_filename(website_url, suffix='page_summary')):
        with open(filename, 'r') as f:
            data = json.load(f)
            print(data)
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(summarize_page_callback, item.get('url'), item.get('content')) for item in
                           data]

                results = [future.result() for future in futures]
                summary_filename = get_filename(website_url, suffix='page_summary')
                with open(summary_filename, 'w+') as page_summary_file:
                    page_summary_file.write(json.dumps(results, ensure_ascii=False))

                    return results


def summarize_page_callback(page_url: str, page_content: str):
    try:
        llm = ChatOpenAI(openai_api_key=OPEN_AI_KEY, temperature=0, model_name="gpt-3.5-turbo-16k")
        chain = load_summarize_chain(llm, chain_type="stuff")

        docs = [Document(page_content=page_content, metadata={
            'source': page_url
        })]

        result = chain.run(docs)

        #     llm = ChatOpenAI(
        #         openai_api_key=OPEN_AI_KEY,
        #         temperature=0.0,
        #
        #     )
        #     print(f'summarising content from: {page_url}')
        #
        #     # Map
        #     map_template = """The following is a set of documents
        #     {docs}
        #     Based on this list of docs, please identify the main themes
        #     Helpful Answer:"""
        #     map_prompt = PromptTemplate.from_template(map_template)
        #     map_chain = LLMChain(llm=llm, prompt=map_prompt)
        #
        #     reduce_template = """The following is set of summaries:
        #     {docs}
        #     Take these and distill it into a final, consolidated summary in form of paragraphs.
        #     Helpful Answer:"""
        #
        #     reduce_prompt = PromptTemplate.from_template(reduce_template)
        #     # reduce_prompt = hub.pull('rlm/map-prompt')
        #     reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
        #
        #     # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        #     combine_documents_chain = StuffDocumentsChain(
        #         llm_chain=reduce_chain, document_variable_name="docs"
        #     )
        #
        #     # Combines and iteravely reduces the mapped documents
        #     reduce_documents_chain = ReduceDocumentsChain(
        #         # This is final chain that is called.
        #         combine_documents_chain=combine_documents_chain,
        #         # If documents exceed context for `StuffDocumentsChain`
        #         collapse_documents_chain=combine_documents_chain,
        #         # The maximum number of tokens to group documents into.
        #         token_max=2000,
        #     )
        #
        #     # Combining documents by mapping a chain over them, then combining results
        #     map_reduce_chain = MapReduceDocumentsChain(
        #         # Map chain
        #         llm_chain=map_chain,
        #         # Reduce chain
        #         reduce_documents_chain=reduce_documents_chain,
        #         # The variable name in the llm_chain to put the documents in
        #         document_variable_name="docs",
        #         # Return the results of the map steps in the output
        #         return_intermediate_steps=False,
        #     )
        #
        #     # create docs from text
        #     docs = [Document(page_content=page_content, metadata={
        #         'source': page_url
        #     })]
        #
        #     text_splitter = RecursiveCharacterTextSplitter(
        #         chunk_size=1000,
        #         chunk_overlap=0,
        #         length_function=len
        #     )
        #
        #     document_chunks = text_splitter.split_documents(docs)
        #
        #     result = map_reduce_chain.run(document_chunks)
        if result:
            item = {
                'page_url': page_url,
                'summary': result
            }
            print(f'SUMMARY: {item}')
            return item

        else:
            item = {
                'page_url': page_url,
                'summary': 'Not Found'
            }
            print(f'SUMMARY: {item}')
            return item
    except Exception as e:
        print(e)
        return None


def generate_page_meta_description(page_url: str, summary: str):
    llm = ChatOpenAI(
        openai_api_key=OPEN_AI_KEY,
        temperature=0.7,
        max_tokens=80
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                template="""You are an intelligent Chat bot that generates an SEO meta description
                for a web page. Use  the given page summary to generate the meta description which 
                should not be more that 155 characters 
                """,
                input_variables=[]
            ),
            HumanMessagePromptTemplate.from_template(
                template="""
                Generate a short Search engine  optimised (seo) meta description using the following page info 
                 SUMMARY: {summary}

                SEO optimized meta description: 
                """,
                input_variables=['summary']
            )
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({'summary': summary})
    return {
        'page_url': page_url,
        'meta_description': result
    }


def save_page_meta_descriptions(website_url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = get_filename(website_url, suffix='page_summary')
    file_path = os.path.join(base_dir, filename)
    if  os.path.exists(file_path) and not os.path.exists(get_filename(website_url, 'meta_descriptions')):
        meta_descriptions = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                meta_descriptions.append({
                    'page_url': item.get('page_url'),
                    'meta_description': generate_page_meta_description(item.get('page_url'), item.get('summary')).get('meta_description')
                })
            meta_filename = get_filename(website_url, suffix='meta_descriptions')
            with open(meta_filename, 'w+') as f:
                f.write(json.dumps(meta_descriptions, ensure_ascii=False))


def get_meta_descriptions(website_url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = get_filename(website_url, suffix='meta_descriptions')
    file_path = os.path.join(base_dir, filename)

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        save_page_meta_descriptions(website_url)
        return {}


def clean_text(raw_text):
    # Remove HTML tags
    text_no_html = BeautifulSoup(raw_text, 'html.parser').get_text()

    # Convert to lowercase
    lowercased_text = text_no_html.lower()

    # Remove non-alphanumeric characters
    alphanumeric_text = re.sub(r'[^a-zA-Z0-9\s]', '', lowercased_text)

    return alphanumeric_text


def get_website_n_grams(website_url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = get_filename(website_url, suffix='page_data')
    file_path = os.path.join(base_dir, filename)
    n_grams = []
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

            for item in data:
                bi_finder = nltk.BigramCollocationFinder.from_words(
                    clean_text(item.get('content')).split())
                tri_finder = nltk.TrigramCollocationFinder.from_words(
                    clean_text(item.get('content')).split())

                item = {
                    'page_url': item.get('url'),
                    # 'bigrams': list(bigrams(clean_text(item.get('content')).split())),
                    'top_10_bigrams': bi_finder.nbest(bigram_measures.pmi, 10),
                    # 'trigrams': trigrams(clean_text(item.get('content')).split()),
                    'top_10_trigrams': tri_finder.nbest(trigram_measures.pmi, 10)

                }
                n_grams.append(item)

        return n_grams

    else:
        save_pages(website_url=website_url, urls=scrape_urls(website_url))
        return


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


def app():
    with st.container():
        st.title = 'Macaws SEO Tool'
        website = st.text_input(placeholder='Enter your website eg. http://example.com', label='Website')
        submit_btn = st.button(label='Submit')
        if submit_btn:
            with st.spinner('Loading website data. This might take a while'):
                urls = scrape_urls(website)
                save_pages(website, urls)
                summarize_page_content(website)
                save_page_meta_descriptions(website)
                meta_desc = get_meta_descriptions(website)
                stats = get_seo_stats(website)
                with st.container():
                    st.warning('SEO Warnings')
                    st.info(f'We found {len(stats.get("warnings"))} SEO warnings')
                    messages = [item['message'] for item in stats.get('warnings')]
                    targets = [item['value'] for item in stats.get('warnings')]
                    table_data = {
                        'Warning Message': messages,
                        'Targets': targets,

                    }
                    st.dataframe(data=table_data)

                with st.container():
                    st.success('Generated Meta Description Based on Page Content')

                    link = [item['page_url'] for item in meta_desc]
                    meta_descriptions = [item['meta_description'] for item in meta_desc]
                    meta_table_data = {
                        'Page Link': link,
                        'MetaDescription': meta_descriptions,

                    }
                    st.dataframe(data=meta_table_data)

                with st.container():
                    st.info('Keyword frequency chart')
                    keywords = [item['keyword'] for item in stats.get('keywords')]
                    frequencies = [item['count'] for item in stats.get('keywords')]
                    data = {
                        'Keywords': keywords,
                        'Frequency': frequencies
                    }
                    df = pd.DataFrame(data)
                    chart = alt.Chart(df).mark_bar().encode(
                        x='Keywords',
                        y='Frequency',
                        color=alt.value('skyblue'),
                    ).properties(
                        width=alt.Step(80)  # Adjust the width of each bar
                    )

                    # Display the plot in Streamlit
                    st.altair_chart(chart, use_container_width=True)




if __name__ == '__main__':
    app()
