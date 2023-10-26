import csv
import json
import os

from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from urllib.parse import urlparse

import nltk
import requests
from bs4 import SoupStrainer, BeautifulSoup
from langchain import hub
from langchain.chains import LLMChain, StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.schema import Document, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words

import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

        filename = get_filename(url=website_url, suffix='urls')
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


def generate_meta_tag_from_page_content(urls: list):
    # print('Downloading nltk models and other resources')
    # nltk.download('popular')
    # print('Nltk resources downloaded')

    results = []
    # 2) analyzes the content on each URL
    for url in urls:
        parser = PlaintextParser.from_string(get_page_content(url), Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")
        description = summarizer(parser.document, 3)
        description = " ".join([sentence._text for sentence in description])
        if len(description) > 155:
            description = description[:152] + '...'
        results.append({
            'url': url,
            'description': description
        })

    pprint(results)

    # 4) exports the results to a csv file
    with open('meta_descriptions.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'description'])
        writer.writeheader()
        writer.writerows(results)


def save_pages(website_url, urls):
    pages_data = []
    for url in urls:
        item = {
            'url': url,
            'content': get_page_content(url)
        }
        print(f'processed {item}')
        pages_data.append(item)
    filename = get_filename(website_url, suffix='page_data')
    with open(filename, 'w+') as f:
        f.write(json.dumps(pages_data, ensure_ascii=False))


def summarize_page_content():
    # use langchain and open ai to summarize the content on a page.
    with open('page_data.json', 'r') as f:
        data = json.load(f)
        print(data)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(summarize_page_callback, item.get('url'), item.get('content')) for item in
                       data[:1]]

            results = [future.result() for future in futures]
            with open('page_summary.json', 'w+') as page_summary_file:
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


def clean_text(raw_text):
    # Remove HTML tags
    text_no_html = BeautifulSoup(raw_text, 'html.parser').get_text()

    # Convert to lowercase
    lowercased_text = text_no_html.lower()

    # Remove non-alphanumeric characters
    alphanumeric_text = re.sub(r'[^a-zA-Z0-9\s]', '', lowercased_text)

    return alphanumeric_text


from nltk.util import bigrams, trigrams


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


def save_meta_descriptions(website_url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = 'page_summary.json'
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                data_item = {
                    'page_url':item.get('url'),
                    'description':g
                }





if __name__ == '__main__':
    website_url = 'https://macaws.ai'
    # urls = scrape_urls(website_url)
    # save_pages(website_url, urls)
    ngrams = get_website_n_grams(website_url)
    pprint(ngrams)
    # summarize_page_content()
    # summary = 'Hudutech Ventures offers industry compliant solutions for enterprise business applications. They provide customized software solutions, including ERP implementation, mobile app development, website and web app development, software design and development, turnkey solutions, training, and more. They work closely with clients to deliver custom solutions that fit their unique needs and provide support and maintenance. Hudutech Ventures also offers digital marketing services to reach a wider audience and increase brand awareness. They have a team of experienced developers and offer ready-to-use software solutions. Clients have praised their technical expertise and ability to customize ERP systems.'
    # generate_page_meta_description(page_url='', summary=summary)
