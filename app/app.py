import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pprint import pprint
from time import sleep
from urllib.parse import urlparse, urljoin

import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import nltk
import requests
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.schema import Document
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from playwright.sync_api import sync_playwright
from pysitemap import crawler
from pysitemap.parsers.lxml_parser import Parser
from seoanalyzer import analyze
from seoanalyzer.website import Website
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words

from dotenv import load_dotenv

load_dotenv()
OPEN_AI_KEY = os.getenv('API_KEY')

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'

def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


def get_filename(url, suffix=''):
    domain = urlparse(url).netloc
    filename = f"{domain.replace('.', '_')}_{suffix}.json"
    return filename


def generate_sitemap(root_url):
    try:
        domain = urlparse(root_url).netloc
        filename = f"{domain.replace('.', '_')}.xml"
        c = crawler(
            root_url, out_file=f"sitemaps/{filename}", exclude_urls=[".pdf", ".jpg", ".zip"],
            http_request_options={"ssl": False}, parser=Parser
        )
        return os.path.isfile(f'sitemaps/{filename}')
    except Exception as e:
        print(e)
        pass


def get_urls_from_sitemap(filename):
    with open(filename, 'r') as file:
        xml_string = file.read()

    # Parse the XML string with BeautifulSoup
    soup = BeautifulSoup(xml_string, 'xml')

    # Find all <url> elements and extract the text inside <loc> element
    urls = [url_element.find('loc').text for url_element in soup.find_all('url')]
    return urls


def scrape_urls(url):
    print(f'started scrapping urls from: {url}')
    filename = get_filename(url, suffix='urls')
    try:
        domain = urlparse(url).netloc
        sitemap_filename = f"{domain.replace('.', '_')}.xml"
        sitemap = f'sitemaps/{sitemap_filename}'
        if not os.path.exists(filename) and len(get_urls_from_sitemap(sitemap)) < 2:
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

                # Close the browser
                browser.close()
                filename = get_filename(url=url, suffix='urls')
                if len(urls) > 0:
                    with open(filename, 'w+') as f:
                        f.write(json.dumps(urls, ensure_ascii=False))
                    print(f'Scraped {len(urls)} URLs')
                    return urls
        else:
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(e)


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
    filename = f"page_data/{get_filename(website_url, suffix='page_data')}"
    try:
        if not os.path.exists(filename):
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
    except Exception as e:
        print(e)


def summarize_page_content(website_url):
    # use langchain and open ai to summarize the content on a page.
    filename = f"page_data/{get_filename(website_url, suffix='page_data')}"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            print(data)
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(summarize_page_callback, item.get('url'), item.get('content')) for item in
                           data]

                results = [future.result() for future in futures]
                summary_filename = f"page_summary/{get_filename(website_url, suffix='page_summary')}"
                with open(summary_filename, 'w+') as page_summary_file:
                    page_summary_file.write(json.dumps(results, ensure_ascii=False))

                    return results


def summarize_page_callback(page_url: str, page_content: str):
    try:
        llm = ChatOpenAI(openai_api_key=OPEN_AI_KEY, temperature=0, model_name="gpt-3.5-turbo-16k")
        chain = load_summarize_chain(llm, chain_type="stuff")
        text = ' '.join(page_content.split()[:12000])
        docs = [Document(page_content=text, metadata={
            'source': page_url
        })]

        result = chain.run(docs)

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
    try:
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
    except Exception as e:
        print(e)
        return {
            'page_url': page_url,
            'meta_description': 'Not Found.'
        }


def save_page_meta_descriptions(website_url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f"page_summary/{get_filename(website_url, suffix='page_summary')}"
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path) and not os.path.exists(
            f"meta_descriptions/{get_filename(website_url, 'meta_descriptions')}"):
        meta_descriptions = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                meta_desc = {
                    'page_url': item.get('page_url', ''),
                    'meta_description': generate_page_meta_description(item.get('page_url', ''),
                                                                       item.get('summary', '')).get('meta_description',
                                                                                                    '')
                }
                print(meta_desc)
                meta_descriptions.append(meta_desc)
            meta_filename = f"meta_descriptions/{get_filename(website_url, suffix='meta_descriptions')}"
            with open(meta_filename, 'w+') as f:
                f.write(json.dumps(meta_descriptions, ensure_ascii=False))


def get_meta_descriptions(website_url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f"meta_descriptions/{get_filename(website_url, suffix='meta_descriptions')}"
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
    try:
        result = analyze(url=url)
        filename = get_filename(url=url, suffix='site_analysis')
        with open(f"site_analysis/{filename}", 'w+') as f:
            f.write(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(e)


# Image TO Text

HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}


def image_to_text(img_url):
    try:
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        image_bytes = BytesIO()
        raw_image.save(image_bytes, format="JPEG")
        image_data = image_bytes.getvalue()
        response = requests.post(API_URL, headers=headers, data=image_data)

        if response.status_code == 200:
            data = response.json()
            return data[0]['generated_text']
        else:
            return ''

    except Exception as e:
        print(e)


def get_images(page_url):
    # Fetch the HTML content of the page
    response = requests.get(page_url, headers={
        'User-Agent': f'{USER_AGENT}'
    })
    response.raise_for_status()  # Raise an error for bad requests

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get the page title
    page_title = soup.title.string.strip() if soup.title else "No title found"
    meta_description_tag = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description_tag['content'].strip() if meta_description_tag else "No meta description found"

    # Find all image tags
    img_tags = soup.find_all('img')

    # Extract image URLs and alt descriptions
    valid_extensions = {'.png', '.jpg', '.jpeg'}

    images_info = []
    for img in img_tags:
        if 'src' in img.attrs:
            image_url = urljoin(page_url, img['src'])
            if any(image_url.lower().endswith(ext) for ext in valid_extensions):
                alt_text = img.get('alt', 'No alt text')
                images_info.append({'url': image_url, 'alt': alt_text})

    # Return the information as a dictionary
    return {'page_title': page_title, 'meta_description': meta_description, 'images_info': images_info}


def generate_image_alt(image_url, current_alt, page_summary):
    try:
        if current_alt != '':
            image_text = current_alt
        else:
            image_text = image_to_text(image_url)
        llm = ChatOpenAI(
            openai_api_key=OPEN_AI_KEY,
            temperature=0.7,
            max_tokens=80
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    template="""You are an intelligent Chat bot that generates an SEO image description
                           for an image on a web page. Use  the provided and image description page summary to generate 
                           the image description should not be more that 155 characters 
                           """,
                    input_variables=[]
                ),
                HumanMessagePromptTemplate.from_template(
                    template="""
                           Generate a short Search engine  optimised (seo) image description using the following page info 
                            SUMMARY: {summary}
                            IMAGE DESCRIPTION: {image_description}

                           SEO optimized image description: 
                           """,
                    input_variables=['summary', 'image_description']
                )
            ]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run({'summary': page_summary, 'image_description': image_text})
        return result
    except Exception as e:
        print(e)
        return None


def save_image_data(website_url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f"page_summary/{get_filename(website_url, suffix='page_summary')}"
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
        page_image_data = []
        for item in data:
            page_url = item['page_url']
            print(f'processing {page_url}')
            summary = item['summary']
            image_data = get_images(page_url)
            images_list = image_data['images_info'][:5]
            ai_image_data = []
            for image in images_list:
                ai_image_item = {
                    'original_alt': image['alt'],
                    'image_url': image['url'],
                    'generated_text': generate_image_alt(
                        image_url=image['url'],
                        current_alt=image['alt'],
                        page_summary=summary
                    )
                }

                ai_image_data.append(ai_image_item)

            page_image_data.append({
                'page_url': page_url,
                'image_data': ai_image_data
            })

        image_data_filename = f"image_data/{get_filename(website_url, suffix='image_data')}"

        with open(image_data_filename, 'w+') as f:
            f.write(json.dumps(page_image_data))


def get_seo_stats(url):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f"site_analysis/{get_filename(url, suffix='site_analysis')}"
    file_path = os.path.join(base_dir, filename)
    keywords = []
    bigrams = []
    trigrams = []
    warnings = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for page_data in data.get('pages', []):

                keyword_records = page_data.get('keywords', [])
                for record in keyword_records:
                    item = {
                        'keyword': record[1],
                        'count': record[0]
                    }
                    keywords.append(item)

                bigram_records = page_data.get('bigrams', [])
                for k, v in bigram_records.items():
                    item = {
                        'bigram': k,
                        'count': v
                    }
                    bigrams.append(item)

                trigram_records = page_data.get('trigrams', [])
                for k, v in trigram_records.items():
                    item = {
                        'trigram': k,
                        'count': v
                    }
                    trigrams.append(item)

                warning_records = list(set(page_data.get('warnings', [])))
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
                            'message': record,
                            'value': record
                        }
                    warnings.append(item)

    else:
        site_analysis(url)
        return {
            'warnings': [],
            'bigrams': [],
            'trigrams': [],
            'keywords': []
        }

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


def get_image_data(website):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f"image_data/{get_filename(website, suffix='image_data')}"
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    return data


def app():
    with st.container():
        st.title = 'Macaws SEO Tool'
        website = st.text_input(placeholder='Enter your website eg. http://example.com', label='Website')
        submit_btn = st.button(label='Submit')
        if submit_btn:
            with st.spinner('Loading website data. This might take a while'):

                meta_desc = get_meta_descriptions(website)
                stats = get_seo_stats(website)
                image_data = get_image_data(website)

                with st.container():
                    st.warning('SEO Warnings')
                    if stats is not None:
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
                with st.container():
                    st.header('Website Images')
                    for item in image_data:
                        if len(item["image_data"]) > 0:
                            st.subheader(f'Images for : {item["page_url"]}')
                            container = st.container()

                            for image in item['image_data']:

                                container.image(image['image_url'], width=280)
                                container.info(f'Old Image Description: {image["original_alt"]}')
                                container.success(f'AI generated Image Description: {image["generated_text"]}')
                        else:
                            st.info(f'No images found at:{item["page_url"]}')


def save_urls(sitemap, url_filename):
    urls = get_urls_from_sitemap(sitemap)
    with open(url_filename, 'w+') as f:
        f.write(json.dumps(urls))


if __name__ == '__main__':
    source_urls = [
        'https://macaws.ai',
        'https://nobacz.com/',
        'https://www.climbingturn.co.uk/',
        'https://ecosytravel.co.uk/',
        'https://gxigroup.com/',


    ]
    app()


