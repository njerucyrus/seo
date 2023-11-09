import json
import os
from io import BytesIO
from pprint import pprint
from urllib.parse import urljoin
import requests
from PIL import Image
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

OPEN_AI_KEY = os.getenv('API_KEY')

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
    response = requests.get(page_url)
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


def generate_image_alt(image_url, current_alt,  page_summary):
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


def process_image_data():
    page_summary = "Macaws.ai is an AI-driven sales and marketing platform that aims to minimize complexity and maximize success for businesses. It offers automation tools to streamline operations, reduce errors, and increase efficiency. By leveraging AI technologies, businesses can analyze data to gain insights into customer behavior and preferences, enhance sales forecasting accuracy, and improve customer support. The platform also provides personalized marketing, predictive analytics, content creation, lead generation, social media management, ad optimization, competitor analysis, email marketing automation, SEO and keyword research services. Macaws.ai aims to empower small businesses to compete with larger enterprises by providing them with valuable resources and tools."

    generated_data = []
    with open('image_data.json', 'r') as f:
        data = json.load(f)
        for item in data[:4]:
            print(f'processing: {item["url"]}')
            alt_desc = generate_image_alt(image_text=item['generated_text'], page_summary=page_summary)
            generated_data.append({
                'alt_desc': alt_desc,
                'source_image_text': item['generated_text'],
                'image_url': item['url']
            })

    with open('ai_image_alts.json', 'w+') as f:
        f.write(json.dumps(generated_data))


if __name__ == '__main__':
    pass
