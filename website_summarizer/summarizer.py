import os
from dotenv import load_dotenv
from scrapper import WebScrapper
from selenium_scrapper import SeleniumWebScrapper
from IPython.display import Markdown, display
from model import Model, Provider

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

url = "https://edwarddonner.com"

if not api_key:
    print("No API Key found")
elif not api_key.startswith("sk-proj-"):
    print("API key found, but it doesnt start with 'sk-proj-'; Please check you are using the right key")
elif api_key.strip() != api_key:
    print("An API key was found, but it might have spaces and indentations at the start or end. Kindly remove them")
else:
    print("API key found! Looks good!" + "\n")


message = "Hello, GPT! this is my first message ever to you! Hi"

messages  = [{"role": "user", "content": message}]

print(messages + "\n")

openai = Model(model="gpt-5-nano")
response = openai.chat(messages=messages)
print(response)

ws = WebScrapper(url)

website = ws.fetch_website_contents()

system_prompt = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""

def messages_for(web_content):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + web_content}
    ]

def summarize(url):
    ws = WebScrapper(url)
    web_content = ws.fetch_website_contents()
    openai_model = Model(model="gpt-4.1-mini")
    response = openai_model.chat(messages= messages_for(web_content))
    return response

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

display_summary(url)


def selenium_summarize(url, wait):
    ws = SeleniumWebScrapper(url, wait)
    web_content = ws.fetch_website_contents()
    openai_model = Model(model="gpt-4.1-mini")
    response = openai_model.chat(messages= messages_for(web_content))
    return response


def display_selenium_summary(url):
    summary = selenium_summarize(url, 10)
    display(Markdown(summary))

display_selenium_summary("https://openai.com")

# FOR OPEN SOURCED MODELS

gemini_model = Model(provider=Provider.GEMINI)
print(gemini_model.chat(messages=[{"role": "user", "content": "Tell me a fun fact"}]))

llama_model = Model(provider=Provider.OLLAMA)
print(llama_model.chat(messages=[{"role": "user", "content": "Tell me a fun fact"}]))

