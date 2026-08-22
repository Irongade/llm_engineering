import os
from typing import TypedDict
from dotenv import load_dotenv
from model import AiModel, ChatOptions
import gradio as gr

load_dotenv(override=True)
open_api_key = os.getenv("OPENAI_API_KEY")

if open_api_key:
    print("OpenAI API key exists")
else:
    print("OpenAi API key not set")

model = AiModel(model="gpt-4.1-mini")

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."

class HistoryItem(TypedDict):
    role: str
    content: str

def chat(message: str, history):
    prev_history: list[HistoryItem] = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + prev_history + [{"role": "user", "content": message}]
    
    options = ChatOptions(stream=True, response_format=None, tools=None)
    reply = ""
    for response in model.chat_stream(messages=messages, options=options):
        reply += response
        yield reply

gr.ChatInterface(fn=chat, type="messages").launch()