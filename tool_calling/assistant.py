import json
import os
from typing import TypedDict
from dotenv import load_dotenv
from model import AiModel, ChatOptions
from model.constants import TOOL_CALLS
import gradio as gr
import sqlite3

load_dotenv(override=True)
open_api_key = os.getenv("OPENAI_API_KEY")

if open_api_key:
    print("OpenAI API key exists")
else:
    print("OpenAi API key not set")

DB = "prices.db"

with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)')
    conn.commit()

model = AiModel(model="gpt-4.1-mini")

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so. And always use the database version of the tool calls.
"""

# TOOLS

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
ticket_prices_for_db = {"london": "$899", "paris": "$999", "tokyo": "$2400", "berlin": "$599"}

def set_ticket_price(destination_city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?', (destination_city.lower(), price, price))

def get_ticket_price(destination_city):
    print(f"Tool called for city {destination_city}")
    price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"

def get_ticket_price_from_db(destination_city):
    print(f"DATABASE TOOL CALLED: Getting price {destination_city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price from prices WHERE city = ?', (destination_city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {destination_city} is {result[0]}" if result else "No price data available for this city"
    
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

price_function_db = {
    "name": "get_ticket_price_from_db",
    "description": "Get the price of a return ticket to the destination city Using the Database.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}, {"type": "function", "function": price_function_db}]


def handle_tool_call(tool_call):
    if tool_call.function.name == "get_ticket_price":
        args = json.loads(tool_call.function.arguments)
        city = args.get('destination_city')
        price_details = get_ticket_price(city)

        response = {
            "role": "tool",
            "content": price_details,
            "tool_call_id": tool_call.id
        }
    elif tool_call.function.name == "get_ticket_price_from_db":
        args = json.loads(tool_call.function.arguments)
        city = args.get('destination_city')
        price_details = get_ticket_price_from_db(city)

        response = {
            "role": "tool",
            "content": price_details,
            "tool_call_id": tool_call.id
        }

    return response

def handle_tool_calls(message):
    responses = []

    for tool_call in message.tool_calls:
        response = handle_tool_call(tool_call)
        responses.append(response)

    return responses

# update the DB with the values.
for city, price in ticket_prices_for_db.items():
    set_ticket_price(city, price)

# END OF TOOLS

class HistoryItem(TypedDict):
    role: str
    content: str

def chat(message: str, history):
    prev_history: list[HistoryItem] = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + prev_history + [{"role": "user", "content": message}]
    
    options = ChatOptions(stream=False, response_format=None, tools=tools)
    response = model.generic_chat(messages=messages, options=options)
    print(response.choices[0])

    if response.choices[0].finish_reason == TOOL_CALLS:
        message = response.choices[0].message
        response = handle_tool_calls(message)
        # add details about tool call then add responses from tool call done locally
        messages.append(message)
        messages = messages + response

        for message in messages:
            print(message)
    
        response = model.chat(messages=messages, options=options)
    else:
        response = response.choices[0].message.content or ""

    return response


gr.ChatInterface(fn=chat, type="messages").launch()