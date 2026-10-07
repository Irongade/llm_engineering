from huggingface_hub import login
from transformers import AutoTokenizer

from dotenv import load_dotenv

load_dotenv(override=True)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)

text = "I am excited to show Tokenizers in action to my LLM engineers"
tokens = tokenizer.encode(text)
print(tokens)

character_count = len(text)
word_count = len(text.split(' '))
token_count = len(tokens)
print(f"There are {character_count} characters, {word_count} words and {token_count} tokens")

print(tokenizer.decode(tokens))

print(tokenizer.batch_decode(tokens))

print(tokenizer.get_added_vocab())

print(len(tokenizer.vocab))


# Instruct Variants

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)


messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]
