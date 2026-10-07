
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc

from dotenv import load_dotenv

load_dotenv(override=True)

hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

LLAMA = "meta-llama/Llama-3.2-1B-Instruct"

PHI = "microsoft/Phi-4-mini-instruct"
GEMMA = "google/gemma-3-270m-it"
QWEN = "Qwen/Qwen3-4B-Instruct-2507"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

messages = [
    {"role": "user", "content": "Tell a joke for a room of Data Scientists"}
]

# Quantization Config - this allows us to load the model into memory and use less memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)


# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

print(inputs)

# The model
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)

memory = model.get_memory_footprint() / 1e6
print(f"Memory footprint: {memory:,.1f} MB")

print(model)

outputs = model.generate(inputs, max_new_tokens=80)
print(outputs[0])

print(tokenizer.decode(outputs[0]))


def generate(model, messages, quant=True, max_new_tokens=80):
  tokenizer = AutoTokenizer.from_pretrained(model)
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
  attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
  streamer = TextStreamer(tokenizer)
  if quant:
    model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant_config).to("cuda")
  else:
    model = AutoModelForCausalLM.from_pretrained(model).to("cuda")
  
  outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, streamer=streamer)


generate(PHI, messages)

messages = [
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]

generate(GEMMA, messages, quant=False)
generate(QWEN, messages)
generate(DEEPSEEK, messages, quant=False, max_new_tokens=500)