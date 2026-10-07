# imports

import os
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HF_TOKEN')

audio_filename = "./denver_extract.mp3"
audio_file = open(audio_filename, "rb")

login(hf_token, add_to_git_credential=True)

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
AUDIO_MODEL = "gpt-4o-mini-transcribe"

# OPEN SOURCE WAY

from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16,
    device='cuda',
    return_timestamps=True
)

result = pipe(audio_filename)
transcription = result["text"]
print(transcription)
open_source_transcription = transcription


# USING A MODEL

openai = OpenAI(api_key=openai_api_key)
transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
print(transcription)

display(Markdown(open_source_transcription))
print("\n\n")
display(Markdown(transcription))


# Using a Real Model for Transcription

system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""

user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

# QUANTIZATION
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# get tokenizer for this model
tokenizer = AutoTokenizer.from_pretrained(LLAMA)

# set pad token to avoid errors
tokenizer.pad_token = tokenizer.eos_token

# generate inputs vectors/matrixes, use cuda for this and pytorch
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
print(inputs)

# define a streamer to stream response
streamer = TextStreamer(tokenizer)

# build the model from already trained llama, with the quant config.
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)

# call the model based on inputs
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

# display response (streamer does something similar)
response = tokenizer.decode(outputs[0])
print(outputs[0])

display(Markdown(response))