import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import discord
from discord.ext import commands
import json
import os

base_model = "./lora-merged"
tokenizer = AutoTokenizer.from_pretrained(base_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map={"": device},
    torch_dtype=torch.float16
)
model.eval()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=".", intents=intents)

MEMORY_FILE = "memory.json"  # persistent memory
user_memories = {}

def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(user_memories, f, ensure_ascii=False, indent=2)

def load_memory():
    global user_memories
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            user_memories = json.load(f)
    else:
        user_memories = {}
load_memory()

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Be a helpful and respectable imaginary roleplay doctor. "
        "You specialize in progeria, which is a disease that causes children to age rapidly. "
        "Answer as concisely as possible. "
        "If you don't know the answer, just say you don't know. "
    )
}

async def generate_response(user_id, user_input):
    if user_id not in user_memories:
        user_memories[user_id] = [SYSTEM_PROMPT.copy()]

    user_memories[user_id].append({
        "role": "user",
        "content": user_input
    })

    encodings = tokenizer.apply_chat_template(
        user_memories[user_id],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True
    )

    if isinstance(encodings, torch.Tensor):
        input_ids = encodings.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
    else:
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        streamer=streamer,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    for token in streamer:
        response += token
    
    user_memories[user_id].append({
        "role": "assistant",
        "content": response
    })
    save_memory()
    return response

@bot.command(name="ask")
async def ask(ctx, *, question):
    user_id = str(ctx.author.id)
    async with ctx.typing():
        response = await generate_response(user_id, question)
    await ctx.send(response)

@bot.command(name="reset")
async def reset(ctx):
    user_id = str(ctx.author.id)
    user_memories[user_id] = [SYSTEM_PROMPT.copy()]
    save_memory()
    await ctx.send("Memory reset.")

bot.run("Put_Your_Discord_Bot_Token_Here")
