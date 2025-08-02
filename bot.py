import os
import discord
import pdfplumber
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from educhain import Educhain, LLMConfig
from langchain_google_genai import ChatGoogleGenerativeAI
import webserver

# Load env variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load casebook PDF and process
def load_pdf_chunks(path, chunk_size=500, overlap=100):
    with pdfplumber.open(path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Build vector store from chunks
def build_vector_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, model

# Build prompt using query and top context
def build_prompt(query, chunks, index, model):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k=5)
    context = "\n\n".join([chunks[i] for i in I[0]])

    return f"""
You are a senior consulting mentor helping students prepare for case interviews.

Use the provided context from a casebook and your knowledge to answer the question.

Casebook context:
{context}

Student question:
{query}

Answer:"""

# Load model + vector DB once
chunks = load_pdf_chunks("casebook.pdf")
index, embeddings, embed_model = build_vector_index(chunks)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
client = Educhain(LLMConfig(custom_model=llm))

# Setup Discord bot
intents = discord.Intents.default()
intents.messages = True
client_bot = discord.Client(intents=intents)

@client_bot.event
async def on_ready():
    print(f'✅ Logged in as {client_bot.user}')

@client_bot.event
async def on_message(message):
    if message.author == client_bot.user:
        return

    if message.content.lower().startswith("!ask"):
        query = message.content[4:].strip()
        if not query:
            await message.channel.send("❗ Please ask a valid question after `!ask`.")
            return

        await message.channel.send("🤖 Thinking...")

        try:
            prompt = build_prompt(query, chunks, index, embed_model)
            response = llm.invoke(prompt)
            answer = response.content or response.text
            await message.channel.send(f"🧠 **Answer:**\n{answer}")
        except Exception as e:
            await message.channel.send(f"❌ Error: {e}")

# Start bot
webserver.keep_alive()
client_bot.run(DISCORD_BOT_TOKEN)
