import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY

GROQ_API_BASE = "https://api.groq.com/openai/v1"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"