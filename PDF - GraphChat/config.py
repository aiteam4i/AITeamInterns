import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# LLM Model names
LLM_MODEL_GRAPH_EXTRACTION = "llama-3.3-70b-versatile" 
LLM_MODEL_RAG = "llama-3.3-70b-versatile"

