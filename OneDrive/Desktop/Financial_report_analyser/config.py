import os
from dotenv import load_dotenv

load_dotenv()

MODEL_KEY = os.getenv("MODEL_KEY")
NVIDIA_EMBEDDING_KEY = os.getenv("NVIDIA_EMBEDDING_KEY")
NVIDIA_HYDE_KEY = os.getenv("NVIDIA_HYDE_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CHROMA_DB_PATH = "chroma_db_finance"