import os
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

file_write_lock = threading.Lock()

API_BASE = "http://localhost:4000/v1"
MODEL_NAME = "GPTOSS-120B"
EMBED_MODEL = "nv-embed"
PAGES_DIR = "pages"
ARCHIVE_DIR = "archive"
USE_OPENAI = True
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DB_URL = os.environ.get("POSTGRES_DB_URL", "postgresql://postgres:12345@localhost:5432/wiki_db")

def init_directories():
    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
