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
NAMESPACES_DIR = "namespaces"
USE_OPENAI = True
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DB_URL = os.environ.get("POSTGRES_DB_URL", "postgresql://postgres:12345@localhost:5432/wiki_db")

def get_pages_dir(division):
    return os.path.join(NAMESPACES_DIR, division, "pages")

def get_archive_dir(division):
    return os.path.join(NAMESPACES_DIR, division, "archive")

def get_raw_dir(division):
    return os.path.join(NAMESPACES_DIR, division, "raw")

def get_index_path(division):
    return os.path.join(NAMESPACES_DIR, division, "index.md")

def init_directories(division):
    if division:
        os.makedirs(get_pages_dir(division), exist_ok=True)
        os.makedirs(get_archive_dir(division), exist_ok=True)
        os.makedirs(get_raw_dir(division), exist_ok=True)
