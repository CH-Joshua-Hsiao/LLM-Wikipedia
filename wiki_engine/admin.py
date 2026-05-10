import os
import shutil
import datetime
import re

from . import config
from .db import init_db
from .utils import log_action
from .ingest import update_index

def rebuild_all_indices(division):
    """Wipes index.md and rebuilds it by reading every file in division pages."""
    print(f"Rebuilding index for division [{division}]...")
    pages_dir = config.get_pages_dir(division)
    if not os.path.exists(pages_dir):
        print(f"No pages found to index in {division}.")
        return

    # Reset index.md
    index_path = config.get_index_path(division)
    with config.file_write_lock:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n")

    files = [f for f in os.listdir(pages_dir) if f.endswith(".md")]
    new_files_data = []

    for filename in sorted(files):
        path = os.path.join(pages_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Determine type
        item_type = "entity"
        if re.search(r'\*\*type\*\*\s*:\s*concept', content, re.IGNORECASE):
            item_type = "concept"
            
        print(f"Summarizing {filename}...")
        new_files_data.append((filename, content, path, item_type))

    if new_files_data:
        update_index(new_files_data, division)
        print(f"Rebuilt {division} index with {len(new_files_data)} entries.")
        log_action("refresh_index", f"Manually rebuilt index for {division}.")

def reset():
    print("Global Reset: Wiping all divisions and databases...")
    
    if os.path.exists(config.NAMESPACES_DIR):
        try:
            shutil.rmtree(config.NAMESPACES_DIR)
            print("Deleted all namespaces.")
        except Exception as e:
            print(f"Failed to delete namespaces: {e}")
            
    with config.file_write_lock:
        with open("log.md", "w", encoding="utf-8") as f:
            pass
    log_action("reset", f"Global System reset initialized.")
    
    conn = init_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS wiki_entities, wiki_links, wiki_claims;")
            print("Cleaned PostgreSQL database tables.")
        except Exception as e:
            print(f"Error cleaning database tables: {e}")
        finally:
            conn.close()
    else:
        print("No PostgreSQL database connection available to drop tables.")
