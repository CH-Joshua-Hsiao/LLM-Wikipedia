import os
import shutil
import datetime
import re

from . import config
from .db import init_db
from .utils import log_action
from .ingest import update_index

def rebuild_all_indices():
    """Wipes index.md and rebuilds it by reading every file in PAGES_DIR."""
    print("Rebuilding entire index with intelligent summaries...")
    if not os.path.exists(config.PAGES_DIR):
        print("No pages found to index.")
        return

    # Reset index.md
    with config.file_write_lock:
        with open("index.md", "w", encoding="utf-8") as f:
            f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n")

    files = [f for f in os.listdir(config.PAGES_DIR) if f.endswith(".md")]
    new_files_data = []

    for filename in sorted(files):
        path = os.path.join(config.PAGES_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Determine type
        item_type = "entity"
        if re.search(r'\*\*type\*\*\s*:\s*concept', content, re.IGNORECASE):
            item_type = "concept"
            
        print(f"Summarizing {filename}...")
        new_files_data.append((filename, content, path, item_type))

    if new_files_data:
        update_index(new_files_data)
        print(f"Rebuilt index with {len(new_files_data)} entries.")
        log_action("refresh_index", f"Manually rebuilt index for {len(new_files_data)} files.")

def reset():
    print("Resetting Knowledge Base...")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    moved_count = 0
    if os.path.exists(config.PAGES_DIR):
        for file in os.listdir(config.PAGES_DIR):
            if file.endswith(".md"):
                src = os.path.join(config.PAGES_DIR, file)
                dst = os.path.join(config.ARCHIVE_DIR, f"{file.replace('.md', '')}_{timestamp}.md")
                shutil.move(src, dst)
                moved_count += 1
    print(f"Moved {moved_count} pages to archive.")
    
    with config.file_write_lock:
        with open("index.md", "w", encoding="utf-8") as f:
            f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n")
    print("Cleared index.md.")
    
    with config.file_write_lock:
        with open("log.md", "w", encoding="utf-8") as f:
            pass
    log_action("reset", f"System reset initialized. Archived {moved_count} pages.")
    
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
