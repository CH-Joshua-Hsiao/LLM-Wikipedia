import os
import re
import datetime
import json
from . import config
from .db import init_db, psycopg2, register_vector
from .llm import embed_text, query_llm

def get_safe_filename(name):
    name_no_dashes = name.replace("-", " ")
    clean = re.sub(r'[^\w\s]', '', name_no_dashes).strip()
    return re.sub(r'[\s]+', '_', clean) + ".md"

def log_action(action, details):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    with config.file_write_lock:
        with open("log.md", "a", encoding="utf-8") as f:
            f.write(f"## [{timestamp}] {action} | {details}\n")
    print(f"[{action}] {details}")

def get_existing_entities():
    """Return a list of known entities for taxonomy cross-referencing."""
    entities = []
    if os.path.exists(config.PAGES_DIR):
        for file in os.listdir(config.PAGES_DIR):
            if file.endswith(".md"):
                entities.append(file.replace(".md", ""))
    return entities

def resolve_entity(raw_name):
    """Calculates vector, checks Postgres, uses LLM if close matches found."""
    if not psycopg2:
        return raw_name 
    
    conn = init_db()
    if not conn:
        return raw_name
        
    vec = embed_text(raw_name)
    if not vec:
        conn.close()
        return raw_name
        
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM wiki_entities WHERE name = %s;", (raw_name,))
            exact = cur.fetchone()
            if exact:
                return exact[0]
                
            vec_literal = "[" + ",".join(map(str, vec)) + "]"
            cur.execute("""
                SELECT name, embedding <=> %s::vector AS distance 
                FROM wiki_entities 
                ORDER BY distance ASC 
                LIMIT 3;
            """, (vec_literal,))
            
            candidates = cur.fetchall()
            # Loosen mathematical threshold significantly to cast a wider net; rely on the LLM veto.
            close_candidates_set = set([c[0] for c in candidates if c[1] < 0.50])
            
            # String based similarity check
            import difflib
            cur.execute("SELECT name FROM wiki_entities;")
            all_names = [row[0] for row in cur.fetchall()]
            string_candidates = difflib.get_close_matches(raw_name, all_names, n=3, cutoff=0.6)
            for sc in string_candidates:
                close_candidates_set.add(sc)
                
            close_candidates = list(close_candidates_set)
            
            if close_candidates:
                 prompt = f"""
We extracted the entity name '{raw_name}'.
Our database has similar existing entities:
{json.dumps(close_candidates, ensure_ascii=False)}

Is '{raw_name}' conceptually identical or an exact alias to any of these existing entities? 
We want to actively consolidate overlapping topics! 

You MUST output ONLY a valid JSON object.
If it is a match, output: {{"match": true, "name": "EXACT_NAME_FROM_THE_LIST"}}
If it is a distinct, separate entity, output: {{"match": false, "name": null}}
"""
                 resp = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an entity resolution agent. Output ONLY valid JSON.")
                 if resp:
                     clean_resp = resp.strip()
                     if clean_resp.startswith("```json"): clean_resp = clean_resp[7:]
                     elif clean_resp.startswith("```"): clean_resp = clean_resp[3:]
                     if clean_resp.endswith("```"): clean_resp = clean_resp[:-3]
                     try:
                         data = json.loads(clean_resp.strip())
                         if data.get("match") and data.get("name"):
                             matched_name = data.get("name")
                             def normalize(s):
                                 return re.sub(r'[^\w]', '', str(s)).lower()
                             for c in close_candidates:
                                 if normalize(matched_name) == normalize(c):
                                     print(f"[RAG] Resolved alias '{raw_name}' -> '{c}'!")
                                     return c
                     except Exception as e:
                         pass
                        
            cur.execute("INSERT INTO wiki_entities (name, embedding) VALUES (%s, %s::vector) ON CONFLICT (name) DO NOTHING;", (raw_name, vec_literal))
            return raw_name
            
    except Exception as e:
        print(f"Entity resolution pipeline error: {e}")
        return raw_name
    finally:
        conn.close()
