"""
wiki_agent.py: The Local LLM Wiki Automation Agent

This script manages your local knowledge base by parsing raw documents, querying the structured
data, and linting the wiki files.

PREREQUISITES:
- Your local LLM must be active (default port: 4000 via LiteLLM) and configured.
- For universal document parsing (PDF, DOCX, XLSX), ensure you install the extractors:
  `pip install pypdf python-docx pandas openpyxl`
- To scale Entity Resolution and Deep Linting, install pgvector:
  `pip install psycopg2-binary pgvector`
- To optionally use the OpenAI API, ensure the `OPENAI_API_KEY` environment variable is set.

USAGE:
1. Ingest a document:
   python wiki_agent.py ingest raw/sample.pdf [--openai]
   -> Automatically parses files, resolves aliases via DB, and builds markdown blocks locally.

2. Query the Knowledge Base:
   python wiki_agent.py query "Are there geopolitical concerns with TSMC?" [--openai]
   -> Generates an answer using the aggregated knowledge base.

3. Lint the Wiki:
   python wiki_agent.py lint [--openai]
   -> Daily Sub-Graph check: Isolates modified files and guarantees they don't break simple links.

   python wiki_agent.py lint --deep [--openai]
   -> Deep RAG Audit: Mathematically audits PostgreSQL to root out systemic logical contradictions instantly!

   python wiki_agent.py lint --fix [--openai]
   -> Auto-Fix & Restructure: Iterates through all wiki pages to correct grammar and automatically upgrade old key-value files into rich Wikipedia-style sections.
"""

import os
import sys
import json
import datetime
import requests
import re
import argparse
import shutil
import time

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None

# Configuration
API_BASE = "http://localhost:4000/v1"
MODEL_NAME = "GPTOSS-120B"
EMBED_MODEL = "nv-embed"
PAGES_DIR = "pages"
ARCHIVE_DIR = "archive"
USE_OPENAI = True
OPENAI_MODEL = "gpt-4o"
DB_URL = os.environ.get("POSTGRES_DB_URL", "postgresql://postgres:12345@localhost:5432/wiki_db")
# docker run --name wiki-postgres -e POSTGRES_PASSWORD=12345 -e POSTGRES_USER=postgres -e POSTGRES_DB=wiki_db -p 5432:5432 -d pgvector/pgvector:pg16
# Managing the Database later
# To stop the database: docker stop wiki-postgres
# To start it again: docker start wiki-postgres
# To delete it (and its data): docker rm wiki-postgres -f

# Ensure directories exist
os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def query_llm(messages, system_prompt="You are a helpful assistant for managing a local knowledge base."):
    if USE_OPENAI:
        url = "https://api.openai.com/v1/chat/completions"
        api_key = ""
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is missing! Export it before using --openai.")
            return None
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + messages
        }
    else:
        url = f"{API_BASE}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + messages
        }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {url}. Is your server running or internet connected?")
        return None
    except Exception as e:
        print(f"Error querying LLM: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response: {response.text}")
        return None

def embed_text(text):
    if USE_OPENAI:
        url = "https://api.openai.com/v1/embeddings"
        api_key = ""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {"input": text, "model": "text-embedding-3-small"}
    else:
        url = f"{API_BASE}/embeddings"
        headers = {"Content-Type": "application/json"}
        payload = {"input": text, "model": EMBED_MODEL}
        
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def init_db():
    if not psycopg2:
        return None
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS wiki_entities (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                embedding vector
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS wiki_links (
                source_file TEXT NOT NULL,
                target_file TEXT NOT NULL,
                PRIMARY KEY (source_file, target_file)
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS wiki_claims (
                id SERIAL PRIMARY KEY,
                source_file TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                embedding vector
            );
            """)
        return conn
    except Exception as e:
        print(f"PostgreSQL connection failed. DB Logic bypassed. Reason: {e}")
        return None

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
            close_candidates = [c[0] for c in candidates if c[1] < 0.25]
            
            if close_candidates:
                prompt = f"""
We extracted the entity name '{raw_name}'.
Our database has similar existing entities: {close_candidates}.
Is '{raw_name}' conceptually identical or an exact alias/translation to any of these existing entities? 
If YES, respond ONLY with the exact matching name from the list.
If NO (it is a distinct, separate entity), respond ONLY with the word NO.
"""
                resp = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an entity resolution agent. Output only the requested exact string.")
                if resp:
                    ans = resp.strip()
                    if ans in close_candidates:
                        print(f"[RAG] Resolved alias '{raw_name}' -> '{ans}'!")
                        return ans
                        
            cur.execute("INSERT INTO wiki_entities (name, embedding) VALUES (%s, %s::vector) ON CONFLICT (name) DO NOTHING;", (raw_name, vec_literal))
            return raw_name
            
    except Exception as e:
        print(f"Entity resolution pipeline error: {e}")
        return raw_name
    finally:
        conn.close()

def extract_and_embed_claims(filename, content):
    """Asks LLM to pull claims from text, embeds them, and shoves them into PostgreSQL wiki_claims table."""
    conn = init_db()
    if not conn: return
    
    print(f"Extracting sub-graph fact claims from {filename} for Deep Linting...")
    prompt = f"""
Read this wiki page and extract the core factual statements/claims into a pure JSON list of strings. 
Isolate distinct atomic facts (e.g. "TSMC plans 3nm volume in 2028").

Document:
{content}
"""
    response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an expert fact extractor. Output ONLY a valid JSON list of strings.")
    if not response: 
        conn.close()
        return
        
    try:
        clean = response.strip()
        if clean.startswith("```json"): clean = clean[7:]
        if clean.startswith("```"): clean = clean[3:]
        if clean.endswith("```"): clean = clean[:-3]
        claims = json.loads(clean.strip())
        
        if isinstance(claims, list):
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM wiki_claims WHERE source_file = %s;", (filename,))
                for claim in claims:
                    vec = embed_text(claim)
                    if vec:
                        vec_literal = "[" + ",".join(map(str, vec)) + "]"
                        cur.execute("INSERT INTO wiki_claims (source_file, claim_text, embedding) VALUES (%s, %s, %s::vector);", (filename, claim, vec_literal))
            print(f"Embedded {len(claims)} fact claims for {filename} into vector storage.")
    except Exception as e:
        print(f"Failed to parse claims JSON for {filename}: {e}")
    finally:
        conn.close()

def update_backlinks(filename, content):
    """Parses new content for edges, upserts to DB, and rewrites target files locally."""
    conn = init_db()
    if not conn: return
    
    link_pattern = re.compile(r'\[.*?\]\(pages/(.*?\.md)\)')
    targets = set(link_pattern.findall(content))
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM wiki_links WHERE source_file = %s;", (filename,))
            if targets:
                for t in targets:
                    cur.execute("INSERT INTO wiki_links (source_file, target_file) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (filename, t))
                    
                # Now fetch backlinks for each target and append locally
                for t in targets:
                    cur.execute("SELECT source_file FROM wiki_links WHERE target_file = %s;", (t,))
                    backlink_sources = [row[0] for row in cur.fetchall()]
                    
                    target_path = os.path.join(PAGES_DIR, t)
                    if os.path.exists(target_path):
                        with open(target_path, "r", encoding="utf-8") as f:
                            t_content = f.read()
                            
                        # Remove existing Backlinks section if it exists to rewrite it cleanly
                        parts = t_content.split("## Backlinks")
                        main_body = parts[0].strip()
                        
                        if backlink_sources:
                            backlink_section = "\n\n## Backlinks\n"
                            for s in sorted(backlink_sources):
                                display = s.replace(".md", "").replace("_", " ")
                                backlink_section += f"- [{display}]({s})\n"
                                
                            with open(target_path, "w", encoding="utf-8") as f:
                                f.write(main_body + backlink_section)
    except Exception as e:
        print(f"Error updating backlinks for {filename}: {e}")
    finally:
        conn.close()

def log_action(action, details):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    with open("log.md", "a", encoding="utf-8") as f:
        f.write(f"## [{timestamp}] {action} | {details}\n")
    print(f"[{action}] {details}")

def get_existing_entities():
    """Return a list of known entities for taxonomy cross-referencing."""
    entities = []
    if os.path.exists(PAGES_DIR):
        for file in os.listdir(PAGES_DIR):
            if file.endswith(".md"):
                entities.append(file.replace(".md", ""))
    return entities

def merge_and_save_entity(filename, new_content):
    target_path = os.path.join(PAGES_DIR, filename)
    taxonomy = get_existing_entities()
    taxonomy_str = ", ".join(taxonomy)

    if os.path.exists(target_path):
        # Archive old version
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{filename.replace('.md', '')}_{timestamp}.md"
        shutil.copy2(target_path, os.path.join(ARCHIVE_DIR, archive_name))
        
        # Read old content
        with open(target_path, "r", encoding="utf-8") as f:
            old_content = f.read()
            
        print(f"Consolidating existing entity: {filename}...")
        
        prompt = f"""
You are a Wikipedia editor. An entity document already exists, but new information has been ingested.
Your task is to merge the New Data into the Existing Document intelligently.
Keep all historical facts, seamlessly weave in the new facts (e.g. extending timelines or sections), and format beautifully in Markdown.

Whenever you mention other known entities in the text, wrap them in Wiki links like `[Entity_Name](pages/Entity_Name.md)`.
Here is the list of currently known entities in the database:
{taxonomy_str}

=== Existing Document ===
{old_content}

=== New Data to Merge ===
{new_content}

Return ONLY the fully merged, comprehensive markdown representation.
"""
        response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an expert technical editor. Output markdown only.")
        if response:
            merged_content = response.strip()
            if merged_content.startswith("```markdown"): merged_content = merged_content[11:]
            if merged_content.startswith("```"): merged_content = merged_content[3:]
            if merged_content.endswith("```"): merged_content = merged_content[:-3]
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(merged_content.strip())
            print(f"Merged and Updated {target_path}")
            
            extract_and_embed_claims(filename, merged_content.strip())
            update_backlinks(filename, merged_content.strip())
            return target_path
    
    prompt = f"""
You are creating a new Wikipedia-style entity document based on the provided raw data.
Please rewrite and structure the raw information into a rich, comprehensive, and beautiful markdown page.
Organize the facts clearly into logical sections such as '## Overview' and '## Key Details' (or specific topics like 'Financials', 'Technology', etc. based on the data).
Synthesize the facts into cohesive paragraphs or bullet points, rather than a raw dump of key-value pairs.

Ensure the output includes a `# Title`, the `**Type**`, and retains the `**Source**` link from the original data.

Additionally, whenever you mention any of these known entities: {taxonomy_str}
Wrap them in Wiki links like `[Entity_Name](pages/Entity_Name.md)`.

=== Raw Data Content ===
{new_content}

Return ONLY the beautifully formatted markdown code.
"""
    response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an expert technical editor. Output markdown only.")
    final_content = new_content
    if response:
         cleaned = response.strip()
         if cleaned.startswith("```markdown"): cleaned = cleaned[11:]
         if cleaned.startswith("```"): cleaned = cleaned[3:]
         if cleaned.endswith("```"): cleaned = cleaned[:-3]
         final_content = cleaned.strip()

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"Created {target_path}")
    
    extract_and_embed_claims(filename, final_content)
    update_backlinks(filename, final_content)
    return target_path

def extract_text_from_file(file_path):
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        if not pypdf: return "Error: pypdf library is not installed."
        try:
            reader = pypdf.PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e: return f"Error extracting PDF: {e}"
    elif ext == ".docx":
        if not docx: return "Error: python-docx library is not installed."
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e: return f"Error extracting DOCX: {e}"
    elif ext == ".xlsx":
        if not pd: return "Error: pandas and openpyxl are not installed."
        try:
            dfs = pd.read_excel(file_path, sheet_name=None)
            output = []
            for sheet_name, df in dfs.items():
                output.append(f"=== Sheet: {sheet_name} ===")
                sheet_str = df.to_csv(index=False)
                if len(sheet_str) > 50000:
                    sheet_str = sheet_str[:50000] + "\n... [DATA TRUNCATED]"
                output.append(sheet_str)
                output.append("")
            return "\n".join(output)
        except Exception as e: return f"Error extracting XLSX: {e}"
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f: return f.read()
        except Exception as e: return f"Error extracting: {e}"

def handle_excel_ingest(file_path):
    if not pd:
        print("Error: pandas and openpyxl are missing. Run 'pip install pandas openpyxl'.")
        return
    print(f"Analyzing schema for large Excel file '{file_path}' using {MODEL_NAME}...")
    dfs = pd.read_excel(file_path, sheet_name=None)
    
    for sheet_name, df in dfs.items():
        if df.empty: continue
        print(f"Asking LLM to generate Python parser for sheet: {sheet_name} (Sampling top 500 rows)...")
        schema_csv = df.head(500).to_csv(index=False)
        
        prompt = f"""
I have a pandas DataFrame containing a knowledge base dataset. Here are the first 500 rows in CSV format:
{schema_csv}

I want to iteratively extract all important entities/concepts into markdown files based on that schema.
Write a raw Python function named `extract_entities(df)` that iterates through the DataFrame `df` and returns a list of dictionaries.
Each dictionary MUST have two keys: 'filename' (e.g. 'Company_Name.md') and 'content' (the markdown string).
CRITICAL RULE: Filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md'). NEVER use generic sub-topic names like 'Financials.md'. If a row is just a sub-topic of a parent entity, use the parent entity's filename and map the data into its content.
Format the markdown 'content' beautifully. Include at least: '# Title', '**Type**: Entity', '**Source**: {file_path}', and map the core data from the row into paragraphs or lists.
OUTPUT ONLY THE PIPELINE FUNCTION CODE. No explanatory text. No markdown formatting.
"""
        response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an expert pandas software engineer. Output raw python code only, starting with `def extract_entities(df):`")
        if not response: continue
            
        code = response.strip()
        if code.startswith("```python"): code = code[9:]
        elif code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
            
        print(f"[LLM parsed pattern] Executing generated mapping script over {len(df)} row dataset locally...")
        namespace = {'pd': pd, 'json': json, 're': re}
        try:
            exec(code.strip(), namespace)
            if 'extract_entities' not in namespace: continue
            extracted_data = namespace['extract_entities'](df)
            
            new_files = []
            for item in extracted_data:
                filename = item.get("filename", "")
                if not filename: continue
                raw_name = filename.replace(".md", "").replace("_", " ")
                resolved_name = resolve_entity(raw_name)
                final_filename = resolved_name.replace(" ", "_").replace("/", "").replace("\\", "") + ".md"
                content = item.get("content", "")
                
                target_path = merge_and_save_entity(final_filename, content)
                new_files.append((final_filename, content, target_path))

            if new_files: update_index(new_files)

        except Exception as e:
            print(f"Failed to execute LLM-written extraction code for sheet '{sheet_name}':\n{e}\n\nGenerated Code:\n{code}")
    log_action("ingest_hybrid_xlsx", os.path.basename(file_path))

def handle_json_ingest(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file '{file_path}': {e}")
        return

    print(f"Analyzing schema for JSON file '{file_path}' using {MODEL_NAME}...")
    
    # Sample the JSON structure to avoid context limits
    def sample_data(d, max_items=10):
        if isinstance(d, list):
            return [sample_data(item, max_items=2) for item in d[:max_items]]
        elif isinstance(d, dict):
            return {k: sample_data(v, max_items=2) for k, v in list(d.items())[:max_items]}
        return d

    sampled_json = json.dumps(sample_data(data, max_items=10), indent=2)
    
    prompt = f"""
I have loaded a JSON dataset. Here is a sample of its structure:
```json
{sampled_json}
```

I want to creatively extract important entities/concepts into markdown files based on this structure.
Write a raw Python function named `extract_entities(data)` that takes the full parsed JSON object `data` and returns a list of dictionaries.
Each dictionary MUST have two keys: 'filename' (e.g. 'Company_Name.md') and 'content' (the markdown string).
CRITICAL RULE: Filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md'). NEVER use generic sub-topic names like 'Financials.md'. If an item is just a sub-topic of a parent entity, use the parent entity's filename and map the data into its content.
Format the markdown 'content' beautifully. Include at least: '# Title', '**Type**: Entity', '**Source**: {file_path}', and map the core facts from the JSON into paragraphs or lists.
OUTPUT ONLY THE PIPELINE FUNCTION CODE. No explanatory text. No markdown formatting.
"""
    response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an expert Python software engineer. Output raw python code only, starting with `def extract_entities(data):`")
    if not response: return
        
    code = response.strip()
    if code.startswith("```python"): code = code[9:]
    elif code.startswith("```"): code = code[3:]
    if code.endswith("```"): code = code[:-3]
        
    print(f"[LLM parsed pattern] Executing generated mapping script over JSON dataset locally...")
    namespace = {'json': json, 're': re}
    try:
        exec(code.strip(), namespace)
        if 'extract_entities' not in namespace: 
            print("Failed: Model did not generate 'extract_entities' function.")
            return

        extracted_data = namespace['extract_entities'](data)
        
        new_files = []
        for item in extracted_data:
            filename = item.get("filename", "")
            if not filename: continue
            raw_name = filename.replace(".md", "").replace("_", " ")
            resolved_name = resolve_entity(raw_name)
            final_filename = resolved_name.replace(" ", "_").replace("/", "").replace("\\", "") + ".md"
            content = item.get("content", "")
            
            target_path = merge_and_save_entity(final_filename, content)
            new_files.append((final_filename, content, target_path))

        if new_files: update_index(new_files)

    except Exception as e:
        print(f"Failed to execute LLM-written extraction code for JSON:\\n{e}\\n\\nGenerated Code:\\n{code}")
    log_action("ingest_hybrid_json", os.path.basename(file_path))

def ingest(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    _, ext = os.path.splitext(file_path.lower())
    if ext == ".xlsx":
        handle_excel_ingest(file_path)
        return
    elif ext == ".json":
        handle_json_ingest(file_path)
        return

    print(f"Extracting text from {file_path}...")
    content = extract_text_from_file(file_path)
    if content.startswith("Error"):
        print(content)
        return

    print(f"Ingesting into semantic entities using {MODEL_NAME}...")
    prompt = f"""
I have extracted text from a document. I want to identify the key entities (e.g., companies, people, technologies) and concepts.
For each important entity or concept, provide a summary formatted as a Markdown file.

CRITICAL RULE: All filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md', 'Tim_Cook.md'). NEVER use generic sub-topic names like 'Financials.md' or 'Q2_Earnings.md'. If the extracted data is a sub-topic of a parent entity, you MUST use the parent entity's filename (e.g., 'Apple_Inc.md') and structure the sub-topic data into its content block.

Your output must be strictly in JSON format, like this:
{{
  "entities": [
    {{
      "filename": "Entity_Name.md",
      "content": "# Entity Name\\n\\n**Type**: Entity\\n**Sources**: [{file_path}]({file_path})\\n\\n## Overview\\nSome information..."
    }}
  ]
}}

Here is the raw text to process:
{content}
"""
    response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a data extraction assistant. Output ONLY valid JSON.")
    if not response: return

    try:
        clean_response = response.strip()
        if clean_response.startswith("```json"): clean_response = clean_response[7:]
        if clean_response.endswith("```"): clean_response = clean_response[:-3]
        extracted_data = json.loads(clean_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as valid JSON: {e}")
        return

    new_files = []
    for entity in extracted_data.get("entities", []) + extracted_data.get("concepts", []):
        filename = entity.get("filename", "")
        if not filename: continue
        raw_name = filename.replace(".md", "").replace("_", " ")
        resolved_name = resolve_entity(raw_name)
        final_filename = resolved_name.replace(" ", "_").replace("/", "").replace("\\", "") + ".md"
        file_content = entity.get("content", "")
        
        target_path = merge_and_save_entity(final_filename, file_content)
        new_files.append((final_filename, file_content, target_path))

    if new_files: update_index(new_files)
    log_action("ingest", os.path.basename(file_path))

def update_index(new_files):
    if not os.path.exists("index.md"):
        with open("index.md", "w", encoding="utf-8") as f:
            f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n")
            
    with open("index.md", "r", encoding="utf-8") as f:
        index_content = f.read()

    modified = False
    for filename, _, target_path in new_files:
        name_display = filename.replace(".md", "").replace("_", " ")
        entry = f"- [{name_display}]({PAGES_DIR}/{filename})"
        if entry not in index_content:
            index_content += f"{entry}\n"
            modified = True
            
    if modified:
        with open("index.md", "w", encoding="utf-8") as f:
            f.write(index_content)
        print("Updated index.md with new entries.")

def query(question):
    print(f"Querying knowledge base: {question}")
    context = ""
    for file in os.listdir(PAGES_DIR):
        if file.endswith(".md"):
            path = os.path.join(PAGES_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                context += f"--- {file} ---\n{f.read()}\n\n"
                
    prompt = f"Using the following knowledge base context, answer the user's question. If the answer is not in the context, say so.\nContext:\n{context}\n\nQuestion: {question}"
    answer = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a helpful analyst.")
    if answer:
        print("\n--- Answer ---\n")
        print(answer)
        print("\n--------------\n")
        log_action("query", question)

def lint_hygiene():
    print("Running Daily Hygiene (Targeted Sub-Graph)...")
    now = time.time()
    one_day_ago = now - 86400
    
    modified_files = []
    if os.path.exists(PAGES_DIR):
        for file in os.listdir(PAGES_DIR):
            if file.endswith(".md"):
                path = os.path.join(PAGES_DIR, file)
                if os.path.getmtime(path) > one_day_ago:
                    modified_files.append(file)
                
    if not modified_files:
        print("No files modified in the last 24 hours. Hygiene check passed blindly!")
        return

    subgraph_files = set(modified_files)
    link_pattern = re.compile(r'\[.*?\]\((.*?\.md)\)')
    
    for mod_file in modified_files:
        path = os.path.join(PAGES_DIR, mod_file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            neighbors = link_pattern.findall(content)
            for neighbor in neighbors:
                if neighbor.startswith(PAGES_DIR + "/"):
                    neighbor = neighbor[len(PAGES_DIR)+1:]
                if neighbor in os.listdir(PAGES_DIR):
                    subgraph_files.add(neighbor)
                    
    print(f"Identified {len(subgraph_files)} interconnected nodes based on recent edits.")
    
    markdown_content = ""
    for file in subgraph_files:
        path = os.path.join(PAGES_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            snippet = f.read()[:800]
            markdown_content += f"--- {file} ---\n{snippet}...\n\n"

    prompt = f"""
Review the following markdown files which represent recently modified interconnected nodes.
1. Check for basic grammar and spelling.
2. Ensure external/formatting links seem logical.

Output a brief list of suggested fixes or state that everything looks good.

Files:
{markdown_content}
"""
    suggestion = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a strict technical writer.")
    if suggestion:
        print("\n--- Sub-Graph Hygiene Report ---\n")
        print(suggestion)
        log_action("lint", "Ran targeted hygiene sub-graph check")

def lint_deep():
    print("Running Deep RAG Contradiction Audit...")
    conn = init_db()
    if not conn:
        print("Cannot run deep linting without PostgreSQL connection.")
        return
        
    try:
        register_vector(conn)
        with conn.cursor() as cur:
             cur.execute("""
                 SELECT a.source_file, a.claim_text, b.source_file, b.claim_text, a.embedding <=> b.embedding AS distance
                 FROM wiki_claims a
                 JOIN wiki_claims b ON a.id < b.id 
                 WHERE a.source_file != b.source_file AND (a.embedding <=> b.embedding) < 0.15
                 ORDER BY distance ASC
                 LIMIT 10;
             """)
             suspicious_pairs = cur.fetchall()
             
        if not suspicious_pairs:
            print("No suspiciously overlapping claims mapped across different core entities. Graph holds logically sound!")
            log_action("lint_deep", "Audited cleanly")
            return
            
        print(f"Mathematical Flag: Found {len(suspicious_pairs)} highly similar semantic trajectories spanning varied entity nodes. Dispatching verification probe...")
        
        contradictions_found = 0
        for pair in suspicious_pairs:
            file_a, claim_a, file_b, claim_b, dist = pair
            prompt = f"""
We found two claims originating from entirely different documents that measure highly similar on dimensional space.
Claim 1 (from {file_a}): "{claim_a}"
Claim 2 (from {file_b}): "{claim_b}"

Are these simply repetitive facts mapping independent resources, or do these two factual claims genuinely CONTRADICT each other contextually? 
If there is NO contradiction, simply respond mathematically exactly with "NO".
If there IS a contradiction, explain precisely why and recommend a remedy.
"""
            resp = query_llm([{"role": "user", "content": prompt}])
            if resp and resp.strip().upper() != "NO":
                print(f"\n[!] POTENTIAL CONTRADICTION ({file_a} vs {file_b}):")
                print(resp.strip())
                contradictions_found += 1
                
        if contradictions_found == 0:
            print("\nLLM referee verified all mathematically suspicious cross-checks. Zero semantic contradictions verified!")
            
        log_action("lint_deep", f"RAG Audited {len(suspicious_pairs)} mathematical proximities. Found {contradictions_found} human contradictions.")
            
    except Exception as e:
        print(f"Deep Lint error breakdown: {e}")
    finally:
        conn.close()

def lint_fix_all():
    print("Running Auto-Fix & Restructure on all pages...")
    taxonomy = get_existing_entities()
    taxonomy_str = ", ".join(taxonomy)
    fixed_count = 0
    if not os.path.exists(PAGES_DIR):
        print("No pages to fix.")
        return

    for file in os.listdir(PAGES_DIR):
        if file.endswith(".md"):
            path = os.path.join(PAGES_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"Revising {file}...")
            prompt = f"""
You are a Wikipedia editor. Please review the following markdown file.
1. Fix any basic grammar or spelling mistakes.
2. Restructure the document into a rich, comprehensive, and beautiful Wikipedia-style page if it is not already. 
   - Organize facts into logical sections like '## Overview' and '## Key Details' (or specific topics based on the data).
   - Convert simple key-value dumps into cohesive paragraphs or bullet points.
3. Keep the `# Title`, `**Type**`, and the `**Source**` link intact.
4. Whenever you mention any of these known entities: {taxonomy_str}
   Wrap them in Wiki links like `[Entity_Name](pages/Entity_Name.md)`.

=== Content ===
{content}

Return ONLY the beautifully formatted markdown code. No explanatory text.
"""
            response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an expert technical editor. Output markdown only.")
            if response:
                cleaned = response.strip()
                if cleaned.startswith("```markdown"): cleaned = cleaned[11:]
                if cleaned.startswith("```"): cleaned = cleaned[3:]
                if cleaned.endswith("```"): cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                if cleaned:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                    fixed_count += 1
                    
    print(f"\\nAuto-fixed and upgraded {fixed_count} pages!")
    log_action("lint_fix", f"Restructured and grammar-checked {fixed_count} pages.")

def lint(deep=False, fix=False):
    if fix:
        lint_fix_all()
    elif deep:
        lint_deep()
    else:
        lint_hygiene()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wiki_agent.py: Local LLM Wiki Automation Agent")
    parser.add_argument("command", choices=["ingest", "query", "lint"], help="Command to execute")
    parser.add_argument("args", nargs="*", help="Arguments for the command.")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API instead of local LiteLLM")
    parser.add_argument("--deep", action="store_true", help="RAG systemic contradiction audit (lint only)")
    parser.add_argument("--fix", action="store_true", help="Automatically revise and restructure all wiki pages during linting")
    
    args = parser.parse_args()
    if args.openai: USE_OPENAI = True
        
    cmd = args.command
    if cmd == "ingest":
        if not args.args:
            print("Usage: python wiki_agent.py ingest <file_path>")
            sys.exit(1)
        ingest(args.args[0])
    elif cmd == "query":
        if not args.args:
            print("Usage: python wiki_agent.py query \"<question>\"")
            sys.exit(1)
        query(args.args[0])
    elif cmd == "lint":
        lint(deep=args.deep, fix=args.fix)
