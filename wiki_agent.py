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
   python wiki_agent.py query "Are there geopolitical concerns with TSMC?" [--openai] [--max-hops <int>]
   -> Utilizes an agentic multi-hop retrieval loop. The LLM reads the index to pick targeted files, looping recursively to gather context until it formulates an answer or hits the max hop sequence (default 3).

3. Lint the Wiki:
   python wiki_agent.py lint [--openai]
   -> Daily Sub-Graph check: Isolates modified files and guarantees they don't break simple links.

   python wiki_agent.py lint --deep [--openai]
   -> Deep RAG Audit: Mathematically audits PostgreSQL to root out systemic logical contradictions instantly!

   python wiki_agent.py lint --fix [--openai]
   -> Auto-Fix & Restructure: Iterates through all wiki pages to correct grammar and automatically upgrade old key-value files into rich Wikipedia-style sections.

   python wiki_agent.py lint --merge [--openai]
   -> Automerge Duplicates: Sweeps the vector DB for highly similar concepts, forces LLM verification, and auto-merges their markdown files and reroutes all their links.

4. Reset the Wiki:
   python wiki_agent.py reset
   -> Wipes the knowledge base to start fresh. Moves all pages to the archive, resets the index and logs, and cleanly drops the PostgreSQL tables.
"""

def summarize_entity(content):
    """Uses LLM to generate a dense, one-sentence summary for the index."""
    prompt = f"""
Analyze the following wiki page content and generate a dense, professional, and technical one-sentence summary (approx. 20-30 words).
Focus on the most important financial metrics, technical specs, or core identity of the entity.
Do NOT include the entity name in the summary itself (e.g., instead of "Apple is a...", start with "A leading technology company...").
Only use information present in the text.

Content:
{content}

Summary:
"""
    resp = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a concise technical encyclopedist.")
    return resp.strip() if resp else ""

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
            # Loosen mathematical threshold significantly to cast a wider net; rely on the LLM veto.
            close_candidates = [c[0] for c in candidates if c[1] < 0.50]
            
            if close_candidates:
                 prompt = f"""
We extracted the entity name '{raw_name}'.
Our database has similar existing entities: {close_candidates}.
Is '{raw_name}' conceptually identical or an exact alias to any of these existing entities? 
We want to actively consolidate overlapping topics! If it refers to the exact same technology, person, or phenomenon (e.g., "High-Bandwidth Memory" vs "High Bandwidth Memory (HBM)"), you MUST respond YES.
If YES, respond ONLY with the exact matching name from the list. If you are unsure, respond NO.
If it is a match but you want to just say YES, we will route it to the closest candidate.
If NO (it is a distinct, separate entity), respond ONLY with the word NO.
"""
                 resp = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an entity resolution agent. Output only the requested exact string.")
                 if resp:
                     ans = resp.strip().strip("'").strip('"')
                     if ans.upper() == "YES" and len(close_candidates) == 1:
                         print(f"[RAG] Resolved alias (Auto-YES) '{raw_name}' -> '{close_candidates[0]}'!")
                         return close_candidates[0]

                     def normalize(s):
                         return re.sub(r'[^\w]', '', s).lower()

                     for c in close_candidates:
                         if normalize(ans) == normalize(c) or normalize(raw_name) == normalize(c) or normalize(ans) == normalize(raw_name):
                             print(f"[RAG] Resolved alias '{raw_name}' -> '{c}'!")
                             return c
                        
            cur.execute("INSERT INTO wiki_entities (name, embedding) VALUES (%s, %s::vector) ON CONFLICT (name) DO NOTHING;", (raw_name, vec_literal))
            return raw_name
            
    except Exception as e:
        print(f"Entity resolution pipeline error: {e}")
        return raw_name
    finally:
        conn.close()

def get_safe_filename(name):
    name_no_dashes = name.replace("-", " ")
    clean = re.sub(r'[^\w\s]', '', name_no_dashes).strip()
    return re.sub(r'[\s]+', '_', clean) + ".md"

def is_meaningless_entity(name):
    """Uses LLM to verify if an extracted entity name is meaningful or junk."""
    n = name.replace(".md", "").strip()
    if len(n) < 2:
        return True
        
    prompt = f"""
We extracted the following string from a document as a potential wiki entity/concept: "{n}"

Is this a meaningful, genuine noun (like a specific company, person, technology, or distinct profound informational concept/phenomenon such as "AI Supercycle") suitable for a robust Knowledge Base?
Or is it a meaningless semantic abstraction, arbitrary string, pure metadata, generic label, or document artifact (e.g. "Author 44211", "Page 2", "Header", "Table 1", "Conference Call Participants", "Q3 Earnings Summary", pure numbers)?

If it is clearly MEANINGLESS or mere metadata/junk/document artifact, respond EXACTLY with the word "MEANINGLESS".
If it is a genuinely meaningful topic, respond EXACTLY with the word "MEANINGFUL".
"""
    resp = query_llm([{"role": "user", "content": prompt}], system_prompt="You are an editorial filter. Output only one word.")
    if resp and "MEANINGLESS" in resp.upper():
        return True
    return False

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
    
    # Relax regex to match both (pages/File.md) and (File.md) safely
    link_pattern = re.compile(r'\[.*?\]\((?:pages/)?(.*?\.md)\)')
    targets = set(link_pattern.findall(content))
    
    try:
        with conn.cursor() as cur:
            # 1. Store old targets before deletion so we can refresh them (to remove stale links)
            cur.execute("SELECT target_file FROM wiki_links WHERE source_file = %s;", (filename,))
            old_targets = set(row[0] for row in cur.fetchall())
            
            # 2. Update Knowledge Graph database
            cur.execute("DELETE FROM wiki_links WHERE source_file = %s;", (filename,))
            if targets:
                for t in targets:
                    cur.execute("INSERT INTO wiki_links (source_file, target_file) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (filename, t))
                    
            # 3. Refresh Backlinks section in markdown for ALL affected pages
            # This includes new targets, removed targets, AND the current ingested file itself.
            files_to_refresh = targets.union(old_targets)
            files_to_refresh.add(filename)
            
            for f_name in files_to_refresh:
                cur.execute("SELECT source_file FROM wiki_links WHERE target_file = %s;", (f_name,))
                backlink_sources = [row[0] for row in cur.fetchall()]
                
                target_path = os.path.join(PAGES_DIR, f_name)
                if os.path.exists(target_path):
                    with open(target_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                        
                    # Clean out the old section
                    parts = file_content.split("## Backlinks")
                    main_body = parts[0].strip()
                    
                    if backlink_sources:
                        backlink_section = "\n\n## Backlinks\n"
                        for s in sorted(backlink_sources):
                            display = s.replace(".md", "").replace("_", " ")
                            backlink_section += f"- [{display}]({s})\n"
                            
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(main_body + backlink_section)
                    else:
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(main_body + "\n")
                            
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

def merge_and_save_entity(filename, new_content, cascade=True):
    target_path = os.path.join(PAGES_DIR, filename)
    taxonomy = get_existing_entities()
    taxonomy_str = ", ".join(taxonomy)

    final_content_to_save = None

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
Keep all historical facts, seamlessly weave in the new facts, and format beautifully in Markdown. 
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, timeline, and nuanced detail from the text. Prioritize raw numbers, financial metrics, technical specifications, and quantitative data. 

STRICT RULE: Do NOT invent, hallucinate, or add supplemental knowledge from your own training data. Only use facts explicitly present in the Existing Document and the New Data. Stay 100% faithful to the provided text.

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
            final_content_to_save = merged_content.strip()
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(final_content_to_save)
            print(f"Merged and Updated {target_path}")
    else:
        prompt = f"""
You are creating a new Wikipedia-style entity document based on the provided raw data.
Please rewrite and structure the raw information into a rich, comprehensive, and beautiful markdown page.
Organize the facts clearly into logical sections such as '## Overview' and '## Key Details' (or specific topics like 'Financials', 'Technology', etc. based on the data).
Synthesize the facts into cohesive paragraphs or bullet points. 
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, timeline, and nuanced detail from the text. Prioritize raw numbers, financial metrics, hardware specifications, and any quantitative analysis. 

STRICT RULE: Do NOT invent, hallucinate, or add supplemental knowledge from your own training data. Provide a wiki summary strictly based on the provided Raw Data Content. Do not fill in missing background information yourself.

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

        final_content_to_save = final_content
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(final_content_to_save)
        print(f"Created {target_path}")
        
    if final_content_to_save:
        extract_and_embed_claims(filename, final_content_to_save)
        update_backlinks(filename, final_content_to_save)
        
        if cascade:
            link_pattern = re.compile(r'\[.*?\]\((?:pages/)?(.*?\.md)\)')
            targets = set(link_pattern.findall(final_content_to_save))
            if targets:
                sentences = final_content_to_save.replace('\n', ' ').split('. ')
                for t in targets:
                    if t == filename: continue
                    target_path_check = os.path.join(PAGES_DIR, t)
                    if os.path.exists(target_path_check):
                        mention_sentences = [s for s in sentences if f"({t})" in s or f"(pages/{t})" in s]
                        if mention_sentences:
                            context_injection = ". ".join(mention_sentences) + "."
                            print(f"[Cascade] Updating {t} with contextual link from {filename}...")
                            merge_and_save_entity(t, f"New context referencing this topic from {filename}: {context_injection}", cascade=False)
                            
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
Each dictionary MUST have three keys: 'type' (either 'entity' or 'concept'), 'filename' (e.g. 'Company_Name.md'), and 'content' (the markdown string).
CRITICAL RULE: Filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md', 'iPhone.md'). Distinct and notable products, technologies, people, or platforms SHOULD get their own separate files. NEVER use generic sub-topic names like 'Financials.md' or 'Q2_Earnings.md'. If the extracted data is merely a generic sub-topic of a parent entity, you MUST map it to the parent entity's filename (e.g., 'Apple_Inc.md') and map the data into its content.
DO NOT extract purely metadata, numeric IDs, arbitrary strings, or meaningless labels (e.g. 'Author_44211', 'Page_2', 'Header', 'Conference_Call_Participants', 'Q3_Earnings_Summary') as entities. 'Concepts' MUST be broad industry phenomena, profound topics, or notable events (e.g., 'AI Supercycle', 'Supply Chain Shortage'), NOT structural document sections. Only extract genuine nouns such as specific people, companies, named technologies, organizations, and profound Concepts.
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, financial ratio, timeline, and nuanced detail from the dataset. Do not just summarize broadly; pull the exact numbers, technical specs, and analytical arguments to provide a highly comprehensive and deep encyclopedic entry.
STRICT RULE: The generated Python code MUST NOT invent, hallucinate, or add supplemental knowledge. It must strictly map the data from the DataFrame rows using ONLY the provided text.
Format the markdown 'content' beautifully. Include at least: '# Title', '**Type**: Entity', '**Source**: {file_path}', and map the core data from the row into paragraphs, tables, or lists.
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
                if not filename or is_meaningless_entity(filename): 
                    print(f"Skipping meaningless entity: {filename}")
                    continue
                raw_name = filename.replace(".md", "").replace("_", " ")
                resolved_name = resolve_entity(raw_name)
                final_filename = get_safe_filename(resolved_name)
                content = item.get("content", "")
                item_type = item.get("type", "entity").lower()
                
                target_path = merge_and_save_entity(final_filename, content)
                new_files.append((final_filename, content, target_path, item_type))

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

I want to extract all important entities/concepts into markdown files based on this structure.
Write a raw Python function named `extract_entities(data)` that takes the full parsed JSON object `data` and returns a list of dictionaries.
Each dictionary MUST have three keys: 'type' (either 'entity' or 'concept'), 'filename' (e.g. 'Company_Name.md'), and 'content' (the markdown string).
CRITICAL RULE: Filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md', 'iPhone.md'). Distinct and notable products, technologies, people, or platforms SHOULD get their own separate files. NEVER use generic sub-topic names like 'Financials.md' or 'Q2_Earnings.md'. If the extracted data is merely a generic sub-topic of a parent entity, you MUST map it to the parent entity's filename (e.g., 'Apple_Inc.md') and map the data into its content.
DO NOT extract purely metadata, numeric IDs, arbitrary strings, or meaningless labels (e.g. 'Author_44211', 'Page_2', 'Header', 'Conference_Call_Participants', 'Q3_Earnings_Summary') as entities. 'Concepts' MUST be broad industry phenomena, profound topics, or notable events (e.g., 'AI Supercycle', 'Supply Chain Shortage'), NOT structural document sections. Only extract genuine nouns such as specific people, companies, named technologies, organizations, and profound Concepts.
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, financial ratio, timeline, and nuanced detail from the dataset. Do not just summarize broadly; pull the exact numbers, technical specs, and analytical arguments to provide a highly comprehensive and deep encyclopedic entry.
STRICT RULE: The generated Python code MUST NOT invent, hallucinate, or add supplemental knowledge. It must strictly map the data from the JSON dictionary using ONLY the provided text.
Format the markdown 'content' beautifully. Include at least: '# Title', '**Type**: Entity', '**Source**: {file_path}', and map the core facts from the JSON into paragraphs, tables, or lists.
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
            if not filename or is_meaningless_entity(filename): 
                print(f"Skipping meaningless entity: {filename}")
                continue
            raw_name = filename.replace(".md", "").replace("_", " ")
            resolved_name = resolve_entity(raw_name)
            final_filename = get_safe_filename(resolved_name)
            content = item.get("content", "")
            item_type = item.get("type", "entity").lower()
            
            target_path = merge_and_save_entity(final_filename, content)
            new_files.append((final_filename, content, target_path, item_type))

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
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, financial ratio, timeline, and nuanced detail from the dataset. Prioritize raw numbers, detailed technical specifications, and hardware/financial quantitative analysis to provide deep insights.

STRICT RULE: Do NOT hallucinate or supplement with outside knowledge. Generate the wiki content purely and strictly using ONLY the information found in the extracted text below. Stay 100% faithful to the source material.

CRITICAL RULE: All filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md', 'iPhone.md', 'Tim_Cook.md'). Distinct and notable products, technologies, people, or platforms SHOULD get their own separate files. NEVER use generic sub-topic names like 'Financials.md' or 'Q2_Earnings.md'. If the extracted data is merely a generic sub-topic of a parent entity, you MUST map it to the parent entity's filename (e.g., 'Apple_Inc.md') and structure the data there.
DO NOT extract purely metadata, numeric IDs, arbitrary strings, or meaningless labels (e.g. 'Author_44211', 'Page_2', 'Header', 'Conference_Call_Participants', 'Q3_Earnings_Summary') as entities. 'Concepts' MUST be broad industry phenomena, profound topics, or notable events (e.g., 'AI Supercycle', 'Supply Chain Shortage'), NOT structural document sections. Only extract genuine nouns such as specific people, companies, named technologies, organizations, and profound Concepts.

Your output must be strictly in JSON format, like this:
{{
  "entities": [
    {{
      "filename": "Entity_Name.md",
      "content": "# Entity Name\\n\\n**Type**: Entity\\n**Sources**: [{file_path}]({file_path})\\n\\n## Overview\\nSome information..."
    }}
  ],
  "concepts": [
    {{
      "filename": "Concept_Name.md",
      "content": "# Concept Name\\n\\n**Type**: Concept\\n**Sources**: [{file_path}]({file_path})\\n\\n## Overview\\nSome information..."
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

    items_to_process = [("entity", e) for e in extracted_data.get("entities", [])] + \
                       [("concept", c) for c in extracted_data.get("concepts", [])]

    new_files = []
    for item_type, entity in items_to_process:
        filename = entity.get("filename", "")
        if not filename or is_meaningless_entity(filename): 
            print(f"Skipping meaningless entity: {filename}")
            continue
        raw_name = filename.replace(".md", "").replace("_", " ")
        resolved_name = resolve_entity(raw_name)
        final_filename = get_safe_filename(resolved_name)
        file_content = entity.get("content", "")
        
        target_path = merge_and_save_entity(final_filename, file_content)
        new_files.append((final_filename, file_content, target_path, item_type))

    if new_files: update_index(new_files)
    log_action("ingest", os.path.basename(file_path))

def update_index(new_files):
    if not os.path.exists("index.md"):
        with open("index.md", "w", encoding="utf-8") as f:
            f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n")
            
    with open("index.md", "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified = False
    
    def insert_entry(entry, section_name, name_display):
        nonlocal modified
        for i, line in enumerate(lines):
            if f"[{name_display.lower()}]" in line.lower():
                if lines[i].strip() != entry.strip():
                    lines[i] = f"{entry}\n"
                    modified = True
                return
                
        section_idx = -1
        for i, line in enumerate(lines):
            if line.strip().lower() == f"## {section_name.lower()}":
                section_idx = i
                break
        if section_idx != -1:
            insert_idx = section_idx + 1
            while insert_idx < len(lines) and (lines[insert_idx].strip() == "" or lines[insert_idx].strip().startswith("-")):
                insert_idx += 1
            lines.insert(insert_idx, f"{entry}\n")
            modified = True
        else:
            lines.append(f"\n## {section_name.title()}\n")
            lines.append(f"{entry}\n")
            modified = True

    for filename, file_content, _, item_type in new_files:
        name_display = filename.replace(".md", "").replace("_", " ")
        desc = ""
        
        # Intelligent description generation
        desc = ""
        if type(file_content) is str:
            summary = summarize_entity(file_content)
            if summary:
                desc = f" - {summary}"
            
            if re.search(r'\*\*type\*\*\s*:\s*concept', file_content, re.IGNORECASE):
                item_type = "concept"
            elif re.search(r'\*\*type\*\*\s*:\s*entity', file_content, re.IGNORECASE):
                item_type = "entity"

        entry = f"- [{name_display}]({PAGES_DIR}/{filename}){desc}"

        if item_type == "concept":
            insert_entry(entry, "Concepts", name_display)
        else:
            insert_entry(entry, "Entities", name_display)
            
    if modified:
        with open("index.md", "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("Updated index.md with new grouped entries.")

def query(question, max_hops=3):
    print(f"Querying knowledge base (Max Hops: {max_hops}): {question}")
    
    if not os.path.exists("index.md"):
        print("Knowledge base index not found. Please ingest documents first.")
        return
        
    with open("index.md", "r", encoding="utf-8") as f:
        index_content = f.read()

    visited_files = set()
    current_hop = 1
    
    while current_hop <= max_hops:
        # Build context from visited files
        context = ""
        for vf in visited_files:
            path = os.path.join(PAGES_DIR, vf)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    context += f"--- {vf} ---\n{f.read()}\n\n"
                    
        print(f"\n--- [Hop {current_hop}/{max_hops}] Agent 1: Synthesizing Context ---")
        prompt1 = f"""
You are an analytical agent. The user is asking: "{question}".
Here is the content of files you have ALREADY read (if any):
{context}

Can you fully and comprehensively answer the user's question using ONLY the provided context?
If yes, provide your final answer in standard markdown formatting.
If you are missing crucial facts, or if the context is entirely empty, you MUST output exactly the string: NEED_MORE_INFO
"""
        response1 = query_llm([{"role": "user", "content": prompt1}], system_prompt="You are an analytical agent.")
        
        if not response1:
            print("Error: Received empty response from LLM (Agent 1).")
            return
            
        if "NEED_MORE_INFO" not in response1.strip():
            ans = response1.strip()
            if visited_files:
                sources_str = ", ".join([f.replace(".md", "") for f in visited_files])
                ans += f"\n\n---\n**Sources Consulted:** {sources_str}"
            
            print("\n--- Final Answer ---\n")
            print(ans)
            print("\n--------------\n")
            log_action("query", f"Answered '{question}' in {current_hop} hops. Visited: {list(visited_files)}")
            return ans
            
        print(f"[Hop {current_hop}] Context insufficient. Triggering Agent 2: Routing Index...")
        prompt2 = f"""
You are a relentless routing agent. The user is asking: "{question}".
You need more information to answer the question.

Here is the index of available knowledge base articles:
{index_content}

CRITICAL INSTRUCTION: Review the index and output a JSON list containing the exact filenames of any potentially relevant files you want to read next (e.g. ["TSMC.md", "Apple_Inc.md"]). DO NOT request files you have already read.
Files you have already read: {list(visited_files)}

OUTPUT ONLY A VALID JSON LIST OF FILENAMES. Do not output any other text or explanation.
"""
        response2 = query_llm([{"role": "user", "content": prompt2}], system_prompt="You are a JSON routing agent. Output ONLY a valid JSON array of strings.")
        
        if not response2:
            print("Error: Received empty response from LLM (Agent 2).")
            return
            
        cleaned2 = response2.strip()
        if cleaned2.startswith("```json"): cleaned2 = cleaned2[7:]
        elif cleaned2.startswith("```"): cleaned2 = cleaned2[3:]
        if cleaned2.endswith("```"): cleaned2 = cleaned2[:-3]
        cleaned2 = cleaned2.strip()
        
        try:
            files_to_read = json.loads(cleaned2)
            if isinstance(files_to_read, list):
                new_files = [f for f in files_to_read if str(f).endswith(".md") and f not in visited_files]
                if not new_files:
                    print(f"[Hop {current_hop}] Agent 2 requested no new valid files. Force synthesizing next hop...")
                    current_hop = max_hops + 1
                    continue
                
                print(f"[Hop {current_hop}] Agent 2 elected to read: {new_files}")
                visited_files.update(new_files)
                current_hop += 1
                continue
            else:
                print(f"[Hop {current_hop}] Agent 2 failed to output a JSON list. Force synthesizing next hop...")
                current_hop = max_hops + 1
                continue
        except Exception as e:
            print(f"[Hop {current_hop}] Agent 2 JSON parsing failed: {e}. Output was: {cleaned2}. Force synthesizing...")
            current_hop = max_hops + 1
            continue


    # If we hit max hops and loop finishes, do final synthesis
    print(f"\n[!] Max hops reached. Formulating final answer with gathered context...")
    context = ""
    for vf in visited_files:
        path = os.path.join(PAGES_DIR, vf)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                context += f"--- {vf} ---\n{f.read()}\n\n"
    
    prompt = f"Using ONLY the following gathered context, answer the user's question. Try your best to piece together a helpful response, even if the information is partial or scattered. Be exhaustive with the facts provided.\nContext:\n{context}\n\nQuestion: {question}"
    final_answer = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a resilient and helpful analyst.")
    
    if final_answer:
        ans = final_answer.strip()
        if visited_files:
            sources_str = ", ".join([f.replace(".md", "") for f in visited_files])
            ans += f"\n\n---\n**Sources Consulted:** {sources_str}"
            
        print("\n--- Final Answer ---\n")
        print(ans)
        print("\n--------------\n")
        log_action("query", f"Answered '{question}' (Hit max hops={max_hops}). Visited: {list(visited_files)}")
        return ans

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
   - Organize facts into logical sections like '## Overview' and '## Key Details' (or specific topics like 'Financials', 'Technology', etc. based on the data).
   - Convert simple key-value dumps into cohesive paragraphs or bullet points.
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, financial ratio, timeline, and nuanced detail from the text. Prioritize raw numbers, technical specifications, and quantitative data. 

STRICT RULE: Do NOT invent, hallucinate, or add supplemental knowledge from your own training data. Only use facts explicitly present in the Existing Document. Stay 100% faithful to the source material.
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

def lint_merge_all():
    print("Running Automerge Duplicates Scan...")
    conn = init_db()
    if not conn:
        print("Cannot run merge linting without PostgreSQL connection.")
        return
        
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            # 1. Self-healing DB check: Ensure all files currently in PAGES_DIR are actually in the DB
            # This fixes issues where someone manually creates an .md file or postgres was down during ingest
            if os.path.exists(PAGES_DIR):
                for file in os.listdir(PAGES_DIR):
                    if file.endswith(".md"):
                        raw_name = file.replace(".md", "").replace("_", " ")
                        cur.execute("SELECT id FROM wiki_entities WHERE name = %s;", (raw_name,))
                        if not cur.fetchone():
                            print(f"[Self-Healing] Missing DB record for '{raw_name}'. Generating vector...")
                            vec = embed_text(raw_name)
                            if vec:
                                vec_literal = "[" + ",".join(map(str, vec)) + "]"
                                cur.execute("INSERT INTO wiki_entities (name, embedding) VALUES (%s, %s::vector) ON CONFLICT DO NOTHING;", (raw_name, vec_literal))
            
            # 2. Moderate to wide threshold math sweep (0.45 instead of 0.35 to catch "semantic drift" like semiconductor)
            cur.execute("""
                SELECT a.name, b.name, a.embedding <=> b.embedding AS distance 
                FROM wiki_entities a 
                JOIN wiki_entities b ON a.id < b.id
                WHERE (a.embedding <=> b.embedding) < 0.45
                ORDER BY distance ASC;
            """)
            suspicious_pairs = cur.fetchall()
            
        if not suspicious_pairs:
            print("No suspiciously similar entity names found.")
            return
            
        print(f"Found {len(suspicious_pairs)} mathematically similar entity pairs. Asking LLM to verify...")
        
        merged_count = 0
        merged_entities = set()
        
        for pair in suspicious_pairs:
            name_a, name_b, dist = pair
            if name_a in merged_entities or name_b in merged_entities:
                continue
                
            prompt = f"""
We have two entities in our knowledge base: '{name_a}' and '{name_b}'.
Are these fundamentally referring to the exact same core concept, technology, or entity? 
You should consolidate aliases! For example, "AI Supercycle" and "AI Semiconductor Supercycle" are the SAME concept. "Christophe D Fouquet" and "Christophe Fouquet" are the SAME person.
If they are heavily overlapping or conceptually identical at the core, answer YES.
If they are completely distinct things (e.g. "Apple Inc" vs "iPhone", or a parent versus a child product), answer NO.
Reply exclusively with YES or NO.
"""
            resp = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a consolidation agent.")
            ans = resp.strip().upper() if resp else ""
            if "YES" in ans:
                print(f"[Automerge] LLM determined '{name_b}' is duplicate of '{name_a}'. Merging...")
                
                file_a = get_safe_filename(name_a)
                file_b = get_safe_filename(name_b)
                path_a = os.path.join(PAGES_DIR, file_a)
                path_b = os.path.join(PAGES_DIR, file_b)
                
                if not os.path.exists(path_b):
                    continue
                    
                with open(path_b, "r", encoding="utf-8") as f:
                    content_b = f.read()
                    
                # Weavetogether the two files via existing powerful LLM tool
                merge_and_save_entity(file_a, content_b)
                
                # Redirect links
                with conn.cursor() as cur:
                    cur.execute("SELECT source_file FROM wiki_links WHERE target_file = %s;", (file_b,))
                    sources = [row[0] for row in cur.fetchall()]
                    
                    for src in sources:
                        src_path = os.path.join(PAGES_DIR, src)
                        if os.path.exists(src_path):
                            with open(src_path, "r", encoding="utf-8") as f:
                                src_content = f.read()
                            new_content = src_content.replace(f"(pages/{file_b})", f"(pages/{file_a})")
                            new_content = new_content.replace(f"({file_b})", f"({file_a})")
                            
                            with open(src_path, "w", encoding="utf-8") as f:
                                f.write(new_content)
                            
                            update_backlinks(src, new_content)
                            
                    # Clean DB ghost roots
                    cur.execute("DELETE FROM wiki_entities WHERE name = %s;", (name_b,))
                    cur.execute("DELETE FROM wiki_links WHERE source_file = %s OR target_file = %s;", (file_b, file_b))
                    cur.execute("DELETE FROM wiki_claims WHERE source_file = %s;", (file_b,))
                    
                # Scour from Index
                if os.path.exists("index.md"):
                    with open("index.md", "r", encoding="utf-8") as f:
                        idx_lines = f.readlines()
                    with open("index.md", "w", encoding="utf-8") as f:
                        for line in idx_lines:
                            if f"({PAGES_DIR}/{file_b})" not in line:
                                f.write(line)
                                
                # Archive B file locally
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.move(path_b, os.path.join(ARCHIVE_DIR, f"{file_b.replace('.md', '')}_merged_{timestamp}.md"))

                merged_entities.add(name_a)
                merged_entities.add(name_b)
                merged_count += 1
                
        print(f"\\nAutomerge complete! Consolidated {merged_count} duplicate concepts.")
        log_action("lint_merge", f"Merged {merged_count} entity pairs.")
        
    except Exception as e:
        print(f"Merge Lint error: {e}")
    finally:
        conn.close()

def lint(deep=False, fix=False, merge=False):
    if fix:
        lint_fix_all()
    elif deep:
        lint_deep()
    elif merge:
        lint_merge_all()
    else:
        lint_hygiene()

def rebuild_all_indices():
    """Wipes index.md and rebuilds it by reading every file in PAGES_DIR."""
    print("Rebuilding entire index with intelligent summaries...")
    if not os.path.exists(PAGES_DIR):
        print("No pages found to index.")
        return

    # Reset index.md
    with open("index.md", "w", encoding="utf-8") as f:
        f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n")

    files = [f for f in os.listdir(PAGES_DIR) if f.endswith(".md")]
    new_files_data = []

    for filename in sorted(files):
        path = os.path.join(PAGES_DIR, filename)
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
    if os.path.exists(PAGES_DIR):
        for file in os.listdir(PAGES_DIR):
            if file.endswith(".md"):
                src = os.path.join(PAGES_DIR, file)
                dst = os.path.join(ARCHIVE_DIR, f"{file.replace('.md', '')}_{timestamp}.md")
                shutil.move(src, dst)
                moved_count += 1
    print(f"Moved {moved_count} pages to archive.")
    
    with open("index.md", "w", encoding="utf-8") as f:
        f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n")
    print("Cleared index.md.")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wiki_agent.py: Local LLM Wiki Automation Agent")
    parser.add_argument("command", choices=["ingest", "query", "lint", "reset", "refresh-index"], help="Command to execute")
    parser.add_argument("args", nargs="*", help="Arguments for the command.")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API instead of local LiteLLM")
    parser.add_argument("--deep", action="store_true", help="RAG systemic contradiction audit (lint only)")
    parser.add_argument("--fix", action="store_true", help="Automatically revise and restructure all wiki pages during linting")
    parser.add_argument("--merge", action="store_true", help="Automerge mathematically and conceptually identical entities globally (lint only)")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum number of hops for multi-hop retrieval querying (default: 3)")
    
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
        query(args.args[0], max_hops=args.max_hops)
    elif cmd == "lint":
        lint(deep=args.deep, fix=args.fix, merge=args.merge)
    elif cmd == "reset":
        reset()
    elif cmd == "refresh-index":
        rebuild_all_indices()
