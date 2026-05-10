import os
import json
import re
import shutil
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config
from .db import init_db, register_vector
from .llm import query_llm, embed_text, summarize_entity
from .utils import resolve_entity, get_safe_filename, log_action, get_existing_entities

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

def extract_report_date(file_path, content_snippet):
    print(f"Extracting chronological timeline date for '{file_path}'...")
    prompt = f"""
I am ingesting a document or report into a knowledge base. The timeline of this document is critical for chronological sorting.
Filename: {file_path}
Document Content:
{content_snippet if content_snippet else ""}

Please extract the exact "Report Date" or "Occurrence Date" this file represents. 
Look at the filename first (e.g., "03-15-26" means March 15, 2026). If the filename does not contain a date, look at the document content.
If no date is found, reply with "Unknown Date".
Respond ONLY with the Date in 'YYYY-MM-DD' format (if a specific day) or 'YYYY-MM' (if just a month) or whatever specific date text you found. Do not provide any conversational text.
"""
    response = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a strict chronological parsing agent. Output only the date.")
    if response:
        return response.strip()
    return "Unknown Date"


def extract_and_embed_claims(filename, content, division):
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
                cur.execute("DELETE FROM wiki_claims WHERE source_file = %s AND division = %s;", (filename, division))
                for claim in claims:
                    vec = embed_text(claim)
                    if vec:
                        vec_literal = "[" + ",".join(map(str, vec)) + "]"
                        cur.execute("INSERT INTO wiki_claims (division, source_file, claim_text, embedding) VALUES (%s, %s, %s, %s::vector);", (division, filename, claim, vec_literal))
            print(f"Embedded {len(claims)} fact claims for {filename} into vector storage.")
    except Exception as e:
        print(f"Failed to parse claims JSON for {filename}: {e}")
    finally:
        conn.close()

def update_backlinks(filename, content, division):
    """Parses new content for edges, upserts to DB, and rewrites target files locally."""
    conn = init_db()
    if not conn: return
    
    # Relax regex to match both (pages/File.md) and (File.md) safely
    link_pattern = re.compile(r'\[.*?\]\((?:pages/)?(.*?\.md)\)')
    targets = set(link_pattern.findall(content))
    
    try:
        with conn.cursor() as cur:
            # 1. Store old targets before deletion so we can refresh them (to remove stale links)
            cur.execute("SELECT target_file FROM wiki_links WHERE source_file = %s AND division = %s;", (filename, division))
            old_targets = set(row[0] for row in cur.fetchall())
            
            # 2. Update Knowledge Graph database
            cur.execute("DELETE FROM wiki_links WHERE source_file = %s AND division = %s;", (filename, division))
            if targets:
                for t in targets:
                    cur.execute("INSERT INTO wiki_links (division, source_file, target_file) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;", (division, filename, t))
                    
            # 3. Refresh Backlinks section in markdown for ALL affected pages
            # This includes new targets, removed targets, AND the current ingested file itself.
            files_to_refresh = targets.union(old_targets)
            files_to_refresh.add(filename)
            
            for f_name in files_to_refresh:
                cur.execute("SELECT source_file FROM wiki_links WHERE target_file = %s AND division = %s;", (f_name, division))
                backlink_sources = [row[0] for row in cur.fetchall()]
                
                target_path = os.path.join(config.get_pages_dir(division), f_name)
                with config.file_write_lock:
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

def merge_and_save_entity(filename, new_content, division, cascade=True, report_date=None):
    target_path = os.path.join(config.get_pages_dir(division), filename)
    taxonomy = get_existing_entities(division)
    taxonomy_str = ", ".join(taxonomy)

    final_content_to_save = None

    if os.path.exists(target_path):
        with config.file_write_lock:
            # Archive old version
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{filename.replace('.md', '')}_{timestamp}.md"
            shutil.copy2(target_path, os.path.join(config.get_archive_dir(division), archive_name))
            
            # Read old content
            with open(target_path, "r", encoding="utf-8") as f:
                old_content = f.read()
            
        print(f"Consolidating existing entity: {filename}...")
        
        date_instruction = f"The New Data is associated with the date/timeline: {report_date}. " if report_date and report_date != "Unknown Date" else ""
        prompt = f"""
You are a Wikipedia editor. An entity document already exists, but new information has been ingested.
Your task is to merge the New Data into the Existing Document intelligently.
Keep all historical facts, seamlessly weave in the new facts, and format beautifully in Markdown. 
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, timeline, and nuanced detail from the text. Prioritize raw numbers, financial metrics, technical specifications, and quantitative data. 

TIMELINE/CHRONOLOGICAL RULE: {date_instruction}Maintain a `## Timeline` or `## Chronological History` section. When adding the New Data, DO NOT just append it to the end. You MUST insert the new facts into the proper chronological order within the timeline section, as reports may be ingested out of sequence. Ensure dates are explicitly stated for these new facts.

CITATION RULE: Ensure all facts in the text are followed by an academic inline citation (e.g., [1], [2]). At the bottom of the document, maintain a `## References` section mapping these numbers to their original sources. The Existing Document and New Data will contain their source links. Consolidate them intelligently.

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
            with config.file_write_lock:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(final_content_to_save)
            print(f"Merged and Updated {target_path}")
    else:
        date_instruction = f"The Raw Data Content is associated with the date/timeline: {report_date}. " if report_date and report_date != "Unknown Date" else ""
        prompt = f"""
You are creating a new Wikipedia-style entity document based on the provided raw data.
Please rewrite and structure the raw information into a rich, comprehensive, and beautiful markdown page.
Organize the facts clearly into logical sections such as '## Overview' and '## Key Details' (or specific topics like 'Financials', 'Technology', etc. based on the data).
Synthesize the facts into cohesive paragraphs or bullet points. 

TIMELINE/CHRONOLOGICAL RULE: {date_instruction}If there is chronological data or a specific report date, explicitly create a `## Timeline` or `## Chronological History` section and format the facts with their associated dates.

CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, timeline, and nuanced detail from the text. Prioritize raw numbers, financial metrics, hardware specifications, and any quantitative analysis. 

CITATION RULE: Ensure all facts in the text are followed by an academic inline citation (e.g., [1]). At the bottom of the document, create a `## References` section mapping this number to the source link provided in the Raw Data Content.

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
        with config.file_write_lock:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(final_content_to_save)
        print(f"Created {target_path}")
        
    if final_content_to_save:
        extract_and_embed_claims(filename, final_content_to_save, division)
        update_backlinks(filename, final_content_to_save, division)
        
        if cascade:
            link_pattern = re.compile(r'\[.*?\]\((?:pages/)?(.*?\.md)\)')
            targets = set(link_pattern.findall(new_content))
            if targets:
                sentences = new_content.replace('\n', ' ').split('. ')
                
                def cascade_task(t):
                    if t == filename: return
                    target_path_check = os.path.join(config.get_pages_dir(division), t)
                    if os.path.exists(target_path_check):
                        mention_sentences = [s for s in sentences if f"({t})" in s or f"(pages/{t})" in s]
                        if mention_sentences:
                            context_injection = ". ".join(mention_sentences) + "."
                            print(f"[Cascade] Updating {t} with contextual link from {filename}...")
                            merge_and_save_entity(t, f"New context referencing this topic from {filename}: {context_injection}", division, cascade=False)
                            
                with ThreadPoolExecutor(max_workers=5) as executor:
                    executor.map(cascade_task, targets)
                            
    return target_path

def update_index(new_files, division):
    with config.file_write_lock:
        index_path = config.get_index_path(division)
        if not os.path.exists(index_path):
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("# LLM Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n")
                
        with open(index_path, "r", encoding="utf-8") as f:
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
            lines.insert(section_idx + 1, entry + "\n")
            modified = True
            
    for filename, content, target_path, item_type in new_files:
        name_display = filename.replace(".md", "").replace("_", " ")
        summary = ""
        desc = ""
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            summary = summarize_entity(file_content)
            if summary:
                desc = f" - {summary}"
            
            if re.search(r'\*\*type\*\*\s*:\s*concept', file_content, re.IGNORECASE):
                item_type = "concept"
            elif re.search(r'\*\*type\*\*\s*:\s*entity', file_content, re.IGNORECASE):
                item_type = "entity"

        entry = f"- [{name_display}](namespaces/{division}/pages/{filename}){desc}"

        if item_type == "concept":
            insert_entry(entry, "Concepts", name_display)
        else:
            insert_entry(entry, "Entities", name_display)
            
    if modified:
        with config.file_write_lock:
            with open(index_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        print("Updated index.md with new grouped entries.")

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

def handle_excel_ingest(file_path, division):
    if not pd:
        print("Error: pandas and openpyxl are missing. Run 'pip install pandas openpyxl'.")
        return
    print(f"Analyzing schema for large Excel file '{file_path}' using {config.MODEL_NAME}...")
    dfs = pd.read_excel(file_path, sheet_name=None)
    
    for sheet_name, df in dfs.items():
        if df.empty: continue
        print(f"Asking LLM to generate Python parser for sheet: {sheet_name} (Sampling top 500 rows)...")
        schema_csv = df.head(500).to_csv(index=False)
        
        report_date = extract_report_date(file_path, schema_csv)
        prompt = f"""
I have a pandas DataFrame containing a knowledge base dataset. Here are the first 500 rows in CSV format:
{schema_csv}

I want to iteratively extract all important entities/concepts into markdown files based on that schema.
Write a raw Python function named `extract_entities(df)` that iterates through the DataFrame `df` and returns a list of dictionaries.
Each dictionary MUST have three keys: 'type' (either 'entity' or 'concept'), 'filename' (e.g. 'Company_Name.md'), and 'content' (the markdown string).
CRITICAL RULE: Filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md', 'iPhone.md'). Distinct and notable products, technologies, people, or platforms SHOULD get their own separate files. NEVER use generic sub-topic names like 'Financials.md' or 'Q2_Earnings.md'. If the extracted data is merely a generic sub-topic of a parent entity, you MUST map it to the parent entity's filename (e.g., 'Apple_Inc.md') and map the data into its content.
DO NOT extract purely metadata, numeric IDs, arbitrary strings, or meaningless labels (e.g. 'Author_44211', 'Page_2', 'Header', 'Conference_Call_Participants', 'Q3_Earnings_Summary') as entities. 'Concepts' MUST be broad industry phenomena, profound topics, or notable events (e.g., 'AI Supercycle', 'Supply Chain Shortage'), NOT structural document sections. Only extract genuine nouns such as specific people, companies, named technologies, organizations, and profound Concepts.
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, financial ratio, timeline, and nuanced detail from the dataset. Do not just summarize broadly; pull the exact numbers, technical specs, and analytical arguments to provide a highly comprehensive and deep encyclopedic entry.
TIMELINE RULE: The dataset is associated with the date/timeline: {report_date}. Ensure you extract any chronological information, and explicitly prefix facts with their dates in the markdown content to preserve timeline accuracy.
CITATION RULE: Append an academic inline citation (e.g., [1]) to every fact you extract. At the bottom of the markdown content, create a `## References` section that maps [1] to the source file: [{file_path}]({file_path}).
STRICT RULE: The generated Python code MUST NOT invent, hallucinate, or add supplemental knowledge. It must strictly map the data from the DataFrame rows using ONLY the provided text.
Format the markdown 'content' beautifully. Include at least: '# Title', '**Type**: Entity', and map the core data from the row into paragraphs, tables, or lists.
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
                if not filename: 
                    continue
                raw_name = filename.replace(".md", "").replace("_", " ")
                resolved_name = resolve_entity(raw_name, division)
                final_filename = get_safe_filename(resolved_name)
                content = item.get("content", "")
                item_type = item.get("type", "entity").lower()
                
                target_path = merge_and_save_entity(final_filename, content, division, report_date=report_date)
                new_files.append((final_filename, content, target_path, item_type))

            if new_files: update_index(new_files, division)

        except Exception as e:
            print(f"Failed to execute LLM-written extraction code for sheet '{sheet_name}':\n{e}\n\nGenerated Code:\n{code}")
    log_action("ingest_hybrid_xlsx", os.path.basename(file_path))

def handle_json_ingest(file_path, division):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file '{file_path}': {e}")
        return

    print(f"Analyzing schema for JSON file '{file_path}' using {config.MODEL_NAME}...")
    
    # Sample the JSON structure to avoid context limits
    def sample_data(d, max_items=10):
        if isinstance(d, list):
            return [sample_data(item, max_items=2) for item in d[:max_items]]
        elif isinstance(d, dict):
            return {k: sample_data(v, max_items=2) for k, v in list(d.items())[:max_items]}
        return d

    sampled_json = json.dumps(sample_data(data, max_items=10), indent=2)
    
    report_date = extract_report_date(file_path, sampled_json)
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
TIMELINE RULE: The dataset is associated with the date/timeline: {report_date}. Ensure you extract any chronological information, and explicitly prefix facts with their dates in the markdown content to preserve timeline accuracy.
CITATION RULE: Append an academic inline citation (e.g., [1]) to every fact you extract. At the bottom of the markdown content, create a `## References` section that maps [1] to the source file: [{file_path}]({file_path}).
STRICT RULE: The generated Python code MUST NOT invent, hallucinate, or add supplemental knowledge. It must strictly map the data from the JSON dictionary using ONLY the provided text.
Format the markdown 'content' beautifully. Include at least: '# Title', '**Type**: Entity', and map the core facts from the JSON into paragraphs, tables, or lists.
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
            if not filename: 
                continue
            raw_name = filename.replace(".md", "").replace("_", " ")
            resolved_name = resolve_entity(raw_name, division)
            final_filename = get_safe_filename(resolved_name)
            content = item.get("content", "")
            item_type = item.get("type", "entity").lower()
            
            target_path = merge_and_save_entity(final_filename, content, division, report_date=report_date)
            new_files.append((final_filename, content, target_path, item_type))

        if new_files: update_index(new_files, division)

    except Exception as e:
        print(f"Failed to execute LLM-written extraction code for JSON:\n{e}\n\nGenerated Code:\n{code}")
    log_action("ingest_hybrid_json", os.path.basename(file_path))

def ingest(file_path, division):
    config.init_directories(division)
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    _, ext = os.path.splitext(file_path.lower())
    if ext == ".xlsx":
        handle_excel_ingest(file_path, division)
        return
    elif ext == ".json":
        handle_json_ingest(file_path, division)
        return

    print(f"Extracting text from {file_path}...")
    content = extract_text_from_file(file_path)
    if content.startswith("Error"):
        print(content)
        return

    report_date = extract_report_date(file_path, content)
    print(f"Ingesting into semantic entities using {config.MODEL_NAME}...")
    prompt = f"""
I have extracted text from a document. I want to identify the key entities (e.g., companies, people, technologies) and concepts.
For each important entity or concept, provide a summary formatted as a Markdown file. 
CRITICAL RULE: Be extremely exhaustive and dense! Extract every single important fact, metric, financial ratio, timeline, and nuanced detail from the dataset. Prioritize raw numbers, detailed technical specifications, and hardware/financial quantitative analysis to provide deep insights.

TIMELINE RULE: The text is associated with the date/timeline: {report_date}. Ensure you extract any chronological information, and explicitly prefix facts with their dates in the markdown content to preserve timeline accuracy.

STRICT RULE: Do NOT hallucinate or supplement with outside knowledge. Generate the wiki content purely and strictly using ONLY the information found in the extracted text below. Stay 100% faithful to the source material.

CITATION RULE: Append an academic inline citation (e.g., [1]) to every fact you extract. At the bottom of the markdown content, create a `## References` section that maps [1] to the source file: [{file_path}]({file_path}).

CRITICAL RULE: All filenames MUST represent globally unique Root Entities (e.g., 'Apple_Inc.md', 'iPhone.md', 'Tim_Cook.md'). Distinct and notable products, technologies, people, or platforms SHOULD get their own separate files. NEVER use generic sub-topic names like 'Financials.md' or 'Q2_Earnings.md'. If the extracted data is merely a generic sub-topic of a parent entity, you MUST map it to the parent entity's filename (e.g., 'Apple_Inc.md') and structure the data there.
DO NOT extract purely metadata, numeric IDs, arbitrary strings, or meaningless labels (e.g. 'Author_44211', 'Page_2', 'Header', 'Conference_Call_Participants', 'Q3_Earnings_Summary') as entities. 'Concepts' MUST be broad industry phenomena, profound topics, or notable events (e.g., 'AI Supercycle', 'Supply Chain Shortage'), NOT structural document sections. Only extract genuine nouns such as specific people, companies, named technologies, organizations, and profound Concepts.

Your output must be strictly in JSON format, like this:
{{
  "entities": [
    {{
      "filename": "Entity_Name.md",
      "content": "# Entity Name\n\n**Type**: Entity\n\n## Overview\nSome information [1].\n\n## References\n[1] [{file_path}]({file_path})"
    }}
  ],
  "concepts": [
    {{
      "filename": "Concept_Name.md",
      "content": "# Concept Name\n\n**Type**: Concept\n\n## Overview\nSome information [1].\n\n## References\n[1] [{file_path}]({file_path})"
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
        if not filename: 
            continue
        raw_name = filename.replace(".md", "").replace("_", " ")
        resolved_name = resolve_entity(raw_name, division)
        final_filename = get_safe_filename(resolved_name)
        file_content = entity.get("content", "")
        
        target_path = merge_and_save_entity(final_filename, file_content, division, report_date=report_date)
        new_files.append((final_filename, file_content, target_path, item_type))

    if new_files: update_index(new_files, division)
    log_action("ingest", os.path.basename(file_path))
