import os
import re
import time
import datetime
import shutil

from . import config
from .db import init_db, register_vector
from .llm import query_llm, embed_text
from .utils import get_existing_entities, get_safe_filename, log_action
from .ingest import update_backlinks, merge_and_save_entity

def lint_hygiene(division):
    print("Running Daily Hygiene (Targeted Sub-Graph)...")
    now = time.time()
    one_day_ago = now - 86400
    
    modified_files = []
    if os.path.exists(config.get_pages_dir(division)):
        for file in os.listdir(config.get_pages_dir(division)):
            if file.endswith(".md"):
                path = os.path.join(config.get_pages_dir(division), file)
                if os.path.getmtime(path) > one_day_ago:
                    modified_files.append(file)
                
    if not modified_files:
        print("No files modified in the last 24 hours. Hygiene check passed blindly!")
        return

    subgraph_files = set(modified_files)
    link_pattern = re.compile(r'\[.*?\]\((.*?\.md)\)')
    
    for mod_file in modified_files:
        path = os.path.join(config.get_pages_dir(division), mod_file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            neighbors = link_pattern.findall(content)
            for neighbor in neighbors:
                if neighbor.startswith(config.get_pages_dir(division) + "/"):
                    neighbor = neighbor[len(config.get_pages_dir(division))+1:]
                if neighbor in os.listdir(config.get_pages_dir(division)):
                    subgraph_files.add(neighbor)
                    
    print(f"Identified {len(subgraph_files)} interconnected nodes based on recent edits.")
    
    markdown_content = ""
    for file in subgraph_files:
        path = os.path.join(config.get_pages_dir(division), file)
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

def lint_deep(division):
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
                 WHERE a.division = %s AND b.division = %s AND a.source_file != b.source_file AND (a.embedding <=> b.embedding) < 0.15
                 ORDER BY distance ASC
                 LIMIT 10;
             """, (division, division))
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

def lint_fix_all(division):
    print("Running Auto-Fix & Restructure on all pages...")
    taxonomy = get_existing_entities(division)
    taxonomy_str = ", ".join(taxonomy)
    fixed_count = 0
    if not os.path.exists(config.get_pages_dir(division)):
        print("No pages to fix.")
        return

    for file in os.listdir(config.get_pages_dir(division)):
        if file.endswith(".md"):
            path = os.path.join(config.get_pages_dir(division), file)
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
                    with config.file_write_lock:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(cleaned)
                    fixed_count += 1
                    
    print(f"\nAuto-fixed and upgraded {fixed_count} pages!")
    log_action("lint_fix", f"Restructured and grammar-checked {fixed_count} pages.")

def lint_merge_all(division):
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
            if os.path.exists(config.get_pages_dir(division)):
                for file in os.listdir(config.get_pages_dir(division)):
                    if file.endswith(".md"):
                        raw_name = file.replace(".md", "").replace("_", " ")
                        cur.execute("SELECT id FROM wiki_entities WHERE division = %s AND name = %s;", (division, raw_name))
                        if not cur.fetchone():
                            print(f"[Self-Healing] Missing DB record for '{raw_name}'. Generating vector...")
                            vec = embed_text(raw_name)
                            if vec:
                                vec_literal = "[" + ",".join(map(str, vec)) + "]"
                                cur.execute("INSERT INTO wiki_entities (division, name, embedding) VALUES (%s, %s, %s::vector) ON CONFLICT DO NOTHING;", (division, raw_name, vec_literal))
            
            # 2. Moderate to wide threshold math sweep (0.45 instead of 0.35 to catch "semantic drift" like semiconductor)
            cur.execute("""
                SELECT a.name, b.name, a.embedding <=> b.embedding AS distance 
                FROM wiki_entities a 
                JOIN wiki_entities b ON a.id < b.id
                WHERE a.division = %s AND b.division = %s AND (a.embedding <=> b.embedding) < 0.45
                ORDER BY distance ASC;
            """, (division, division))
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
                path_a = os.path.join(config.get_pages_dir(division), file_a)
                path_b = os.path.join(config.get_pages_dir(division), file_b)
                
                if not os.path.exists(path_b):
                    continue
                    
                with config.file_write_lock:
                    with open(path_b, "r", encoding="utf-8") as f:
                        content_b = f.read()
                    
                # Weavetogether the two files via existing powerful LLM tool
                merge_and_save_entity(file_a, content_b, division)
                
                # Redirect links
                with conn.cursor() as cur:
                    cur.execute("SELECT source_file FROM wiki_links WHERE division = %s AND target_file = %s;", (division, file_b))
                    sources = [row[0] for row in cur.fetchall()]
                    
                    for src in sources:
                        src_path = os.path.join(config.get_pages_dir(division), src)
                        if os.path.exists(src_path):
                            with config.file_write_lock:
                                with open(src_path, "r", encoding="utf-8") as f:
                                    src_content = f.read()
                                new_content = src_content.replace(f"(pages/{file_b})", f"(pages/{file_a})")
                                new_content = new_content.replace(f"({file_b})", f"({file_a})")
                                
                                with open(src_path, "w", encoding="utf-8") as f:
                                    f.write(new_content)
                            
                            update_backlinks(src, new_content, division)
                            
                    # Clean DB ghost roots
                    cur.execute("DELETE FROM wiki_entities WHERE division = %s AND name = %s;", (name_b,))
                    cur.execute("DELETE FROM wiki_links WHERE division = %s AND (source_file = %s OR target_file = %s);", (division, file_b, file_b))
                    cur.execute("DELETE FROM wiki_claims WHERE division = %s AND source_file = %s;", (division, file_b))
                    
                # Scour from Index
                with config.file_write_lock:
                    if os.path.exists(config.get_index_path(division)):
                        with open(config.get_index_path(division), "r", encoding="utf-8") as f:
                            idx_lines = f.readlines()
                        with open(config.get_index_path(division), "w", encoding="utf-8") as f:
                            for line in idx_lines:
                                if f"({config.get_pages_dir(division)}/{file_b})" not in line:
                                    f.write(line)
                                    
                # Archive B file locally
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.move(path_b, os.path.join(config.get_archive_dir(division), f"{file_b.replace('.md', '')}_merged_{timestamp}.md"))

                merged_entities.add(name_a)
                merged_entities.add(name_b)
                merged_count += 1
                
        print(f"\nAutomerge complete! Consolidated {merged_count} duplicate concepts.")
        log_action("lint_merge", f"Merged {merged_count} entity pairs.")
        
    except Exception as e:
        print(f"Merge Lint error: {e}")
    finally:
        conn.close()

def lint(deep=False, fix=False, merge=False, division=None):
    if fix:
        lint_fix_all(division)
    elif deep:
        lint_deep(division)
    elif merge:
        lint_merge_all(division)
    else:
        lint_hygiene(division)
