import os
import json
from concurrent.futures import ThreadPoolExecutor
from . import config
from .llm import query_llm
from .utils import log_action

def query(question, max_hops=3):
    print(f"Querying knowledge base (Max Hops: {max_hops}): {question}")
    
    if not os.path.exists("index.md"):
        print("Knowledge base index not found. Please ingest documents first.")
        return
        
    with open("index.md", "r", encoding="utf-8") as f:
        index_content = f.read()

    visited_files = set()
    extracted_contexts = {}
    current_hop = 1
    
    while current_hop <= max_hops:
        # Build context from extracted_contexts
        context = ""
        for vf in visited_files:
            if vf in extracted_contexts:
                context += f"--- {vf} ---\n{extracted_contexts[vf]}\n\n"
                    
        print(f"\n--- [Hop {current_hop}/{max_hops}] Agent 1: Synthesizing Context ---")
        prompt1 = f"""
            You are a highly skeptical analytical agent. The user is asking: "{question}".
            Here is the content of files you have ALREADY read (if any):
            {context}

            Can you fully and comprehensively answer the user's question using ONLY the provided context?
            CRITICAL INSTRUCTION: Do NOT be overconfident. Review the documents provided. Pay close attention to any Wiki links `[link](...)` or `## Backlinks` mentioned inside the text. If the text mentions a crucial related concept, product variation, competitor, or timeline event that seems highly relevant to the question, but you DO NOT have the full text for it in your context, you MUST output exactly the string: NEED_MORE_INFO to trigger another search round.
            If you are missing crucial facts, or if the context is entirely empty, you MUST output exactly the string: NEED_MORE_INFO

            If you are absolutely certain you have ALL the necessary nuances and related information to provide a comprehensive answer, provide your final answer in standard markdown formatting.
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
        
        index_lines = index_content.strip().split('\n')
        chunk_size = 200
        index_chunks = ["\n".join(index_lines[i:i + chunk_size]) for i in range(0, len(index_lines), chunk_size)]
        
        def route_chunk(chunk):
            prompt2 = f"""
                You are a relentless routing agent. The user is asking: "{question}".
                You need more information to answer the question comprehensively.

                Here is a chunk of the index of available knowledge base articles:
                {chunk}

                CRITICAL INSTRUCTION: Review this chunk of the index and output a JSON list containing the exact filenames of any potentially relevant files you want to read next. 
                Be extremely exhaustive and aggressive! If a user asks about a specific technology, company, or concept, you MUST also select related variations, successors, predecessors, or competitors found in this index chunk. Do not just pick one file if multiple related variations exist (e.g., if asked about "Product A", also select "Product A Pro", "Product A Max", etc., if they exist in the index).
                DO NOT request files you have already read.
                Files you have already read: {list(visited_files)}

                OUTPUT ONLY A VALID JSON LIST OF FILENAMES (e.g. ["File1.md", "File2.md"]). Do not output any other text or explanation.
                """
            resp = query_llm([{"role": "user", "content": prompt2}], system_prompt="You are a JSON routing agent. Output ONLY a valid JSON array of strings.")
            if not resp: return []
            
            cleaned2 = resp.strip()
            if cleaned2.startswith("```json"): cleaned2 = cleaned2[7:]
            elif cleaned2.startswith("```"): cleaned2 = cleaned2[3:]
            if cleaned2.endswith("```"): cleaned2 = cleaned2[:-3]
            
            try:
                files_to_read = json.loads(cleaned2.strip())
                if isinstance(files_to_read, list):
                    return [f for f in files_to_read if str(f).endswith(".md") and f not in visited_files]
            except Exception as e:
                pass
            return []

        print(f"[Hop {current_hop}] Index split into {len(index_chunks)} chunks. Running parallel Map-Reduce routing...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            chunk_results = executor.map(route_chunk, index_chunks)
            
        new_files = []
        for res_list in chunk_results:
            for f in res_list:
                if f not in new_files:
                    new_files.append(f)
                    
        if not new_files:
            print(f"[Hop {current_hop}] Agent 2 requested no new valid files across all chunks. Search exhausted early.")
            break
                
        print(f"[Hop {current_hop}] Agent 2 elected to read: {new_files}")
        
        def extract_relevant_info(vf):
            path = os.path.join(config.PAGES_DIR, vf)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                ext_prompt = f"Document '{vf}':\n{file_content}\n\nUser Question: '{question}'\nExtract ALL facts, metrics, and quotes relevant to answering the user's question. Be dense and exhaustive. If there is absolutely nothing relevant in this document, reply exactly with 'NO_RELEVANT_INFO'."
                resp = query_llm([{"role": "user", "content": ext_prompt}], system_prompt="You are a strict data extraction assistant.")
                if resp and 'NO_RELEVANT_INFO' not in resp.strip().upper():
                    return vf, resp.strip()
            return vf, ""
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(extract_relevant_info, new_files)
            for vf, ext_content in results:
                if ext_content:
                    extracted_contexts[vf] = ext_content
                    
        visited_files.update(new_files)
        current_hop += 1
        continue



    # If we hit max hops or loop breaks, do final synthesis
    if current_hop > max_hops:
        print(f"\n[!] Max hops reached. Formulating final answer with gathered context...")
    else:
        print(f"\n[!] Search exhausted early (stopped at Hop {current_hop}). Formulating final answer with gathered context...")
    context = ""
    for vf in visited_files:
        if vf in extracted_contexts:
            context += f"--- {vf} ---\n{extracted_contexts[vf]}\n\n"
    
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
