import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from . import config
from .llm import query_llm
from .utils import log_action

def query(question, divisions, max_hops=3, st_placeholder=None):
    output_logs = []
    def out(msg):
        print(msg)
        output_logs.append(str(msg))
        if st_placeholder:
            st_placeholder.code("\n".join(output_logs[-10:]), language="text")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    out(f"Querying knowledge base (Max Hops: {max_hops}): {question}")
    
    index_content = ""
    for div in divisions:
        index_path = config.get_index_path(div)
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_content += f"--- Index for {div} ---\n{f.read()}\n\n"
                
    if not index_content.strip():
        out(f"No knowledge base indices found for the authorized divisions: {divisions}")
        return

    visited_files = set()
    extracted_contexts = {}
    current_hop = 1
    
    while current_hop <= max_hops:
        # Build context from extracted_contexts
        context = ""
        source_mapping = {}
        for idx, vf in enumerate(list(visited_files), 1):
            source_mapping[idx] = vf
            if vf in extracted_contexts:
                context += f"--- Source [{idx}]: {vf} ---\n{extracted_contexts[vf]}\n\n"
                    
        out(f"\n--- [Hop {current_hop}/{max_hops}] Agent 1: Synthesizing Context ---")
        prompt1 = f"""
            You are a highly skeptical analytical agent. The user is asking: "{question}".
            Today's Date is: {today}.
            Here is the content of files you have ALREADY read (if any):
            {context}

            Can you fully and comprehensively answer the user's question using ONLY the provided context?
            CRITICAL INSTRUCTION: Do NOT be overconfident. Review the documents provided. Pay close attention to any Wiki links `[link](...)` or `## Backlinks` mentioned inside the text. If the text mentions a crucial related concept, product variation, competitor, or timeline event that seems highly relevant to the question, but you DO NOT have the full text for it in your context, you MUST output exactly the string: NEED_MORE_INFO to trigger another search round.
            If you are missing crucial facts, or if the context is entirely empty, you MUST output exactly the string: NEED_MORE_INFO

            If you are absolutely certain you have ALL the necessary nuances and related information to provide a comprehensive answer, provide your final answer in standard markdown formatting.
            CRITICAL INSTRUCTION: You MUST use academic inline citations (e.g., [1], [2]) when stating facts, corresponding to the Source IDs provided in the context.
            CRITICAL INSTRUCTION: Your final answer MUST be written in the exact same language as the user's original question.
            """
        response1 = query_llm([{"role": "user", "content": prompt1}], system_prompt="You are an analytical agent.")
        
        if not response1:
            out("Error: Received empty response from LLM (Agent 1).")
            return
            
        if "NEED_MORE_INFO" not in response1.strip():
            ans = response1.strip()
            if visited_files:
                ans += "\n\n---\n**Sources Consulted:**\n"
                for idx, vf in source_mapping.items():
                    display_name = vf.split("/")[-1].replace(".md", "")
                    ans += f"[{idx}] [{display_name}]({vf})\n"
            
            out("\n--- Final Answer ---\n")
            out(ans)
            out("\n--------------\n")
            log_action("query", f"Answered '{question}' in {current_hop} hops. Visited: {list(visited_files)}")
            return ans
            
        out(f"[Hop {current_hop}] Context insufficient. Triggering Agent 2: Routing Index...")
        
        index_lines = index_content.strip().split('\n')
        chunk_size = 200
        index_chunks = ["\n".join(index_lines[i:i + chunk_size]) for i in range(0, len(index_lines), chunk_size)]
        
        def route_chunk(chunk):
            prompt2 = f"""
                You are a relentless routing agent. The user is asking: "{question}".
                You need more information to answer the question comprehensively.

                Here is a chunk of the index of available knowledge base articles:
                {chunk}

                CRITICAL INSTRUCTION: Review this chunk of the index and output a JSON list containing the exact filepaths (the part inside the parentheses) of any potentially relevant files you want to read next. 
                If a user asks about a specific technology, company, or concept, you MUST also select related variations, successors, predecessors, or competitors found in this index chunk. Do not just pick one file if multiple related variations exist.
                DO NOT request files you have already read.
                Files you have already read: {list(visited_files)}

                OUTPUT ONLY A VALID JSON LIST OF FILEPATHS (e.g. ["namespaces/PPD/pages/File1.md", "namespaces/ORP/pages/File2.md"]). Do not output any other text or explanation.
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
                    return [f for f in files_to_read if "namespaces/" in str(f) and str(f).endswith(".md") and f not in visited_files]
            except Exception as e:
                pass
            return []

        out(f"[Hop {current_hop}] Index split into {len(index_chunks)} chunks. Running parallel Map-Reduce routing...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            chunk_results = executor.map(route_chunk, index_chunks)
            
        new_files = []
        for res_list in chunk_results:
            for f in res_list:
                if f not in new_files:
                    new_files.append(f)
                    
        if not new_files:
            out(f"[Hop {current_hop}] Agent 2 requested no new valid files across all chunks. Search exhausted early.")
            break
                
        out(f"[Hop {current_hop}] Agent 2 elected to read: {new_files}")
        
        def extract_relevant_info(vf):
            path = vf
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
        out(f"\n[!] Max hops reached. Formulating final answer with gathered context...")
    else:
        out(f"\n[!] Search exhausted early (stopped at Hop {current_hop}). Formulating final answer with gathered context...")
    context = ""
    source_mapping = {}
    for idx, vf in enumerate(list(visited_files), 1):
        source_mapping[idx] = vf
        if vf in extracted_contexts:
            context += f"--- Source [{idx}]: {vf} ---\n{extracted_contexts[vf]}\n\n"
    
    prompt = f"Using ONLY the following gathered context, answer the user's question. Try your best to piece together a helpful response, even if the information is partial or scattered. Be exhaustive with the facts provided.\nCRITICAL INSTRUCTION: You MUST use academic inline citations (e.g., [1], [2]) when stating facts, corresponding to the Source IDs provided in the context.\nCRITICAL INSTRUCTION: Your final answer MUST be written in the exact same language as the user's original question.\n\nToday's Date: {today}\n\nContext:\n{context}\n\nQuestion: {question}"
    final_answer = query_llm([{"role": "user", "content": prompt}], system_prompt="You are a resilient and helpful analyst.")
    
    if final_answer:
        ans = final_answer.strip()
        if visited_files:
            ans += "\n\n---\n**Sources Consulted:**\n"
            for idx, vf in source_mapping.items():
                display_name = vf.split("/")[-1].replace(".md", "")
                ans += f"[{idx}] [{display_name}]({vf})\n"
            
        out("\n--- Final Answer ---\n")
        out(ans)
        out("\n--------------\n")
        log_action("query", f"Answered '{question}' (Hit max hops={max_hops}). Visited: {list(visited_files)}")
        return ans
