"""
LLM-Wikipedia Streamlit Interface & Knowledge Compounding Agent

This module provides a GUI wrapper for `wiki_agent.py`. It enables stateful ChatGPT-style conversation
with two modes:
1. Normal Chat: Relying purely on conversation memory.
2. Wiki Query Mode: Activating the Multi-Hop RAG engine to query the local database.

Features:
- File Upload Ingest: Drag and drop files to populate the database.
- Knowledge Compounding: Synthesize insights from your chat history and push them into the permanent Wiki.

To run this application, type the following command in your terminal:
    streamlit run app.py
"""
import streamlit as st
import os
import re
import tempfile
import sys
import contextlib
import io
import datetime
import wiki_agent # Import the backend engine

# --- Config & State ---
st.set_page_config(page_title="Weekly Wiki", page_icon="📖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_wiki_file" not in st.session_state:
    st.session_state.active_wiki_file = None

# Function to capture printed stdout to show in UI
@contextlib.contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
         yield buffer
    finally:
         sys.stdout = old_stdout

# --- Sidebar: User Profile & Admin ---
if not os.path.exists("users"):
    os.makedirs("users", exist_ok=True)

with st.sidebar:
    st.title("👤 User Profile")
    user_id = st.text_input("User ID", value="guest")
    user_file_path = os.path.join("users", f"{user_id}.md")
    
    current_personality = ""
    if os.path.exists(user_file_path):
        with open(user_file_path, "r", encoding="utf-8") as f:
            current_personality = f.read()
            
    with st.expander("📝 Edit Personality / Preferences"):
        personality_text = st.text_area("Your QA Preferences", value=current_personality, height=150)
        
        col1, col2 = st.columns(2)
        if col1.button("Save Profile"):
            with open(user_file_path, "w", encoding="utf-8") as f:
                f.write(personality_text)
            st.success("Saved!")
            st.rerun()
            
        if col2.button("🧠 Auto-Evolve"):
            if not st.session_state.messages:
                st.warning("No chat history.")
            else:
                with st.spinner("Analyzing..."):
                    chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-10:]])
                    evolve_prompt = f"Analyze the following chat history and the user's EXISTING personality profile. Evolve and update the personality profile to better reflect their communication style, language preference, and technical depth. Output ONLY the updated personality profile text.\n\n=== Existing Profile ===\n{current_personality}\n\n=== Chat History ===\n{chat_history}"
                    new_personality = wiki_agent.query_llm([{"role": "user", "content": evolve_prompt}], system_prompt="You are a behavioral analyst.")
                    if new_personality:
                        with open(user_file_path, "w", encoding="utf-8") as f:
                            f.write(new_personality.strip())
                        st.success("Evolved!")
                        st.rerun()
                        
    st.markdown("---")
    st.title("📚 Wiki Admin")
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Upload Data (PDF, DOCX, XLSX, JSON, TXT)", type=['pdf', 'docx', 'xlsx', 'json', 'txt'])
    ingest_div = st.text_input("Target Division for Ingest (e.g. PPD)", value="General")
    
    if uploaded_file is not None:
        if st.button("Ingest File", type="primary"):
            with st.status(f"Ingesting {uploaded_file.name}...", expanded=True) as status:
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Capture standard output from wiki_agent to display in Streamlit
                with capture_stdout() as output:
                    wiki_agent.ingest(tmp_path, division=ingest_div)
                
                logs = output.getvalue()
                st.code(logs, language="text")
                status.update(label="Ingestion Complete!", state="complete", expanded=False)
    
    st.markdown("---")
    st.subheader("Query Scope")
    available_divisions = []
    if os.path.exists("namespaces"):
        available_divisions = [d for d in os.listdir("namespaces") if os.path.isdir(os.path.join("namespaces", d))]
    selected_divisions = st.multiselect("Select Divisions to Query", options=available_divisions, default=available_divisions)
    st.markdown("---")
    st.subheader("Knowledge Compounding")
    st.markdown("Extract insights from your chat history and save them to the Wiki.")
    if st.button("Synthesize & Ingest Chat"):
        if not st.session_state.messages:
             st.warning("No chat history to synthesize!")
        else:
             with st.status("Analyzing chat history...", expanded=True) as status:
                 # Reconstruct chat log
                 chat_history = ""
                 for msg in st.session_state.messages:
                     chat_history += f"{msg['role'].upper()}: {msg['content']}\n\n"
                 
                 # Task the LLM to summarize
                 compounding_prompt = f"""
You are a Knowledge Extraction Agent. Review the following chat history and extract any novel, factual concepts or entities discussed.
Format your response as a comprehensive, structured text article focusing purely on the facts, avoiding conversational filler.

=== Chat History ===
{chat_history}
"""
                 sys.stdout.write("Drafting summary logic...\n")
                 summary_response = wiki_agent.query_llm([{"role": "user", "content": compounding_prompt}], system_prompt="You are an expert structural editor.")
                 
                 if summary_response:
                     # Save temporarily to trigger normal ingest
                     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                     tmp_path = os.path.join(tempfile.gettempdir(), f"chat_summary_{timestamp}.txt")
                     with open(tmp_path, "w", encoding="utf-8") as f:
                         f.write(summary_response)
                     
                     sys.stdout.write("Summarization complete. Handing off to Wiki Ingest Engine...\n")
                     
                     with capture_stdout() as output:
                         wiki_agent.ingest(tmp_path, division=ingest_div)
                         
                     logs = output.getvalue()
                     st.code(logs, language="text")
                     status.update(label="Knowledge Compounded Successfully!", state="complete")
                     # Clear memory optionally if we want the wiki to start fresh
                 else:
                     status.update(label="Failed to generate summary.", state="error")

# --- Chat UI ---
chat_col, wiki_col = st.columns([6, 4])

with chat_col:
    st.title("💬 Weekly Wiki")

    query_mode = st.toggle("🔍 Wiki Query Mode (Enable to trigger Multi-Hop RAG)", value=False)

    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message.get("sources"):
                cols = st.columns(min(len(message["sources"]), 4))
                for i, (name, path) in enumerate(message["sources"]):
                    if cols[i % 4].button(f"📄 {name}", key=f"hist_btn_{idx}_{i}"):
                        st.session_state.active_wiki_file = path

    # Input box
    if prompt := st.chat_input("Ask me anything..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=True)

        # Process AI Response
        with st.chat_message("assistant"):
            if query_mode:
                # Trigger wiki multi-hop
                with st.status("Consulting Wiki Database...", expanded=True) as status:
                    st_placeholder = st.empty()
                    
                    injected_prompt = prompt
                    if os.path.exists(user_file_path):
                        with open(user_file_path, "r", encoding="utf-8") as f:
                            pref = f.read().strip()
                        if pref:
                            injected_prompt = f"[User QA Preference: You are answering {user_id}. Follow these rules closely:\n{pref}]\n\nUser Question: {prompt}"
                    
                    wiki_response = wiki_agent.query(injected_prompt, divisions=selected_divisions, max_hops=3, st_placeholder=st_placeholder)
                    
                    if not wiki_response:
                        wiki_response = "Sorry, I could not find an answer in the database."
                        status.update(label="Search Failed", state="error")
                    else:
                        status.update(label="Answer Found", state="complete")
                        
                st.markdown(wiki_response, unsafe_allow_html=True)
                sources = []
                if "**Sources Consulted:**" in wiki_response:
                    sources_part = wiki_response.split("**Sources Consulted:**")[1]
                    sources = re.findall(r"\[([^\]]+)\]\(([^)]+\.md)\)", sources_part)
                
                st.session_state.messages.append({"role": "assistant", "content": wiki_response, "sources": sources})
                st.rerun()
                
            else:
                # Normal conversational mode
                full_context = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                
                system_prompt = "You are a helpful AI assistant. Use context from the current chat only."
                if os.path.exists(user_file_path):
                    with open(user_file_path, "r", encoding="utf-8") as f:
                        pref = f.read().strip()
                    if pref:
                        system_prompt += f"\n\nUser QA Preference (You are answering {user_id}):\n{pref}"
                
                with st.spinner("Thinking..."):
                    normal_response = wiki_agent.query_llm(full_context, system_prompt=system_prompt)
                    if not normal_response:
                        normal_response = "Error parsing LLM response."
                
                st.markdown(normal_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": normal_response})


with wiki_col:
    st.title("📖 Wiki Viewer")
    st.markdown("---")
    if st.session_state.active_wiki_file and os.path.exists(st.session_state.active_wiki_file):
        current_file = st.session_state.active_wiki_file
        st.subheader(os.path.basename(current_file).replace(".md", "").replace("_", " "))
        
        with open(current_file, "r", encoding="utf-8") as f:
            wiki_md = f.read()
        
        # Pre-process: strip internal .md links and raw file links so they don't render as
        # broken browser hyperlinks. Navigation is handled by buttons at the bottom of the page.
        display_md = re.sub(r'\[([^\]]+)\]\([^)]*\.md\)', r'**\1**', wiki_md)
        display_md = re.sub(r'\[([^\]]+)\]\([^)]+\.(?:pdf|docx|xlsx|txt|csv|json|pptx)\)', r'`\1`', display_md, flags=re.IGNORECASE)
        
        # Render the wiki page content (internal links stripped, navigation via buttons below)
        st.markdown(display_md, unsafe_allow_html=True)
        
        # --- Related Pages Navigation ---
        # Scan the markdown for all internal .md links and render them as buttons
        internal_links = re.findall(r'\[([^\]]+)\]\(([^)]*\.md)\)', wiki_md)
        
        # Scan for raw source file links (pdf, docx, xlsx, txt, csv, json) in References section
        raw_links = re.findall(r'\[([^\]]+)\]\(([^)]+\.(?:pdf|docx|xlsx|txt|csv|json|pptx))\)', wiki_md, re.IGNORECASE)
        
        # Resolve path helpers — must be computed before both sections below
        current_dir = os.path.dirname(current_file)
        base_ns_dir = os.path.dirname(current_dir)  # e.g. namespaces/PPD
        
        if internal_links:
            st.markdown("---")
            st.markdown("**🔗 Related Pages**")
            
            btn_cols = st.columns(min(len(internal_links), 3))
            rendered = set()
            for i, (link_text, link_path) in enumerate(internal_links):
                if link_text in rendered:
                    continue
                rendered.add(link_text)
                
                # Try resolving the path in several ways to handle LLM inconsistencies
                candidates = [
                    os.path.normpath(os.path.join(current_dir, link_path)),                 # relative to current file
                    os.path.normpath(os.path.join(base_ns_dir, link_path)),                 # relative to namespace root
                    os.path.normpath(os.path.join(base_ns_dir, "pages", os.path.basename(link_path))),  # always in pages/
                    os.path.normpath(os.path.join(current_dir, os.path.basename(link_path))), # just filename in same dir
                ]
                resolved = next((c for c in candidates if os.path.exists(c)), None)
                
                if resolved:
                    if btn_cols[i % 3].button(f"📄 {link_text}", key=f"wiki_nav_{i}_{link_text}"):
                        st.session_state.active_wiki_file = resolved
                        st.rerun()
        
        # --- Raw Source Files ---
        if raw_links:
            st.markdown("---")
            st.markdown("**📎 Raw Source Files**")
            
            rendered_raw = set()
            for raw_text, raw_path in raw_links:
                raw_basename = os.path.basename(raw_path)
                if raw_basename in rendered_raw:
                    continue
                rendered_raw.add(raw_basename)
                
                # Try to resolve the raw file path
                candidates = [
                    os.path.normpath(raw_path),                                                              # as-is (relative to CWD)
                    os.path.normpath(os.path.join(current_dir, raw_path)),                                   # relative to wiki file
                    os.path.normpath(os.path.join(base_ns_dir, raw_path)),                                   # relative to namespace
                    os.path.normpath(os.path.join(base_ns_dir, "raw", raw_basename)),                        # always in raw/
                    os.path.normpath(os.path.join(current_dir, "raw", raw_basename)),                        # raw/ next to pages/
                ]
                resolved_raw = next((c for c in candidates if os.path.exists(c)), None)
                
                if resolved_raw:
                    ext = os.path.splitext(resolved_raw)[1].lower()
                    with open(resolved_raw, "rb") as rf:
                        raw_bytes = rf.read()
                    
                    col_dl, col_preview = st.columns([1, 3])
                    col_dl.download_button(
                        label=f"⬇️ {raw_basename}",
                        data=raw_bytes,
                        file_name=raw_basename,
                        key=f"dl_{raw_basename}"
                    )
                    # Preview text-based files inline with word wrap
                    if ext == ".txt":
                        with col_preview.expander(f"Preview: {raw_basename}"):
                            import html as _html
                            import streamlit.components.v1 as components
                            text_content = raw_bytes.decode("utf-8", errors="replace")
                            escaped = _html.escape(text_content)
                            line_count = text_content.count('\n') + 1
                            height = min(max(line_count * 20, 150), 500)
                            components.html(
                                f'<html><body style="margin:0;background:#0e1117;">'
                                f'<pre style="white-space:pre-wrap;word-break:break-word;'
                                f'font-family:monospace;font-size:13px;color:#fafafa;padding:12px;margin:0;">'
                                f'{escaped}</pre></body></html>',
                                height=height,
                                scrolling=True
                            )
                else:
                    st.caption(f"⚠️ {raw_basename} — file not found on server")
    else:
        st.info("👈 Click a source button in the chat to view the Wiki page here.")

