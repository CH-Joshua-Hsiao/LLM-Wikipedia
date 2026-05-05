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
import tempfile
import sys
import contextlib
import io
import datetime
import wiki_agent # Import the backend engine

# --- Config & State ---
st.set_page_config(page_title="LLM Wiki Chat", page_icon="📚", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

# --- Sidebar: Ingest & Compounding ---
with st.sidebar:
    st.title("📚 Wiki Admin")
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Upload Data (PDF, DOCX, XLSX, JSON, TXT)", type=['pdf', 'docx', 'xlsx', 'json', 'txt'])
    
    if uploaded_file is not None:
        if st.button("Ingest File", type="primary"):
            with st.status(f"Ingesting {uploaded_file.name}...", expanded=True) as status:
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Capture standard output from wiki_agent to display in Streamlit
                with capture_stdout() as output:
                    wiki_agent.ingest(tmp_path)
                
                logs = output.getvalue()
                st.code(logs, language="text")
                status.update(label="Ingestion Complete!", state="complete", expanded=False)
    
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
                         wiki_agent.ingest(tmp_path)
                         
                     logs = output.getvalue()
                     st.code(logs, language="text")
                     status.update(label="Knowledge Compounded Successfully!", state="complete")
                     # Clear memory optionally if we want the wiki to start fresh
                 else:
                     status.update(label="Failed to generate summary.", state="error")

# --- Chat UI ---
st.title("💬 LLM Wiki Assistant")

query_mode = st.toggle("🔍 Wiki Query Mode (Enable to trigger Multi-Hop RAG)", value=False)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

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
                with capture_stdout() as output:
                    wiki_response = wiki_agent.query(prompt, max_hops=3)
                
                logs = output.getvalue()
                st.code(logs, language="text") # Show the hops
                
                if not wiki_response:
                    wiki_response = "Sorry, I could not find an answer in the database."
                    status.update(label="Search Failed", state="error")
                else:
                    status.update(label="Answer Found", state="complete")
                    
            st.markdown(wiki_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": wiki_response})
            
        else:
            # Normal conversational mode
            full_context = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            with st.spinner("Thinking..."):
                normal_response = wiki_agent.query_llm(full_context, system_prompt="You are a helpful AI assistant. Use context from the current chat only.")
                if not normal_response:
                    normal_response = "Error parsing LLM response."
            
            st.markdown(normal_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": normal_response})
