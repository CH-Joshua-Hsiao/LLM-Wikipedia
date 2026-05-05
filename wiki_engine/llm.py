import requests
import os
from . import config

def query_llm(messages, system_prompt="You are a helpful assistant for managing a local knowledge base."):
    if config.USE_OPENAI:
        url = "https://api.openai.com/v1/chat/completions"
        api_key = config.OPENAI_API_KEY
        if not api_key:
            print("Error: OPENAI_API_KEY is missing! Ensure it is set in your .env file.")
            return None
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + messages
        }
    else:
        url = f"{config.API_BASE}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": config.MODEL_NAME,
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
        return None

def embed_text(text):
    if config.USE_OPENAI:
        url = "https://api.openai.com/v1/embeddings"
        api_key = config.OPENAI_API_KEY
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {"input": text, "model": "text-embedding-3-small"}
    else:
        url = f"{config.API_BASE}/embeddings"
        headers = {"Content-Type": "application/json"}
        payload = {"input": text, "model": config.EMBED_MODEL}
        
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

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
