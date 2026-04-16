---
description: How to use the LLM Wiki Agent (Ingest, Query, Lint)
---
# Using the LLM Wiki Agent

If the underlying architecture seems a bit complex, don't worry! Using the agent on a daily basis is actually incredibly simple. This workflow walks you through the 3 primary commands you need to know.

## Step 1: Ingesting New Information

Whenever you find a new transcript, news article, or dataset, save it in the `raw/` folder. Then, use the `ingest` command. The AI will automatically read it, pull out the important topics, deduplicate them against the database, and write them directly into your `pages/` directory and `index.md`!

```powershell
# Example: Ingest an arbitrary file (Text, PDF, JSON, or Excel)
python wiki_agent.py ingest "raw/sample_document.pdf"
```

## Step 2: Asking Questions (Multi-Hop Query)

When you want to search your knowledge base, use the `query` command. The AI will act like an autonomous researcher: it reads `index.md`, decides which files look relevant, opens them, and keeps "hopping" between files until it finds the perfect answer.

```powershell
# Example: Ask an advanced question and give it 4 "hops" to search
python wiki_agent.py query "How does TSMC's manufacturing affect Apple's timeline?" --max-hops 4
```

## Step 3: Maintenance and Linting

Over time, your wiki will get massive! You can run the `lint` command occasionally to automatically clean up the files, repair broken links, and fix formatting issues.

```powershell
# Run a quick check to fix broken markdown links
python wiki_agent.py lint
```

```powershell
# Run a deep Auto-Fix to forcefully rewrite older files into rich Wikipedia articles!
python wiki_agent.py lint --fix
```
