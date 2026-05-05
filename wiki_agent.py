"""
wiki_agent.py: The Local LLM Wiki Automation Agent (CLI Wrapper)

This script manages your local knowledge base by parsing raw documents, querying the structured
data, and linting the wiki files.

It acts as a thin wrapper over the `wiki_engine` package.
"""
import sys
import argparse
from wiki_engine import config, ingest, query, lint, reset, rebuild_all_indices, query_llm

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
    if args.openai: config.USE_OPENAI = True
        
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
