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
    usage_text = """
wiki_agent.py: The Local LLM Wiki Automation Agent (Multi-Tenant Version)

This agent acts as a corporate knowledge engine with hierarchical division namespaces.
Data is securely isolated per division, and cross-division queries are permitted
based on provided scope.

--- EXAMPLES ---
1. Ingest a document into the PPD division:
   python wiki_agent.py ingest "report.pdf" --division PPD

2. Query information across multiple divisions (e.g., PPD and ORP):
   python wiki_agent.py query "What is the status of Alpha Project?" --divisions PPD ORP

3. Perform a hygiene linting on the PPD division:
   python wiki_agent.py lint --division PPD

4. Perform a Deep RAG contradiction audit on the PPD division:
   python wiki_agent.py lint --deep --division PPD

5. Fully reset the entire database and wipe all namespaces:
   python wiki_agent.py reset
"""
    parser = argparse.ArgumentParser(
        description=usage_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["ingest", "query", "lint", "reset", "refresh-index"], help="Command to execute")
    parser.add_argument("args", nargs="*", help="Arguments for the command.")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API instead of local LiteLLM")
    parser.add_argument("--deep", action="store_true", help="RAG systemic contradiction audit (lint only)")
    parser.add_argument("--fix", action="store_true", help="Automatically revise and restructure all wiki pages during linting")
    parser.add_argument("--merge", action="store_true", help="Automerge mathematically and conceptually identical entities globally (lint only)")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum number of hops for multi-hop retrieval querying (default: 3)")
    parser.add_argument("--division", type=str, required=False, help="Division namespace for ingest/lint/refresh (e.g. PPD)")
    parser.add_argument("--divisions", type=str, nargs="+", required=False, help="Allowed divisions for query (e.g. PPD ORP)")
    
    args = parser.parse_args()
    if args.openai: config.USE_OPENAI = True
        
    cmd = args.command
    if cmd == "ingest":
        if not args.args or not args.division:
            print("Usage: python wiki_agent.py ingest <file_path> --division <div_name>")
            sys.exit(1)
        ingest(args.args[0], division=args.division)
    elif cmd == "query":
        if not args.args or not args.divisions:
            print("Usage: python wiki_agent.py query \"<question>\" --divisions <div1> <div2>")
            sys.exit(1)
        query(args.args[0], divisions=args.divisions, max_hops=args.max_hops)
    elif cmd == "lint":
        if not args.division:
            print("Usage: python wiki_agent.py lint --division <div_name>")
            sys.exit(1)
        lint(deep=args.deep, fix=args.fix, merge=args.merge, division=args.division)
    elif cmd == "reset":
        reset()
    elif cmd == "refresh-index":
        if not args.division:
            print("Usage: python wiki_agent.py refresh-index --division <div_name>")
            sys.exit(1)
        rebuild_all_indices(division=args.division)
