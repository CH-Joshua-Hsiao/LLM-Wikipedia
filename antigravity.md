# LLM Wiki Schema (Antigravity)

This configuration file outlines the structure, schemas, and workflows for you (Antigravity), the LLM acting as the maintainer of this knowledge base.

## Architecture

- `raw/`: Your immutable sources. Do not modify these.
- `index.md`: A catalog of all wiki pages categorized by Entity, Concept, and Source. Update this when ingesting new sources.
- `log.md`: A chronological, append-only log of changes. Format: `## [YYYY-MM-DD] action | Description`.

## Operations

### Ingest
When instructed to ingest a source from the `raw/` directory:
1. Read the source file.
2. Abstract the core ideas and create/update relevant concept or entity markdown files in the wiki root.
3. Update `index.md` with links to the new/updated pages.
4. Record the operation in `log.md` starting with `## [YYYY-MM-DD] ingest | <source name>`.

### Query
When asked a query about the facts residing within the wiki:
1. Examine `index.md` to identify relevant wiki pages.
2. Synthesize an answer, pulling from those specific pages, and mention where contradictions or evolving concepts arise.
3. Create a new markdown page covering this new synthesis to compound the knowledge, if significant.

### Lint
When instructed to lint or health-check the wiki:
1. Iterate over the wiki files via tools to check for orphan content, disconnected concepts, or lack of cross-references.
2. Inform the user of contradictions, missing sources, or necessary cleanups.
