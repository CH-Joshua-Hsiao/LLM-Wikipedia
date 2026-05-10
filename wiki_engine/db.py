from . import config

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None
    register_vector = None

def init_db():
    if not psycopg2:
        return None
    try:
        conn = psycopg2.connect(config.DB_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS wiki_entities (
                id SERIAL PRIMARY KEY,
                division TEXT NOT NULL,
                name TEXT NOT NULL,
                embedding vector,
                UNIQUE(division, name)
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS wiki_links (
                division TEXT NOT NULL,
                source_file TEXT NOT NULL,
                target_file TEXT NOT NULL,
                PRIMARY KEY (division, source_file, target_file)
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS wiki_claims (
                id SERIAL PRIMARY KEY,
                division TEXT NOT NULL,
                source_file TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                embedding vector
            );
            """)
        return conn
    except Exception as e:
        print(f"PostgreSQL connection failed. DB Logic bypassed. Reason: {e}")
        return None
