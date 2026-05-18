from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_PATH = BASE_DIR / r"C:\Users\sumed\PycharmProjects\RAG_tech_assistant\docs"
CHROMA_PATH = BASE_DIR / r"C:\Users\sumed\PycharmProjects\RAG_tech_assistant\chrome_db"


DOCS_PATH.mkdir(exist_ok=True)
CHROMA_PATH.mkdir(exist_ok=True)