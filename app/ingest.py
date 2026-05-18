import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from app.config import DOCS_PATH, CHROMA_PATH

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

def load_documents() -> List:

    logger.info("Loading documents")

    documents = []

    supported_extensions = [".txt", ".md"]

    for file_path in DOCS_PATH.iterdir():

        if (
            file_path.is_file()
            and file_path.suffix in supported_extensions
        ):

            try:

                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

                loaded_docs = loader.load()

                # -----------------------------------------
                # ADD METADATA
                # -----------------------------------------

                for doc in loaded_docs:

                    doc.metadata["source"] = file_path.name

                documents.extend(loaded_docs)

                logger.info(
                    f"Loaded document: {file_path.name}"
                )

            except Exception as e:

                logger.error(
                    f"Failed to load {file_path.name}: {str(e)}"
                )

    logger.info(
        f"Total documents loaded: {len(documents)}"
    )

    return documents

def split_documents(documents: List):

    logger.info("Splitting documents into chunks")

    chunks = text_splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = idx

    logger.info(
        f"Created {len(chunks)} chunks"
    )

    return chunks

def create_vectorstore(chunks: List):

    logger.info("Creating Chroma vector store")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PATH)
    )

    vectorstore.persist()

    logger.info(
        "Documents embedded and stored in ChromaDB"
    )

def run_ingestion_pipeline():

    logger.info("Starting ingestion pipeline")

    documents = load_documents()

    if not documents:

        logger.warning("No documents found")

        return

    chunks = split_documents(documents)

    create_vectorstore(chunks)

    logger.info("Ingestion pipeline completed")

if __name__ == "__main__":

    run_ingestion_pipeline()