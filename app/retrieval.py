import logging
from typing import List

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import CHROMA_PATH

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model
)

def retrieve_documents(
    query: str,
    k: int = 4
) -> List:

    logger.info(f"Running retrieval for query: {query}")

    try:
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )

        retrieved_docs = []

        for doc, score in results:


            doc.metadata["similarity_score"] = round(score, 4)

            retrieved_docs.append(doc)

        logger.info(
            f"Retrieved {len(retrieved_docs)} documents"
        )

        return retrieved_docs

    except Exception as e:

        logger.error(f"Retrieval failed: {str(e)}")

        return []