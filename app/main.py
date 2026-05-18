import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.langgraph_workflow import app as rag_workflow

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

api = FastAPI(
    title="RAG Technical Documentation Assistant",
    version="2.0",
    description="RAG-powered technical documentation assistant using LangGraph"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = BASE_DIR / "docs"

DOCS_DIR.mkdir(exist_ok=True)

chat_sessions = {}

class QueryRequest(BaseModel):
    session_id: str
    question: str


class FeedbackRequest(BaseModel):
    rating: str
    comment: Optional[str] = None

@api.get("/")
def root():

    return {
        "message": "RAG Technical Documentation Assistant API",
        "status": "running"
    }


@api.post("/query")
def query_rag(request: QueryRequest):

    try:

        logger.info(f"Received query: {request.question}")

        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []

        history = chat_sessions[request.session_id]
        result = rag_workflow.invoke({
            "question": request.question,
            "chat_history": history,
            "retries": 0
        })

        answer = result.get("generation", "")

        formatted_answer = answer.replace("\\n", "\n")
        formatted_sources = []

        for idx, doc in enumerate(result.get("relevant_docs", []), start=1):

            source_data = {
                "id": idx,
                "document": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            }

            formatted_sources.append(source_data)

        history.append(f"User: {request.question}")

        history.append(f"Assistant: {formatted_answer}")

        logger.info("Query processed successfully")

        return {
            "question": request.question,
            "answer": formatted_answer,
            "sources": formatted_sources,
            "retrieved_documents": len(formatted_sources),
            "session_id": request.session_id
        }

    except Exception as e:

        logger.error(f"Query processing failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}"
        )

@api.get("/documents")
def list_documents():

    try:

        documents = []

        for file in DOCS_DIR.iterdir():

            if file.is_file():

                documents.append({
                    "filename": file.name,
                    "size_kb": round(file.stat().st_size / 1024, 2)
                })

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@api.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):

    try:

        uploaded_files = []

        for file in files:

            save_path = DOCS_DIR / file.filename

            content = await file.read()

            with open(save_path, "wb") as f:
                f.write(content)

            uploaded_files.append(file.filename)

            logger.info(f"Uploaded file: {file.filename}")

        return {
            "message": "Documents uploaded successfully",
            "uploaded_files": uploaded_files,
            "count": len(uploaded_files)
        }

    except Exception as e:

        logger.error(f"Upload failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )

@api.post("/feedback")
def feedback(data: FeedbackRequest):

    logger.info(f"Feedback received: {data.rating}")

    return {
        "message": "Feedback received successfully",
        "rating": data.rating,
        "comment": data.comment
    }