import logging
from typing import TypedDict, List

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import CHROMA_PATH
load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model
)


class GraphState(TypedDict):

    question: str

    rewritten_query: str

    retrieved_docs: List

    relevant_docs: List

    generation: str

    retries: int

    chat_history: List[str]


def query_analysis(state: GraphState):

    logger.info("Running query analysis node")

    question = state["question"]

    history = "\n".join(
        state.get("chat_history", [])
    )

    prompt = f"""
You are a query rewriting assistant for a RAG system.

Your task:
- Understand the user's current question
- Use conversation history for context
- Rewrite the question into a fully self-contained query
- Improve retrieval quality
- Preserve technical meaning

Conversation History:
{history}

Current User Question:
{question}

Return ONLY the rewritten query.
"""

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()

    logger.info(f"Rewritten Query: {rewritten_query}")

    return {
        "rewritten_query": rewritten_query
    }


def retrieval(state: GraphState):

    logger.info("Running retrieval node")

    query = state["rewritten_query"]

    docs = vectorstore.similarity_search(
        query,
        k=4
    )

    logger.info(f"Retrieved {len(docs)} documents")

    return {
        "retrieved_docs": docs
    }

def document_grading(state: GraphState):

    logger.info("Running document grading node")

    question = state["question"]

    retrieved_docs = state["retrieved_docs"]

    relevant_docs = []

    grading_prompt = """
You are a document relevance grader.

Your task:
Determine whether the document helps answer the user's question.

Rules:
- YES if fully relevant
- YES if partially relevant
- NO only if unrelated

Respond ONLY with:
YES
or
NO

Question:
{question}

Document:
{document}
"""

    for doc in retrieved_docs:

        prompt = grading_prompt.format(
            question=question,
            document=doc.page_content
        )

        response = llm.invoke(prompt)

        grade = response.content.strip().upper()

        logger.info(f"Document Grade: {grade}")

        if "YES" in grade:
            relevant_docs.append(doc)

    logger.info(f"Relevant Docs Count: {len(relevant_docs)}")

    return {
        "relevant_docs": relevant_docs
    }

def decide_next_step(state: GraphState):

    relevant_docs = state["relevant_docs"]

    retries = state.get("retries", 0)

    if len(relevant_docs) > 0:

        logger.info("Routing to generation")

        return "generate"

    elif retries >= 1:

        logger.info("No relevant docs after retry")

        return "stop"

    else:

        logger.info("Retrying retrieval")

        return "retry"

def retry_query(state: GraphState):

    logger.info("Running retry query node")

    retries = state.get("retries", 0) + 1

    question = state["question"]

    retry_prompt = f"""
The previous retrieval attempt failed.

Rewrite the user's question differently
to improve semantic document retrieval.

Original Question:
{question}

Return ONLY the rewritten query.
"""

    response = llm.invoke(retry_prompt)

    rewritten_query = response.content.strip()

    logger.info(f"Retry Query: {rewritten_query}")

    return {
        "rewritten_query": rewritten_query,
        "retries": retries
    }

def generate_answer(state: GraphState):

    logger.info("Running generation node")

    question = state["question"]

    relevant_docs = state["relevant_docs"]

    context = "\n\n".join([
        doc.page_content
        for doc in relevant_docs
    ])

    generation_prompt = f"""
You are a technical documentation assistant.

IMPORTANT RULES:
- Answer ONLY using the provided context
- Do NOT hallucinate
- If answer is not present, say:
  "I could not find this in the documentation."
- Use markdown formatting
- Use bullet points where useful
- Use code blocks for technical examples
- Keep answers concise but technical

Question:
{question}

Context:
{context}

Provide the final answer.
"""

    response = llm.invoke(generation_prompt)

    answer = response.content.strip()

    logger.info("Answer generation completed")

    return {
        "generation": answer
    }

workflow = StateGraph(GraphState)


workflow.add_node(
    "query_analysis",
    query_analysis
)

workflow.add_node(
    "retrieval",
    retrieval
)

workflow.add_node(
    "document_grading",
    document_grading
)

workflow.add_node(
    "retry_query",
    retry_query
)

workflow.add_node(
    "generate_answer",
    generate_answer
)


workflow.set_entry_point(
    "query_analysis"
)

workflow.add_edge(
    "query_analysis",
    "retrieval"
)

workflow.add_edge(
    "retrieval",
    "document_grading"
)

workflow.add_conditional_edges(
    "document_grading",
    decide_next_step,
    {
        "generate": "generate_answer",
        "retry": "retry_query",
        "stop": END
    }
)

workflow.add_edge(
    "retry_query",
    "retrieval"
)

workflow.add_edge(
    "generate_answer",
    END
)

app = workflow.compile()