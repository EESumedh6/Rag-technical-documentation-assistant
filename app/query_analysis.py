import logging
from typing import Dict, List

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

query_prompt = PromptTemplate(
    input_variables=[
        "chat_history",
        "question"
    ],
    template="""
You are a query rewriting assistant for a RAG system.

Your responsibilities:
1. Rewrite the user's question to improve semantic retrieval.
2. Make the query self-contained.
3. Use conversation history for context.
4. Preserve technical meaning.
5. Classify the query type.

Possible query types:
- conceptual
- how-to
- troubleshooting
- api-reference

Conversation History:
{chat_history}

Current User Question:
{question}

Return ONLY in this exact format:

Rewritten Query:
<rewritten query>

Query Type:
<query type>
"""
)

def analyze_query(
    question: str,
    chat_history: List[str] = None
) -> Dict:

    logger.info("Running query analysis")

    if chat_history is None:
        chat_history = []

    history_text = "\n".join(chat_history)

    formatted_prompt = query_prompt.format(
        chat_history=history_text,
        question=question
    )

    response = llm.invoke(formatted_prompt)

    output = response.content.strip()

    logger.info(f"Raw Query Analysis Output: {output}")

    rewritten_query = question

    query_type = "conceptual"

    try:

        lines = output.splitlines()

        for line in lines:

            if line.startswith("Rewritten Query:"):

                rewritten_query = (
                    line.replace(
                        "Rewritten Query:",
                        ""
                    ).strip()
                )

            elif line.startswith("Query Type:"):

                query_type = (
                    line.replace(
                        "Query Type:",
                        ""
                    ).strip()
                )

    except Exception as e:

        logger.error(f"Query parsing failed: {str(e)}")

    logger.info(f"Rewritten Query: {rewritten_query}")

    logger.info(f"Query Type: {query_type}")

    return {
        "rewritten_query": rewritten_query,
        "query_type": query_type
    }