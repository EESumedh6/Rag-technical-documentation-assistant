import logging
from typing import List

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


grading_prompt = PromptTemplate(
    input_variables=[
        "question",
        "document"
    ],
    template="""
You are a document relevance grader for a RAG pipeline.

Your task:
Determine whether the retrieved document
contains information useful for answering
the user's question.

Grading Rules:
- YES if directly relevant
- YES if partially relevant
- YES if technically related
- NO only if completely unrelated

Be slightly permissive.

Respond ONLY with:
YES
or
NO

User Question:
{question}

Retrieved Document:
{document}
"""
)

def grade_documents(
    question: str,
    documents: List
) -> List:

    logger.info("Running document grading")

    relevant_docs = []

    for idx, doc in enumerate(documents, start=1):

        try:

            logger.info(f"Grading document {idx}")

            prompt = grading_prompt.format(
                question=question,
                document=doc.page_content
            )

            response = llm.invoke(prompt)

            grade = response.content.strip().upper()

            logger.info(f"Document {idx} Grade: {grade}")

            if "YES" in grade:

                doc.metadata["relevance_grade"] = "YES"

                relevant_docs.append(doc)

            else:

                doc.metadata["relevance_grade"] = "NO"

        except Exception as e:

            logger.error(
                f"Document grading failed: {str(e)}"
            )

    logger.info(
        f"Relevant documents selected: {len(relevant_docs)}"
    )

    return relevant_docs