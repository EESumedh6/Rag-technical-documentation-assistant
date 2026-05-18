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

generation_prompt = PromptTemplate(
    input_variables=[
        "question",
        "context"
    ],
    template="""
You are a technical documentation assistant.

Your job is to answer the user's question
ONLY using the provided documentation context.

IMPORTANT RULES:
- Do NOT hallucinate
- Do NOT invent APIs or features
- If answer is missing, say:
  "I could not find this in the documentation."
- Keep answers technically accurate
- Use markdown formatting
- Use bullet points when helpful
- Use code blocks for examples
- Be concise but informative

User Question:
{question}

Documentation Context:
{context}

Generate a clear technical answer.
"""
)

def generate_answer(
    question: str,
    documents: List
) -> str:

    logger.info("Running answer generation")

    if not documents:

        logger.warning("No documents provided for generation")

        return (
            "I could not find relevant information "
            "in the documentation."
        )

    try:

        context = "\n\n".join([
            doc.page_content
            for doc in documents
        ])

        final_prompt = generation_prompt.format(
            question=question,
            context=context
        )
        response = llm.invoke(final_prompt)

        answer = response.content.strip()

        logger.info("Answer generated successfully")
        sources = []

        for idx, doc in enumerate(documents, start=1):

            source_name = doc.metadata.get(
                "source",
                "Unknown Source"
            )

            sources.append(
                f"[{idx}] {source_name}"
            )

        if sources:

            answer += "\n\n---\n## Sources\n"

            for source in sources:

                answer += f"- {source}\n"

        return answer

    except Exception as e:

        logger.error(
            f"Answer generation failed: {str(e)}"
        )

        return (
            "An error occurred while generating "
            "the response."
        )