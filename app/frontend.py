import uuid
import requests
import streamlit as st

st.set_page_config(
    page_title="RAG Technical Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:

    st.title("⚙️ Settings")

    st.markdown("---")

    st.markdown("""
### Features
- LangGraph Workflow
- ChromaDB Vector Search
- Groq LLM
- Query Rewriting
- Document Grading
- Conversational Memory
""")

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.markdown(
        "Built with FastAPI + LangGraph + Streamlit"
    )

st.title("📚 RAG Technical Documentation Assistant")

st.markdown("""
Ask questions about technical documentation using an AI-powered RAG pipeline.
""")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

question = st.chat_input(
    "Ask your question here..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        try:

            with st.spinner("Thinking..."):

                response = requests.post(
                    "http://localhost:8000/query",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": question
                    }
                )

                if response.status_code != 200:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                else:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer returned."
                    )

                    sources = data.get(
                        "sources",
                        []
                    )

                    formatted_response = answer

                    if sources:

                        formatted_response += (
                            "\n\n---\n## Sources\n"
                        )

                        for source in sources:

                            document = source.get(
                                "document",
                                "Unknown"
                            )

                            page = source.get(
                                "page",
                                "N/A"
                            )

                            formatted_response += (
                                f"- **{document}** "
                                f"(Page: {page})\n"
                            )

                    message_placeholder.markdown(
                        formatted_response
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_response
                    })

        except Exception as e:

            st.error(f"Error: {str(e)}")


st.markdown("---")

st.caption(
    "RAG Technical Documentation Assistant • "
    "Powered by LangGraph + ChromaDB + Groq"
)