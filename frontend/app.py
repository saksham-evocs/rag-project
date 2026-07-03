import requests
import streamlit as st # type: ignore

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper RAG Assistant")

st.write(
    "Ask questions from your 10 research papers. "
    "The answer will be generated using Qdrant + Ollama."
)

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What are the main findings across these papers?",
    height=120,
)

if st.button("Ask Question"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching papers and generating answer..."):
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=300,
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")

            for i, source in enumerate(data["sources"], start=1):
                with st.expander(
                    f"Source {i}: {source['paper_name']} | Page {source['page_number']} | Score {source['score']:.4f}"
                ):
                    st.write(source["text"])
        else:
            st.error("Backend error.")
            st.write(response.text)