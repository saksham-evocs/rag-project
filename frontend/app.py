import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper RAG Assistant")

st.write("Hi, what's on your mind today?")

question = st.text_area(
    "Ask your question:",
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
                source_type = source.get("type", "text")

                if source_type == "figure":
                    title = (
                        f"Figure {i}: {source['paper_name']} | "
                        f"Page {source['page_number']} | "
                        f"Score {source['score']:.4f}"
                    )

                    with st.expander(title):
                        st.write("Caption:")
                        st.write(source.get("caption", "No caption found."))

                        if source.get("image_path"):
                            st.image(
                                source["image_path"],
                                caption=f"{source['paper_name']} | Page {source['page_number']}",
                                use_container_width=True,
                            )

                else:
                    title = (
                        f"Text Source {i}: {source['paper_name']} | "
                        f"Page {source['page_number']} | "
                        f"Score {source['score']:.4f}"
                    )

                    with st.expander(title):
                        st.write(source["text"])

        else:
            st.error("Backend error.")
            st.write(response.text)