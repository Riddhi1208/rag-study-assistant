import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.retrieval import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA_MULT,
)

DB_PATH = "chroma_db"

st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="centered",
)
st.markdown(
    """
    <style>
    .block-container {
        max-width: 860px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )


vectordb = load_vectorstore()
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": RETRIEVAL_K,
        "fetch_k": RETRIEVAL_FETCH_K,
        "lambda_mult": RETRIEVAL_LAMBDA_MULT,
    },
)

llm = ChatOllama(
    model="phi3",
    temperature=0,
    num_predict=128,
)
prompt = ChatPromptTemplate.from_template(
    """
You are a friendly study assistant. Answer using only the supplied context.
Explain concepts clearly and concisely. If the context does not contain enough
information, say so instead of guessing.

Context:
{context}

Question:
{question}
"""
)
chain = prompt | llm | StrOutputParser()


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            label = source["source"]
            page = source["page"]
            if page is not None:
                page_label = page + 1 if isinstance(page, int) else page
                label = f"{label} · page {page_label}"
            st.write(f"**{index}. {label}**")
            st.caption(source["content"][:500])


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 📚 Study desk")
    st.caption("A private, document-grounded study space.")
    st.divider()
    st.markdown("### Get more from your materials")
    st.markdown(
        "- Ask a question about a concept.\n"
        "- Request a short explanation or summary.\n"
        "- Open **Sources** under an answer to review its supporting passages."
    )
    st.divider()
    st.caption("Your PDFs, embeddings, and answers stay on this machine.")

st.title("Study Assistant")
st.markdown("### Learn from your own notes")
st.write(
    "Ask questions about your study materials and get clear answers grounded "
    "in the passages they contain."
)

if not st.session_state.messages:
    st.info(
        "Start with a question about your indexed PDFs. "
        "I’ll look for relevant passages and show the sources behind each answer."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask about your study materials...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your materials..."):
            docs = retriever.invoke(question)
            if docs:
                context = "\n\n".join(doc.page_content for doc in docs)
                response = chain.invoke(
                    {
                        "context": context,
                        "question": question,
                    }
                )
                sources = [
                    {
                        "source": doc.metadata.get("source", "Study materials"),
                        "page": doc.metadata.get("page"),
                        "content": doc.page_content,
                    }
                    for doc in docs
                ]
            else:
                response = (
                    "I couldn't find relevant passages in your indexed materials. "
                    "Add PDFs to the data folder and rebuild the index with "
                    "`python ingest.py`."
                )
                sources = []

        st.markdown(response)
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources,
        }
    )
