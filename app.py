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

st.set_page_config(page_title="Local RAG Chatbot")

st.title("Local RAG Chatbot")
st.write("Chat with your notes and PDFs")


@st.cache_resource
def load_vectorstore():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return vectordb


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
    num_predict=128
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using the below as context.

Context:
{context}

Question:
{question}
"""
)

query = st.text_input("Ask a question")

if query:

    docs = retriever.invoke(query)

    st.subheader("Retrieved Chunks")

    for i, doc in enumerate(docs):
        st.write(f"Chunk {i+1}")
        st.write(doc.page_content[:500])
        st.divider()

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Generating answer..."):

        response = chain.invoke({
            "context": context,
            "question": query
        })

        st.subheader("Answer")
        st.write(response)

    st.subheader("Sources")

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        st.write(f"Source: {source} | Page: {page}")