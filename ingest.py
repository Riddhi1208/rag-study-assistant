from collections.abc import Sequence
from io import BytesIO
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from utils.retrieval import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

DATA_PATH = "data"
DB_PATH = "chroma_db"


def load_documents():

    documents = []

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(DATA_PATH, file)

            loader = PyPDFLoader(pdf_path)

            documents.extend(loader.load())

    return documents


def load_uploaded_documents(
    uploaded_files: Sequence[tuple[str, bytes]],
) -> tuple[list[Document], list[str]]:
    documents = []
    files_without_text = []

    for filename, file_bytes in uploaded_files:
        try:
            reader = PdfReader(BytesIO(file_bytes))
        except PdfReadError as error:
            raise ValueError(f"Could not read uploaded PDF '{filename}'") from error

        if reader.is_encrypted:
            raise ValueError(f"Password-protected PDF '{filename}' is not supported")

        pages_with_text = 0
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"source": filename, "page": page_number},
                    )
                )
                pages_with_text += 1

        if not pages_with_text:
            files_without_text.append(filename)

    return documents, files_without_text


def split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    ).delete_collection()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
    )

    print("Vector DB rebuilt successfully")


if __name__ == "__main__":

    docs = load_documents()

    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    create_vector_store(chunks)