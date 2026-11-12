from uuid import uuid4

import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ingest import load_uploaded_documents, split_documents
from quiz import (
    QUIZ_QUESTION_COUNT,
    QUIZ_RESPONSE_SCHEMA,
    QuizOutputError,
    parse_quiz_response,
)
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
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        display: none !important;
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


folder_vectorstore = load_vectorstore()

llm = ChatOllama(
    model="phi3",
    temperature=0,
    num_predict=128,
)
quiz_llm = ChatOllama(
    model="phi3",
    temperature=0,
    num_predict=1536,
    format=QUIZ_RESPONSE_SCHEMA,
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
quiz_prompt = ChatPromptTemplate.from_template(
    """
You are an expert study quiz writer. Use only the supplied source passages.
Create exactly 10 distinct multiple-choice questions for the requested topic.
Each question must be answerable from the supplied source passages. Write four
distinct, short options with exactly one correct answer. Make all options
similar in length and grammar. Never repeat an option or make one option a
longer version of another. Use three plausible but incorrect distractors.
Avoid questions that require outside knowledge. Keep each question under 20
words, each option under 10 words, and each explanation to one sentence under
20 words.

Return only a JSON object in this format:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": 0,
      "explanation": "Why the correct answer follows from the source"
    }}
  ]
}}

`correct_answer` is a zero-based option index. Keep the JSON compact and do
not include extra fields.

Topic:
{topic}

Source passages:
{context}
"""
)
quiz_chain = quiz_prompt | quiz_llm | StrOutputParser()


def source_label(source):
    label = source["source"]
    page = source["page"]
    if page is not None:
        page_label = page + 1 if isinstance(page, int) else page
        label = f"{label} · page {page_label}"
    return label


def render_sources(sources, title="Sources"):
    with st.expander(f"{title} ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            st.write(f"**{index}. {source_label(source)}**")
            st.caption(source["content"][:500])


def render_quiz_feedback(quiz, question_index):
    question = quiz["questions"][question_index]
    selected_answer = quiz["answers"][question_index]
    is_correct = selected_answer == question["correct_answer"]

    if is_correct:
        st.success("Correct!")
    else:
        st.error(
            "Not quite. The correct answer is "
            f"**{question['options'][question['correct_answer']]}**."
        )
    st.write(question["explanation"])


if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "uploaded_vectorstore" not in st.session_state:
    st.session_state.uploaded_vectorstore = None
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

st.title("Study Assistant")
st.markdown("### Learn from your own notes")
st.write(
    "Ask questions about your study materials and get clear answers grounded "
    "in the passages they contain."
)

with st.container(border=True):
    st.subheader("Study materials")
    st.caption(
        "Upload PDFs to make them the active sources for this session. "
        "This won't change the saved folder index."
    )
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="study_pdf_upload",
    )
    if st.button("Index uploaded PDFs", type="primary"):
        if not uploaded_files:
            st.warning("Choose at least one PDF to index.")
        else:
            file_bytes = [
                (uploaded_file.name, uploaded_file.getvalue())
                for uploaded_file in uploaded_files
            ]
            try:
                uploaded_documents, files_without_text = load_uploaded_documents(
                    file_bytes
                )
            except ValueError as error:
                st.error(str(error))
            else:
                if not uploaded_documents:
                    st.error(
                        "No selectable text was found in the uploaded PDFs. "
                        "Scanned PDFs need OCR before they can be indexed."
                    )
                else:
                    chunks = split_documents(uploaded_documents)
                    if not chunks:
                        st.error("No text chunks could be created from these PDFs.")
                    else:
                        with st.spinner("Embedding your uploaded study materials..."):
                            uploaded_vectorstore = Chroma.from_documents(
                                documents=chunks,
                                embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
                                collection_name=f"study_upload_{uuid4().hex}",
                            )
                        st.session_state.uploaded_vectorstore = uploaded_vectorstore
                        st.session_state.uploaded_filenames = [
                            uploaded_file.name for uploaded_file in uploaded_files
                        ]
                        st.session_state.quiz = None
                        st.session_state.messages = []
                        st.success(
                            f"Indexed {len(chunks)} chunks from "
                            f"{len(uploaded_files)} PDF(s) for this session."
                        )
                        if files_without_text:
                            st.warning(
                                "No selectable text was found in: "
                                + ", ".join(files_without_text)
                            )
if st.session_state.uploaded_vectorstore is not None:
    active_vectorstore = st.session_state.uploaded_vectorstore
    active_documents_label = "Uploaded PDFs: " + ", ".join(
        st.session_state.uploaded_filenames
    )
else:
    active_vectorstore = folder_vectorstore
    active_documents_label = "Using the PDFs indexed from the data folder."
st.caption(f"Active materials: {active_documents_label}")
retriever = active_vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": RETRIEVAL_K,
        "fetch_k": RETRIEVAL_FETCH_K,
        "lambda_mult": RETRIEVAL_LAMBDA_MULT,
    },
)

chat_tab, quiz_tab = st.tabs(["Ask a question", "Practice quiz"])

with chat_tab:
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
                        "Upload PDFs above or add PDFs to the data folder and rebuild "
                        "the index with `python ingest.py`."
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

with quiz_tab:
    st.markdown(f"Create a {QUIZ_QUESTION_COUNT}-question quiz from your materials.")
    topic = st.text_input(
        "Topic or focus (optional)",
        placeholder="For example: cell structure",
        key="quiz_topic",
    )
    if st.button("Generate quiz", type="primary", key="generate_quiz"):
        st.session_state.quiz = None
        search_query = topic.strip() or "key concepts and important facts"
        with st.spinner("Finding study passages and writing your quiz..."):
            quiz_docs = active_vectorstore.max_marginal_relevance_search(
                search_query,
                k=10,
                fetch_k=max(10, RETRIEVAL_FETCH_K),
                lambda_mult=RETRIEVAL_LAMBDA_MULT,
            )
            if not quiz_docs:
                st.warning(
                    "I couldn't find any indexed passages to build a quiz from. "
                    "Upload PDFs above or add PDFs to the data folder and rebuild "
                    "the index with `python ingest.py`."
                )
            else:
                sources = [
                    {
                        "source": doc.metadata.get("source", "Study materials"),
                        "page": doc.metadata.get("page"),
                        "content": doc.page_content,
                    }
                    for doc in quiz_docs
                ]
                context = "\n\n".join(
                    f"[Source {index}: {source_label(source)}]\n{source['content']}"
                    for index, source in enumerate(sources, start=1)
                )
                quiz_topic = topic.strip() or "General review"
                raw_response = quiz_chain.invoke(
                    {
                        "context": context,
                        "topic": quiz_topic,
                    }
                )
                try:
                    questions = parse_quiz_response(raw_response)
                except QuizOutputError as error:
                    st.error(f"Could not create a valid quiz: {error}")
                else:
                    st.session_state.quiz = {
                        "id": uuid4().hex,
                        "topic": quiz_topic,
                        "questions": questions,
                        "sources": sources,
                        "answers": [None] * QUIZ_QUESTION_COUNT,
                        "current_index": 0,
                        "completed": False,
                    }

    quiz = st.session_state.quiz
    if quiz is None:
        st.info(
            "Choose a topic or leave it blank for a general review, then generate "
            "a quiz grounded in your indexed study materials."
        )
    elif quiz["completed"]:
        correct_count = sum(
            answer == question["correct_answer"]
            for answer, question in zip(quiz["answers"], quiz["questions"])
        )
        st.progress(1.0)
        st.metric(
            "Final score",
            f"{correct_count}/{QUIZ_QUESTION_COUNT}",
            f"{correct_count / QUIZ_QUESTION_COUNT:.0%}",
        )
        if correct_count == QUIZ_QUESTION_COUNT:
            st.success("Excellent work — you got every question right.")
        else:
            st.subheader("Review questions to revisit")
            for index, (question, answer) in enumerate(
                zip(quiz["questions"], quiz["answers"]),
                start=1,
            ):
                if answer == question["correct_answer"]:
                    continue
                with st.expander(f"Question {index}: {question['question']}"):
                    st.write(
                        "Your answer: "
                        f"{question['options'][answer]}"
                    )
                    st.write(
                        "Correct answer: "
                        f"{question['options'][question['correct_answer']]}"
                    )
                    st.write(question["explanation"])
        render_sources(quiz["sources"], title="Retrieved passages for this quiz")
    else:
        question_index = quiz["current_index"]
        question = quiz["questions"][question_index]
        answered_count = sum(answer is not None for answer in quiz["answers"])
        st.progress(answered_count / QUIZ_QUESTION_COUNT)
        st.caption(
            f"Question {question_index + 1} of {QUIZ_QUESTION_COUNT}"
            f" · {quiz['topic']}"
        )
        st.markdown(f"**{question['question']}**")

        if quiz["answers"][question_index] is None:
            selected = st.radio(
                "Choose one answer",
                question["options"],
                index=None,
                key=f"quiz-choice-{quiz['id']}-{question_index}",
            )
            if st.button(
                "Check answer",
                disabled=selected is None,
                key=f"check-answer-{quiz['id']}-{question_index}",
            ):
                if selected is not None:
                    quiz["answers"][question_index] = question["options"].index(
                        selected
                    )
                    st.rerun()
        else:
            render_quiz_feedback(quiz, question_index)
            if question_index + 1 < QUIZ_QUESTION_COUNT:
                if st.button(
                    "Next question",
                    type="primary",
                    key=f"next-question-{quiz['id']}-{question_index}",
                ):
                    quiz["current_index"] += 1
                    st.rerun()
            elif st.button(
                "Finish quiz",
                type="primary",
                key=f"finish-quiz-{quiz['id']}",
            ):
                quiz["completed"] = True
                st.rerun()
        render_sources(quiz["sources"], title="Retrieved passages for this quiz")
