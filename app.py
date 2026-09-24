import os
import sys

from dotenv import load_dotenv

load_dotenv()

from loaders import load_all_documents
from preprocess import preprocess_documents, split_documents
from vector_store import create_vector_store, load_vector_store
from retriever import create_retriever
from rag_chain import generate_answer
from memory import create_memory


PDF_DIRECTORY = "data/pdfs"
TEXT_DIRECTORY = "data/text"
WEB_SOURCE_FILE = "web_sources.txt"

TOP_K = 4
RELEVANCE_THRESHOLD = 0.35


def build_knowledge_base():
    print("\n--- Knowledge Base Construction ---")
    print("Loading documents from PDFs, Text files, and Web sources...")

    documents = load_all_documents(
        PDF_DIRECTORY,
        TEXT_DIRECTORY,
        WEB_SOURCE_FILE
    )

    print(f"Loaded total raw documents: {len(documents)}")

    print("Cleaning and preprocessing document content...")

    documents = preprocess_documents(documents)

    print(f"Cleaned documents count: {len(documents)}")

    print("Splitting text recursively into chunks...")

    chunks = split_documents(
        documents,
        chunk_size=800,
        chunk_overlap=150
    )

    print(f"Generated text chunks: {len(chunks)}")

    print(
        "Generating Hugging Face embeddings and "
        "creating FAISS vector database..."
    )

    vector_store = create_vector_store(chunks)

    print("Vector database successfully built and persisted to disk.\n")

    return vector_store


def get_vector_store():
    vector_store = load_vector_store()

    if vector_store is not None:
        print("Existing persisted FAISS vector database loaded from disk.")
        return vector_store

    print(
        "No existing vector database found. "
        "Building initial knowledge base..."
    )

    return build_knowledge_base()


def process_query(retriever, question: str, memory=None):
    conversation_summary = memory.get_summary() if memory else ""

    # Handle clearly ambiguous questions before retrieval.
    normalized_question = " ".join(
        question.strip().lower().split()
    )

    if normalized_question.rstrip("?").strip() in {
        "what is the policy",
        "tell me the policy",
        "what policy",
        "which policy"
    }:
        clarification = (
            "Could you please specify which policy you mean, "
            "such as the leave policy, IT policy, or another policy?"
        )

        if memory:
            memory.save_context(
                question,
                clarification
            )

        return clarification

    # Include previous conversation context for follow-up questions.
    retrieval_query = question

    if conversation_summary:
        retrieval_query = (
            f"Previous conversation:\n{conversation_summary}\n\n"
            f"Current question:\n{question}"
        )

    results = retriever.invoke(retrieval_query)

    if not results:
        fallback_msg = (
            "I could not find sufficient information in the indexed "
            "knowledge base to answer this question."
        )

        if memory:
            memory.save_context(
                question,
                fallback_msg
            )

        return fallback_msg

    retrieved_docs = results

    answer = generate_answer(
        documents=retrieved_docs,
        question=question,
        conversation_summary=conversation_summary
    )

    if memory:
        memory.save_context(
            question,
            answer
        )

    return answer


def main():
    print("=" * 60)
    print("       Enterprise Knowledge Assistant (RAG Pipeline)")
    print("=" * 60)

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set in environment variables.")
        sys.exit(1)

    vector_store = get_vector_store()

    retriever = create_retriever(
        vector_store,
        k=TOP_K,
        threshold=RELEVANCE_THRESHOLD
    )

    memory = create_memory()

    print("\nAssistant initialized and ready.")
    print("Commands:")
    print("  'exit'    - Quit application")
    print("  'rebuild' - Re-ingest documents and rebuild vector store\n")

    while True:
        try:
            question = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if question.lower() == "rebuild":
            vector_store = build_knowledge_base()

            retriever = create_retriever(
                vector_store,
                k=TOP_K,
                threshold=RELEVANCE_THRESHOLD
            )

            memory.clear()

            print("Knowledge base rebuilt and memory reset.")
            continue

        response = process_query(
            retriever,
            question,
            memory
        )

        print("\nAssistant:")
        print(response)


if __name__ == "__main__":
    main()