import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def load_rag_prompt():
    prompt_path = os.path.join(
        "prompts",
        "rag_prompt.txt"
    )

    with open(
        prompt_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


def create_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        ),
        temperature=0
    )


def create_rag_prompt():
    prompt_text = load_rag_prompt()

    return PromptTemplate(
        template=prompt_text,
        input_variables=[
            "context",
            "conversation_summary",
            "question"
        ]
    )


def format_documents(documents):
    formatted = []

    for document in documents:
        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown source"
        )

        if "page" in metadata:
            page = metadata["page"]

            if isinstance(page, int):
                page_num = page + 1
            else:
                page_num = page

            source_header = (
                f"[Source: {source} — Page {page_num}]"
            )
        else:
            source_header = (
                f"[Source: {source}]"
            )

        formatted.append(
            f"{source_header}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted)


def create_rag_chain():

    prompt = create_rag_prompt()
    llm = create_llm()

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def generate_answer(
    documents,
    question,
    conversation_summary=""
):

    if not documents:
        return (
            "I could not find sufficient information in the indexed "
            "knowledge base to answer this question."
        )

    context = format_documents(documents)

    conversation_context = (
        conversation_summary
        if conversation_summary
        else "None"
    )

    try:
        rag_chain = create_rag_chain()

        answer = rag_chain.invoke(
            {
                "context": context,
                "conversation_summary": conversation_context,
                "question": question
            }
        )

        return answer.strip()

    except Exception as e:
        print(
            f"Error generating answer from RAG chain: {e}"
        )

        return (
            "I could not find sufficient information in the indexed "
            "knowledge base to answer this question."
        )

