import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def load_rag_prompt():
    prompt_path = os.path.join("prompts", "rag_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def create_llm():\
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
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

    for idx, document in enumerate(documents, 1):
        metadata = document.metadata
        source = metadata.get("source", "Unknown source")

        if "page" in metadata:
            # Display 1-indexed page numbers for human readability
            page_num = metadata["page"] + 1 if isinstance(metadata["page"], int) else metadata["page"]
            source_header = f"[Source: {source} — Page {page_num}]"
        else:
            source_header = f"[Source: {source}]"

        formatted.append(
            f"{source_header}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted)


def generate_answer(
    documents,
    question,
    conversation_summary=""
):

    if not documents:
        return "I could not find sufficient information in the indexed knowledge base to answer this question."

    llm = create_llm()
    prompt = create_rag_prompt()

    context = format_documents(documents)

    formatted_prompt = prompt.format(
        context=context,
        conversation_summary=conversation_summary if conversation_summary else "None",
        question=question
    )

    try:
        response = llm.invoke(formatted_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating answer from LLM: {e}")
        return "I could not find sufficient information in the indexed knowledge base to answer this question."