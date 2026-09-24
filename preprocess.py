import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def preprocess_documents(documents):
    cleaned_documents = []

    for document in documents:
        cleaned_content = clean_text(document.page_content)

        if not cleaned_content or len(cleaned_content) < 10:
            continue

        document.page_content = cleaned_content
        cleaned_documents.append(document)

    return cleaned_documents


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)