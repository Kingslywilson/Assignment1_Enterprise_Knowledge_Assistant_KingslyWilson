import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)


def load_pdf_documents(pdf_directory: str = "data/pdfs"):
    documents = []

    if not os.path.exists(pdf_directory):
        return documents

    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)

            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path)

                documents.extend(docs)
            except Exception as e:
                print(f"Error loading PDF {filename}: {e}")

    return documents


def load_text_documents(text_directory: str = "data/text"):
    documents = []

    if not os.path.exists(text_directory):
        return documents

    for filename in os.listdir(text_directory):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(text_directory, filename)

            try:
                loader = TextLoader(
                    file_path,
                    encoding="utf-8"
                )

                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path)

                documents.extend(docs)
            except Exception as e:
                print(f"Error loading text file {filename}: {e}")

    return documents


def load_web_documents(url_file: str = "web_sources.txt"):
    documents = []

    if not os.path.exists(url_file):
        if os.path.exists("websource.txt"):
            url_file = "websource.txt"
        else:
            return documents

    with open(url_file, "r", encoding="utf-8") as file:
        urls = [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = url

            documents.extend(docs)
        except Exception as e:
            print(f"Error loading Web URL {url}: {e}")

    return documents


def load_all_documents(
    pdf_directory: str = "data/pdfs",
    text_directory: str = "data/text",
    url_file: str = "web_sources.txt"
):
    pdf_documents = load_pdf_documents(pdf_directory)
    text_documents = load_text_documents(text_directory)
    web_documents = load_web_documents(url_file)

    return pdf_documents + text_documents + web_documents