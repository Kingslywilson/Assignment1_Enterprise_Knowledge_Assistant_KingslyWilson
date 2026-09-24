# Technical Strategy & Architecture Documentation

## Project Overview
This document details the architectural strategy, component design, experimentation, and design decisions behind the **Enterprise Knowledge Assistant** — an Advanced Retrieval-Augmented Generation (RAG) system built using Python, LangChain, FAISS, and Groq LLMs.

---

## 1. Document Ingestion & Multi-Source Loading

### LangChain Loaders Used
The application ingests enterprise knowledge across three distinct format types using standard LangChain document loaders:

1. **PDF Documents**: Implemented using `PyPDFLoader` (`langchain_community.document_loaders.PyPDFLoader`).
   - Extracts page-by-page document text.
   - Retains exact source metadata: document filename (e.g., `employee_handbook.pdf`) and 0-indexed page number (e.g., `page: 0` mapped to human-readable Page 1).
2. **Text Files**: Implemented using `TextLoader` (`langchain_community.document_loaders.TextLoader`) with UTF-8 encoding.
   - Loads structured `.txt` files (e.g., `it_policy.txt`).
   - Retains source metadata: text filename (e.g., `it_policy.txt`).
3. **Public Web Pages**: Implemented using `WebBaseLoader` (`langchain_community.document_loaders.WebBaseLoader`).
   - Fetches and parses public HTML web content (e.g., `https://www.python.org/about/gettingstarted/`).
   - Retains source metadata: full target URL.

### Metadata Preservation
Every document loader standardizes metadata attributes prior to downstream text processing:
```python
{
    "source": "employee_handbook.pdf",
    "page": 0
}
```
This guarantees complete traceability from raw document sources down to retrieved vectors and final grounded answer citations.

---

## 2. Preprocessing & Content Cleaning

### Cleaning Pipeline
Raw text from PDFs, text files, and web pages often contains formatting artifacts, excess whitespace, broken line breaks, and boilerplates. `preprocess.py` implements regular expression sanitization:
- **Excess Whitespace**: Multiple horizontal spaces/tabs are collapsed to single spaces (`re.sub(r"[ \t]+", " ", text)`).
- **Line Break Normalization**: Three or more consecutive newline characters are truncated to two newlines (`\n\n`) to preserve paragraph structure without inflating token count.
- **Empty / Garbage Document Filtering**: Documents stripped of text or containing fewer than 10 characters are automatically discarded.

---

## 3. Recursive Text Splitting

### Splitting Strategy
Text splitting is handled via `RecursiveCharacterTextSplitter` from `langchain_text_splitters`.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)
```

### Parameter Selection & Rationale
- **Chunk Size (`800` characters)**: Selected after evaluating chunk sizes ranging from 400 to 1500 characters. 800 characters (~150–200 words) provides sufficient context to capture complete policy rules or procedure steps without diluting vector representations with unrelated subjects.
- **Chunk Overlap (`150` characters)**: Ensures critical boundary context (such as policy conditions spanning sentence boundaries) is not lost across chunk cuts.
- **Metadata Inheritance**: Every generated chunk inherits the exact `source` and `page` metadata of its parent document.

---

## 4. Embeddings Model Selection

### Provider & Model
- **Provider**: Hugging Face via `HuggingFaceEmbeddings` (`langchain_huggingface`).
- **Model Name**: `sentence-transformers/all-MiniLM-L6-v2`.

### Rationale
1. **Semantic Accuracy**: Outputs 384-dimensional dense vectors fine-tuned specifically for semantic search and sentence similarity tasks.
2. **Normalized Embeddings**: Initialized with `encode_kwargs={"normalize_embeddings": True}`, ensuring vector dot-products equal cosine similarity.
3. **Efficiency & Local Execution**: Runs locally on CPU with minimal latency (~20ms per query batch), eliminating third-party embedding API rate limits, costs, and external dependencies.

---

## 5. Vector Database & Persistence

### Vector Store Selection
- **Database**: `FAISS` (Facebook AI Similarity Search) via `langchain_community.vectorstores.FAISS`.

### Persistence Architecture
- The vector store is persisted locally in the `vector_store/` directory (`index.faiss` and `index.pkl`).
- **Cross-Run Reusability**: Upon application launch, `vector_store.load_vector_store()` inspects disk state. If the persisted index exists, it reloads instantly without re-embedding or re-parsing raw documents.
- **Rebuild Mechanism**: The application CLI provides a `rebuild` command allowing administrators to force re-indexing when source files are updated.

---

## 6. Semantic Retrieval & Similarity Thresholding

### Retrieval Pipeline
`retriever.py` implements top-k search with L2-distance-to-cosine-relevance score mapping and strict thresholding:

```python
# L2 Distance mapped to normalized similarity score in [0.0, 1.0]
score = max(0.0, 1.0 - (float(l2_distance) / 2.0))
```

### Parameters
- **Top-k (`k=4`)**: Retrieves up to 4 candidate chunks per user query. `k=4` balances context richness with LLM context window constraints.
- **Relevance Threshold (`0.35` / `0.40`)**: Candidate chunks scoring below the threshold are filtered out.
- **Fallback Triggering**: If no chunks satisfy the similarity threshold, the retriever returns an empty set, instantly triggering the grounded fallback response without invoking ungrounded LLM inference.

---

## 7. Custom Prompt Engineering & Grounding

### Prompt Architecture (`prompts/rag_prompt.txt`)
The RAG prompt enforces strict corporate assistant persona rules:
1. **Strict Grounding**: The LLM is restricted to retrieved `DOCUMENT CONTEXT`. Outside knowledge is explicitly forbidden for factual queries.
2. **Exact Fallback Response**: If retrieved context is empty or insufficient, the LLM must return:
   `I could not find sufficient information in the indexed knowledge base to answer this question.`
3. **Source Citations**: Every factual answer must end with a `Sources:` block formatted as:
   - PDF: `- <filename> — Page <page_number>`
   - Text: `- <filename>`
   - Web: `- <URL>`
4. **Conflicting Information Handling**: If multiple sources disagree (e.g., notice period in Handbook vs. HR Policy), the LLM must highlight the disagreement, present both claims, and cite both sources.
5. **Ambiguous Query Handling**: If the query is vague (e.g., "What is the policy?"), the LLM asks a clear clarifying question.

---

## 8. Conversation Summary Memory

### Architecture (`memory.py`)
- Implements `ConversationSummaryMemory` using Groq LLM (`openai/gpt-oss-120b`).
- Maintains a running summary of past turns to resolve references in multi-turn dialogues (e.g., Turn 1: "Annual leave policy", Turn 2: "What about during probation?").
- **Decoupled Truth**: The conversation summary is provided *only* to interpret user intent in follow-up queries. Factual answers remain strictly grounded in retrieved vector chunks.
