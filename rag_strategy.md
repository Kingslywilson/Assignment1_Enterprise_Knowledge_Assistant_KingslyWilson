# Technical Strategy & Architecture Documentation

## Project Overview

The **Enterprise Knowledge Assistant** is an Advanced Retrieval-Augmented Generation (RAG) system built using Python, LangChain, FAISS, Hugging Face embeddings, and Groq LLMs.

The system combines information from PDF files, TXT files, and public web pages into a searchable knowledge base. User questions are processed through semantic retrieval, relevant document chunks are selected using a similarity threshold, and the Groq LLM generates an answer grounded only in the retrieved context.

The system also supports source citations, PDF page references, ambiguous questions, conflicting information, and multi-turn conversations using conversation summary memory.

---

## 1. Document Ingestion & Multi-Source Loading

The application supports three different document source types using LangChain document loaders.

### PDF Documents

PDF files are loaded using:

```python
PyPDFLoader
```

The application reads PDF files from:

```text
data/pdfs/
```

Each PDF page is loaded as a separate document. The filename is stored as the source metadata and the page number provided by the loader is preserved.

Example metadata:

```python
{
    "source": "employee_handbook.pdf",
    "page": 0
}
```

The stored page number is 0-indexed internally and is converted to a human-readable 1-indexed page number when displayed to the user.

### Text Documents

TXT files are loaded using:

```python
TextLoader
```

with UTF-8 encoding.

Text files are read from:

```text
data/text/
```

The filename is stored in the `source` metadata.

Example:

```python
{
    "source": "it_policy.txt"
}
```

### Public Web Pages

Public web pages are loaded using:

```python
WebBaseLoader
```

URLs are read from:

```text
web_sources.txt
```

Each successfully loaded web document stores the original URL as its source metadata.

Example:

```python
{
    "source": "https://www.python.org/about/gettingstarted/"
}
```

### Combined Ingestion

The `load_all_documents()` function combines:

```text
PDF documents
TXT documents
Web documents
```

into a single collection before preprocessing and indexing.

---

## 2. Preprocessing & Content Cleaning

The raw document content is cleaned before being embedded.

The preprocessing pipeline is implemented in `preprocess.py`.

### Cleaning Operations

The system performs the following operations:

### Excess Newline Removal

Three or more consecutive newline characters are reduced to two:

```python
re.sub(r"\n{3,}", "\n\n", text)
```

This helps preserve paragraph structure while removing excessive blank lines.

### Whitespace Normalization

Multiple spaces and tabs are collapsed into a single space:

```python
re.sub(r"[ \t]+", " ", text)
```

### Line Cleanup

Each line is stripped of unnecessary leading and trailing whitespace.

### Empty or Very Short Documents

Documents are discarded when:

* The cleaned content is empty.
* The cleaned content contains fewer than 10 characters.

This prevents meaningless content from entering the retrieval system.

---

## 3. Recursive Text Splitting

After preprocessing, documents are divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

The current configuration is:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)
```

### Chunk Size

The chunk size is set to:

```text
800 characters
```

This provides reasonably sized text units for semantic embedding and retrieval while keeping related information together.

### Chunk Overlap

The overlap is set to:

```text
150 characters
```

The overlap helps preserve contextual information when an important sentence or rule crosses a chunk boundary.

### Separators

The splitter attempts to divide text in the following order:

```text
1. Paragraph break
2. Line break
3. Space
4. Individual characters
```

This allows the splitter to preserve natural text boundaries where possible.

### Metadata Preservation

The `split_documents()` method preserves document metadata when creating chunks.

Therefore, source information such as:

```text
source
page
```

remains available after splitting.

---

## 4. Embeddings Strategy

The system uses Hugging Face embeddings through:

```python
HuggingFaceEmbeddings
```

The configured model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Configuration

The model is loaded using:

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Normalized Embeddings

Embeddings are normalized before storage:

```python
"normalize_embeddings": True
```

This makes the vectors suitable for similarity-based retrieval and supports conversion of FAISS L2 distance into a cosine-style similarity score.

### Local Execution

The embedding model runs locally using the CPU.

This means document embedding does not require an external embedding API.

---

## 5. Vector Store & Persistence

The application uses:

```text
FAISS
```

as the vector database.

FAISS is used to store and search the embedding vectors generated from the document chunks.

### Vector Store Location

The persisted database is stored in:

```text
vector_store/
```

The main files are:

```text
vector_store/index.faiss
vector_store/index.pkl
```

### Creating the Vector Store

The vector store is created using:

```python
FAISS.from_documents(
    documents,
    embeddings
)
```

It is then persisted locally using:

```python
vector_store.save_local(path)
```

### Loading an Existing Vector Store

When the application starts, it first checks whether an existing FAISS index is available.

If the persisted vector store exists, the application loads it instead of rebuilding the entire knowledge base.

This improves startup efficiency.

### Rebuild Command

The application also supports the:

```text
rebuild
```

command.

This command:

1. Loads the latest PDF, TXT, and web documents.
2. Cleans the documents.
3. Splits them into chunks.
4. Generates embeddings.
5. Rebuilds the FAISS index.
6. Persists the updated vector store.
7. Clears conversation memory.

This allows the knowledge base to be refreshed when source content changes.

---

## 6. Semantic Retrieval & Relevance Thresholding

The retrieval component is implemented in:

```text
retriever.py
```

The application uses:

```python
similarity_search_with_score()
```

to retrieve the top matching document chunks.

### Top-k Retrieval

The main application configuration is:

```python
TOP_K = 4
```

Therefore, up to four candidate chunks are retrieved for each query.

### FAISS Distance

FAISS returns an L2 distance for the retrieved vectors.

Because the embedding vectors are normalized, the system converts the L2 distance into a cosine-style similarity score using:

```python
score = 1.0 - (distance / 2.0)
```

The resulting score is used for threshold-based filtering.

The implementation does not explicitly clamp the score to a minimum of `0.0` or a maximum of `1.0`.

### Relevance Threshold

The application uses:

```python
RELEVANCE_THRESHOLD = 0.35
```

Only retrieved chunks whose calculated score is greater than or equal to the configured threshold are passed to the RAG generation stage.

This reduces the chance of using unrelated content.

### Fallback Behavior

When no retrieved document meets the threshold, the retriever returns an empty result.

The application then returns the fallback response:

```text
I could not find sufficient information in the indexed knowledge base to answer this question.
```

This prevents the LLM from answering unsupported questions using general knowledge.

---

## 7. RAG Prompt & Grounding Strategy

The RAG prompt is stored in:

```text
prompts/rag_prompt.txt
```

The prompt is created using LangChain's:

```python
PromptTemplate
```

### Grounding Rules

The prompt instructs the LLM to:

* Use only information contained in the retrieved document context.
* Avoid outside knowledge.
* Avoid guessing or fabricating information.
* Treat the conversation summary only as context for understanding follow-up questions.
* Use the retrieved documents as the factual evidence.

### Fallback

The required fallback response is:

```text
I could not find sufficient information in the indexed knowledge base to answer this question.
```

The prompt instructs the model to return this response when the retrieved context is insufficient, unrelated, or empty.

### Source Citations

Grounded answers include a `Sources:` section.

For PDFs:

```text
- filename.pdf — Page N
```

For TXT files:

```text
- filename.txt
```

For web sources:

```text
- URL
```

Only sources contributing to the answer should be listed.

### Conflicting Information

When multiple retrieved sources disagree, the prompt instructs the LLM to:

1. Clearly identify the conflict.
2. Present the information stated by each source.
3. Avoid silently selecting one source.
4. Cite the relevant sources.

This behavior was tested using conflicting resignation notice information in the Employee Handbook and HR Policy.

### Ambiguous Questions

When the user's question is too vague to identify a specific policy or topic, the system asks the user for clarification.

The application also contains an explicit ambiguity check before retrieval for common questions such as:

```text
What is the policy?
What policy?
Which policy?
Tell me the policy?
```

---

## 8. Conversation Summary Memory

Conversation handling is implemented in:

```text
memory.py
```

The project includes a custom:

```python
ConversationSummaryMemory
```

implementation.

It uses the Groq LLM to maintain a concise running summary of previous interactions.

### Stored Information

The memory maintains:

```text
conversation summary
conversation history
```

The summary captures important entities, topics, policies, and questions from earlier turns.

### Follow-Up Question Handling

When a new question is submitted and a conversation summary exists, the application builds a retrieval query containing:

```text
Previous conversation:
<summary>

Current question:
<question>
```

This allows follow-up questions to be interpreted using the context of earlier conversation turns.

For example:

```text
User:
How much notice is required when resigning?

User:
Which document says that?
```

The second question can use the previous context to retrieve the relevant resignation policy documents.

### Grounding Protection

The conversation summary is not treated as factual evidence.

Only retrieved document chunks are used as the factual source for generated answers.

This prevents conversation memory from becoming an unsupported knowledge source.

---

## 9. End-to-End RAG Flow

The complete processing pipeline is:

```text
PDF / TXT / Web Sources
          ↓
   LangChain Loaders
          ↓
Cleaning & Preprocessing
          ↓
Recursive Character Splitting
          ↓
Hugging Face Embeddings
          ↓
       FAISS Store
          ↓
   Semantic Retrieval
          ↓
Top-k Candidate Documents
          ↓
Relevance Threshold Filtering
          ↓
Retrieved Document Context
          ↓
Conversation Context
          ↓
    Custom RAG Prompt
          ↓
       Groq LLM
          ↓
Grounded Answer + Sources
```

For follow-up questions, conversation summary is also incorporated into the retrieval query.

---

## 10. Failure & Edge Case Handling

The system is designed to handle several common failure scenarios.

### Unavailable Information

When relevant information is not found, the system returns the defined fallback response.

### Low-Relevance Questions

Questions unrelated to the indexed knowledge base are rejected when retrieved similarity scores remain below the configured threshold.

### Ambiguous Questions

Questions that do not clearly identify a topic or policy trigger a clarification request.

### Conflicting Information

When retrieved documents contain contradictory information, the assistant reports both source statements and cites them.

### LLM Generation Errors

If an error occurs during LLM response generation, the application prints the error and returns the standard fallback response.

### Vector Store Loading Errors

If a persisted FAISS index cannot be loaded, the application can fall back to building the knowledge base again.

### Empty User Input

Blank user input is ignored by the interactive application.

### Application Commands

The CLI provides:

```text
exit
rebuild
```

for closing the application and rebuilding the knowledge base.

---

## 11. Design Decisions

### FAISS

FAISS was selected because it provides efficient local vector similarity search and supports persistent vector indexes.

### Hugging Face Embeddings

Hugging Face embeddings were selected so embeddings can be generated locally without requiring a separate embedding API.

### Groq LLM

Groq is used as the LLM provider through LangChain's `ChatGroq` integration.

The configured model is:

```text
openai/gpt-oss-120b
```

### Relevance Threshold

A threshold of:

```text
0.35
```

is used to reduce irrelevant context and provide a fallback when the knowledge base does not contain sufficient information.

### Persistent Knowledge Base

Persisting the FAISS vector store avoids unnecessary re-indexing every time the application starts.

### Conversation Summary

A running summary provides lightweight conversational context while keeping factual grounding restricted to retrieved documents.

---

## 12. Testing Strategy

The system was tested using ten scenarios covering the required RAG behaviors:

1. PDF-based information retrieval.
2. TXT-based information retrieval.
3. Public web information retrieval.
4. Multi-source retrieval.
5. Unavailable information and fallback.
6. Low-relevance query handling.
7. Ambiguous question clarification.
8. Multi-turn conversation handling.
9. Conflicting information handling.
10. Follow-up source identification.

The final test results are documented in:

```text
test_log.md
```

### Final Test Result

```text
10/10 tests passed successfully.
```

The successful tests demonstrate that the system can retrieve information from multiple source types, remain grounded in the indexed knowledge base, handle unsupported questions, preserve conversation context, identify conflicting information, and provide source references.
