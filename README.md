Enterprise Knowledge Assistant

An advanced RAG-based Enterprise Knowledge Assistant built using Python, LangChain, Groq, Hugging Face embeddings, and FAISS.

The system allows users to ask questions about information stored in PDF files, TXT files, and public web pages. It retrieves relevant content from the knowledge base and generates grounded answers with source references.

Features
PDF document ingestion using PyPDFLoader
TXT document ingestion using TextLoader
Web page ingestion using WebBaseLoader
Document cleaning and preprocessing
Recursive text splitting using RecursiveCharacterTextSplitter
Hugging Face embeddings
FAISS vector database with persistence
Semantic similarity search
Top-k retrieval with relevance threshold
Grounded answers using a custom RAG prompt
Source and PDF page references
Fallback for unavailable information
Ambiguous question clarification
Multi-source information retrieval
Conflicting information handling
Multi-turn conversation support
Conversation summary memory
Groq LLM using openai/gpt-oss-120b

Project Structure
Assignment1_Enterprise_Knowledge_Assistant_KingslyWilson/
│
├── app.py
├── loaders.py
├── preprocess.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── rag_chain.py
├── memory.py
│
├── prompts/
│   └── rag_prompt.txt
│
├── data/
│   ├── pdfs/
│   │   ├── employee_handbook.pdf
│   │   └── hr_policy.pdf
│   │
│   └── text/
│       └── it_policy.txt
│
├── web_sources.txt
├── vector_store/
│
├── rag_strategy.md
├── test_log.md
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
Requirements
Python 3.11
Groq API key
Internet connection for web-source ingestion and model access
Installation

Create and activate a virtual environment, then install the required packages.

pip install -r requirements.txt
Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
USER_AGENT=EnterpriseKnowledgeAssistant/1.0

Do not include the actual .env file or API key in the submission ZIP.

Running the Application

Start the application with:

python app.py

The application loads the existing FAISS vector database when available.

Available commands:

exit

Closes the application.

rebuild

Re-ingests the PDF, TXT, and web sources and rebuilds the FAISS vector database.

RAG Workflow

The application follows this workflow:

PDF / TXT / Web Sources
          ↓
   Document Loading
          ↓
 Cleaning & Preprocessing
          ↓
 Recursive Text Splitting
          ↓
 Hugging Face Embeddings
          ↓
      FAISS Vector DB
          ↓
  Semantic Similarity Search
          ↓
 Relevance Threshold Filtering
          ↓
   Retrieved Documents
          ↓
      RAG Prompt
          ↓
       Groq LLM
          ↓
 Grounded Answer + Sources
Source Handling

The system preserves document source metadata during ingestion.

For PDF documents, page numbers are preserved and displayed in the final answer.

For web documents, the source URL is retained as metadata.

Conversation Handling

The assistant supports multi-turn conversations using a custom ConversationSummaryMemory-compatible implementation.

Previous conversation context is included when processing follow-up questions so that questions such as:

How much notice is required when resigning?
Which document says that?

can be answered using the context of the previous interaction.

Grounding and Fallback

The assistant is instructed to use only information retrieved from the indexed knowledge base.

When sufficient relevant information cannot be found, the application returns:

I could not find sufficient information in the indexed knowledge base to answer this question.

The application also handles ambiguous questions by requesting clarification before retrieval.

Testing

The application was tested against 10 required scenarios covering:

PDF retrieval
TXT retrieval
Web retrieval
Multi-source retrieval
Unavailable information
Low-relevance queries
Ambiguous questions
Multi-turn conversations
Conflicting information
Follow-up source identification

Final result: 10/10 tests passed successfully.

Detailed test results are available in test_log.md.

Strategy

The RAG architecture and design decisions are documented in:

rag_strategy.md
Submission Notes

Before creating the final ZIP, make sure the following are excluded:

.env
venv/
__pycache__/
.pytest_cache/
IDE configuration files
Temporary files
API keys or other secrets
Unnecessary cache/model files

The final ZIP should be named:

Assignment1_Enterprise_Knowledge_Assistant_KingslyWilson.zip