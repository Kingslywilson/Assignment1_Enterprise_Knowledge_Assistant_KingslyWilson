Test Log — Enterprise Knowledge Assistant

Project

Assignment: Enterprise Knowledge Assistant using Python & LangChain — Advanced RAG System
Project: Assignment1_Enterprise_Knowledge_Assistant_KingslyWilson

Test Environment

Python: 3.11

Framework: LangChain

LLM: Groq — openai/gpt-oss-120b

Embeddings: Hugging Face

Vector Store: FAISS

Knowledge Sources:

PDF documents

TXT document

Public web page

Retrieval: Semantic similarity search with top-k retrieval and relevance threshold

Conversation: Conversation summary memory

Test Count: 10

Final Result: 10/10 PASS

Knowledge Base Build Test

The knowledge base was rebuilt successfully using the PDF, TXT, and web sources.

Actual Output

--- Knowledge Base Construction ---
Loading documents from PDFs, Text files, and Web sources...
Loaded total raw documents: 8
Cleaning and preprocessing document content...
Cleaned documents count: 8
Splitting text recursively into chunks...
Generated text chunks: 17
Generating Hugging Face embeddings and creating FAISS vector database...
Vector database successfully built and persisted to disk.

Knowledge base rebuilt and memory reset.

Result: PASS

Test 1 — PDF Source: Leave Policy

Question

What is the company's leave policy?

Actual Retrieval

- employee_handbook.pdf — Page 1: 0.4092
- hr_policy.pdf — Page 2: 0.3447
- hr_policy.pdf — Page 1: 0.2281
- employee_handbook.pdf — Page 3: 0.0745

Actual Answer

The Employee Handbook states that employees are entitled to 18 days
of annual leave per year, subject to the company's leave policy.

Source:
employee_handbook.pdf — Page 1

Result

PASS

The answer was grounded in the uploaded PDF and included the source page.

Test 2 — TXT Source: IT Password Reset

Question

How do I reset my company password?

Actual Retrieval

The it_policy.txt document was the highest relevant source.

- it_policy.txt: 0.4979

Actual Answer

The assistant stated that employees should use the self-service password reset portal. The policy specifies a minimum password length of 14 characters and password expiration after 90 days. It also states that IT staff will never ask users for their password.

Source

it_policy.txt

Result

PASS

The answer was grounded in the TXT document and did not rely on outside knowledge.

Test 3 — Web Source: Python Information

Question

What is Python?

Actual Retrieval

- Python.org: 0.5535
- Python.org: 0.4639
- Python.org: 0.4057
- Python.org: 0.3611

Actual Answer

The assistant explained Python using information retrieved from the indexed Python.org web source.

Source

Python.org web source

Result

PASS

The answer was generated from the indexed public web source.

Test 4 — Multi-Source Retrieval: Emergency Leave During Probation

Question

Can an employee take emergency leave during probation?

Actual Retrieval

- hr_policy.pdf — Page 1: 0.5993
- employee_handbook.pdf — Page 1: 0.4501

Actual Answer

The assistant combined information from both sources:

Employees are not eligible for standard annual leave during probation.

Emergency leave requires manager approval.

The emergency leave should be reported within the required time period.

Sources

hr_policy.pdf — Page 1
employee_handbook.pdf — Page 1

Result

PASS

The system successfully retrieved and combined information from multiple sources.

Test 5 — Unavailable Information

Question

Does the company provide a remote work allowance?

Actual Retrieval

The retrieved documents did not contain sufficiently relevant information.

The highest retrieval score was low:

- 0.0710

Actual Answer

I could not find sufficient information in the indexed
knowledge base to answer this question.

Result

PASS

The assistant correctly used the fallback instead of inventing an answer.

Test 6 — Low-Relevance Question

Question

What is the capital of France?

Actual Retrieval

The retrieved results had very low/negative similarity scores and did not provide relevant information.

Actual Answer

I could not find sufficient information in the indexed
knowledge base to answer this question.

Result

PASS

The assistant correctly rejected a question outside the indexed knowledge base instead of using general LLM knowledge.

Test 7 — Ambiguous Question

Question

what is the policy ?

Actual Answer

Could you please specify which policy you mean, such as the leave policy,
IT policy, or another policy?

Result

PASS

The system recognized that the question was ambiguous and requested clarification before performing retrieval.

No retrieval scores were generated for this ambiguous query.

Test 8 — Multi-Turn Conversation

First Question

How many days of annual leave do employees get?

Actual Retrieval

- employee_handbook.pdf — Page 1: 0.6098

Answer

The assistant stated that employees receive 18 days of annual leave.

Follow-Up Question

What about during probation?

Actual Retrieval

- hr_policy.pdf — Page 1: 0.3554

Answer

The assistant explained that employees are not eligible for standard annual leave during probation and provided the relevant emergency-leave information.

Source

hr_policy.pdf — Page 1

Result

PASS

The assistant successfully maintained conversation context and used it to answer the follow-up question.

Test 9 — Conflicting Information

Question

How much notice is required when resigning?

Actual Retrieval

- employee_handbook.pdf — Page 1: 0.4194
- hr_policy.pdf — Page 2: 0.4028

Actual Answer

The indexed sources contain conflicting information regarding
the required resignation notice period.

- The Employee Handbook states that employees must give a
  30-day written notice to their supervisor.
- The HR Policy states that all full-time employees must submit
  a 60-day written notice to HR before leaving.

Sources:
- employee_handbook.pdf — Page 1
- hr_policy.pdf — Page 2

Result

PASS

The assistant correctly identified the conflict and reported both source statements instead of choosing or inventing a single answer.

Test 10 — Follow-Up Source Identification

Previous Question

How much notice is required when resigning?

The assistant identified conflicting information from:

employee_handbook.pdf — Page 1
hr_policy.pdf — Page 2

Follow-Up Question

Which document says that?

Actual Retrieval

- employee_handbook.pdf — Page 1: 0.6527
- hr_policy.pdf — Page 2: 0.6427
- hr_policy.pdf — Page 1: 0.5097
- hr_policy.pdf — Page 3: 0.3514

Actual Answer

- The Employee Handbook states that employees must provide a
  30-day written notice to their supervisor before resigning.
- The HR Policy states that all full-time employees must submit
  a 60-day written notice to HR prior to departure.

Sources:
- employee_handbook.pdf — Page 1
- hr_policy.pdf — Page 2

Result

PASS

The assistant successfully used the previous conversation context to resolve the follow-up question and identify the relevant documents.

Final Test Summary

#

Test Scenario

Result

1

PDF — Leave Policy

PASS

2

TXT — IT Password Reset

PASS

3

Web — Python Information

PASS

4

Multi-Source — Emergency Leave

PASS

5

Unavailable Information

PASS

6

Low-Relevance Question

PASS

7

Ambiguous Question

PASS

8

Multi-Turn Conversation

PASS

9

Conflicting Information

PASS

10

Follow-Up / Source Identification

PASS

Overall Result

10/10 tests passed successfully.

The Enterprise Knowledge Assistant successfully demonstrated:

PDF document retrieval

TXT document retrieval

Public web source retrieval

Multi-source retrieval

Semantic similarity search

Relevance threshold filtering

Grounded responses

Source citations

PDF page citations

Fallback for unavailable information

Low-relevance query handling

Ambiguous-question clarification

Multi-turn conversation handling

Conversation-context retrieval

Conflicting-information detection

Follow-up source identification

Conclusion

The application was tested against all required scenarios and all 10 tests passed successfully. The RAG pipeline is functioning as expected with the configured PDF, TXT, and web knowledge sources.