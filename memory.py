import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


class ConversationSummaryMemory:
    def __init__(self, llm=None):
        self.llm = llm or ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b"
            ),
            temperature=0
        )

        self.messages = []

        self.summary = ""
        self.summary_prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "You are a conversation summarizer. "
                        "Update the existing conversation summary "
                        "using the latest user question and assistant answer. "
                        "Do not invent information."
                    )
                ),
                (
                    "human",
                    """
Previous Summary:
{summary}

Latest User Question:
{user_input}

Latest Assistant Answer:
{assistant_output}

Create a concise updated summary in 2-4 sentences.

Preserve important:
- topics
- policies
- entities
- user questions
- facts needed for future follow-up questions

Return ONLY the updated summary.
"""
                )
            ]
        )

        self.summary_chain = (
            self.summary_prompt
            | self.llm
            | StrOutputParser()
        )

    def get_summary(self) -> str:
        
        return self.summary.strip()

    def load_memory_variables(self):
        
        return {
            "history": self.get_summary()
        }

    def save_context(
        self,
        user_input: str,
        assistant_output: str
    ):
        self.messages.append(
            HumanMessage(content=user_input)
        )

        self.messages.append(
            AIMessage(content=assistant_output)
        )

        try:
            updated_summary = self.summary_chain.invoke(
                {
                    "summary": self.summary or "None",
                    "user_input": user_input,
                    "assistant_output": assistant_output
                }
            )

            self.summary = updated_summary.strip()

        except Exception as e:
            print(
                f"Memory summarization error: {e}"
            )

            fallback_summary = (
                f"User asked: {user_input}. "
                f"Assistant answered: {assistant_output[:200]}"
            )

            if self.summary:
                self.summary = (
                    f"{self.summary} {fallback_summary}"
                )
            else:
                self.summary = fallback_summary

    def clear(self):
        
        self.messages.clear()
        self.summary = ""


def create_memory():
    
    return ConversationSummaryMemory()
