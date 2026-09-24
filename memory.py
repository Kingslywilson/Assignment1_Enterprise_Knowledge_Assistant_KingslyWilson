import os
from langchain_groq import ChatGroq


class ConversationSummaryMemory:

    def __init__(self, llm=None):
        self.llm = llm or ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0
        )
        self.summary = ""
        self.history = []

    def get_summary(self) -> str:
        return self.summary.strip()

    def load_memory_variables(self):
        return {"history": self.get_summary()}

    def save_context(self, user_input: str, assistant_output: str):
        self.history.append({"user": user_input, "assistant": assistant_output})

        prompt = (
            f"You are a conversation summarizer. Update the running conversation summary given the latest interaction.\n\n"
            f"Current Summary:\n{self.summary if self.summary else 'None'}\n\n"
            f"New User Question:\n{user_input}\n\n"
            f"New Assistant Answer:\n{assistant_output}\n\n"
            f"Write an updated concise summary (2-4 sentences max) capturing the key entities, topics, policies, or questions discussed. "
            f"Return ONLY the updated summary text without any surrounding quotes or preamble."
        )

        try:
            response = self.llm.invoke(prompt)
            self.summary = response.content.strip()
        except Exception as e:
            self.summary += f"\nUser asked: {user_input}. Assistant replied: {assistant_output[:100]}..."

    def clear(self):
        self.summary = ""
        self.history = []


def create_memory():
    return ConversationSummaryMemory()