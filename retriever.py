import math
from typing import Any

from pydantic import ConfigDict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ThresholdRetriever(BaseRetriever):

    vector_store: Any
    k: int = 4
    threshold: float = 0.35

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if self.vector_store is None or not query.strip():
            return []

        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=self.k
            )
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

        filtered_documents = []

        print("\nRetrieval scores:")

        for document, distance in results:
            distance = float(distance)
            score = 1.0 - (math.sqrt(max(distance, 0.0)) / 2.0)
            score = max(0.0, min(1.0, score))

            source = document.metadata.get(
                "source",
                "Unknown source"
            )

            page = document.metadata.get("page")

            if isinstance(page, int):
                source = f"{source} — Page {page + 1}"

            print(f"- {source}: {score:.4f}")

            if score >= self.threshold:
                filtered_documents.append(document)

        return filtered_documents


def create_retriever(
    vector_store,
    k: int = 4,
    threshold: float = 0.35
):

    return ThresholdRetriever(
        vector_store=vector_store,
        k=k,
        threshold=threshold
    )
