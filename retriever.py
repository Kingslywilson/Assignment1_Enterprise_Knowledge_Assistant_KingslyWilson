def search_documents(
    vector_store,
    query: str,
    k: int = 5,
    threshold: float = 0.30
):
    if vector_store is None or not query:
        return []

    try:
        results = vector_store.similarity_search_with_score(
            query,
            k=k
        )
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

    filtered_documents = []

    print("\nRetrieval scores:")

    for document, distance in results:
        distance = float(distance)

        score = 1.0 - (distance / 2.0)

        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        if isinstance(page, int):
            source = f"{source} — Page {page + 1}"

        print(f"- {source}: {score:.4f}")

        if score >= threshold:
            filtered_documents.append((document, score))

    return filtered_documents