from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore
from core.llm import LLM
from core.rag import RAG


def main():
    embed = EmbeddingModel()
    store = VectorStore()
    llm = LLM()

    rag = RAG(embed, store, llm)

    while True:
        q = input("Вы: ")
        answer, sources = rag.ask(q)

        print("\nМодель:", answer)
        print("Источники:", ", ".join(sources))
        print("-" * 50)


if __name__ == "__main__":
    main()
