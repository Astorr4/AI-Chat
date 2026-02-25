from qdrant_client import QdrantClient
from config import QDRANT_PATH, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.collection = COLLECTION_NAME

    def search(self, vector, limit=5, doc_type=None):

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        query_filter = None

        if doc_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=doc_type)
                    )
                ]
            )

        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_vectors=True,
            query_filter=query_filter
        )
        return result.points if result else []
