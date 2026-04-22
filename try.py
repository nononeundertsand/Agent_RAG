from qdrant_client import QdrantClient

client = QdrantClient(
    url="",
    api_key=""
)

print(client.get_collections())

