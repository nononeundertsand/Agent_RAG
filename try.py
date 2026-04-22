from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://287dba7c-88ff-4e65-8fb9-6d0b2af5ff82.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-a3rJYYRZNeo07kENdn6mWoy_Ys8jR_KwrJ056LWe3k"
)

print(client.get_collections())

