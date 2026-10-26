from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

client = QdrantClient(url="http://127.0.0.1:6333")  # bez klucza API

COLLECTION = "test"

# utwórz kolekcję od nowa
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)

# dodaj kilka punktów
client.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(id=1, vector=[0.9, 0.1, 0.0, 0.0], payload={"nazwa": "kot"}),
        PointStruct(id=2, vector=[0.8, 0.2, 0.1, 0.0], payload={"nazwa": "pies"}),
        PointStruct(id=3, vector=[0.0, 0.1, 0.9, 0.8], payload={"nazwa": "samochód"}),
    ],
)

# wyszukaj najbliższe do wektora podobnego do "kot"
wynik = client.query_points(
    collection_name=COLLECTION,
    query=[0.85, 0.15, 0.0, 0.0],
    limit=2,
)

for p in wynik.points:
    print(p.id, p.payload, round(p.score, 4))

print("OK, Qdrant działa")