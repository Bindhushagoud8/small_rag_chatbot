import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


# ==============================
# 1️⃣ Load API Key
# ==============================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# ==============================
# 2️⃣ Load Document
with open("docs.txt", "r") as f:
    text = f.read()

chunks = []

for line in text.splitlines():
    cleaned = line.strip()
    if cleaned:
        chunks.append(cleaned)

print("Number of chunks:", len(chunks))

print("Chunks list:")
for i, c in enumerate(chunks):
    print(i, repr(c))

print("Number of chunks:", len(chunks))


# ==============================
# 3️⃣ Load Embedding Model
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()
print("Chunks:", chunks)
print("Number of chunks:", len(chunks))

# ==============================
# 4️⃣ Setup Qdrant (local storage instead of memory)
# ==============================
qdrant = QdrantClient(path="qdrant_storage")

collection_name = "docs"

# Create collection (only if not exists)
collection_name = "docs"

if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name)

qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=len(embeddings[0]),
        distance=Distance.COSINE,
    ),
)

# ==============================
# 5️⃣ Store Embeddings
# ==============================
points = []

for i, chunk in enumerate(chunks):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": chunk},
        )
    )

qdrant.upsert(collection_name=collection_name, points=points)


# ==============================
# 6️⃣ Ask Questions Loop
# ==============================
while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # Embed question
    question_embedding = model.encode(question).tolist()

    # Search
    search_result = qdrant.query_points(
        collection_name=collection_name,
        query=question_embedding,
        limit=3
    )

    if not search_result.points:
        print("No relevant context found.")
        continue

    contexts = [point.payload["text"] for point in search_result.points]
    context = "\n".join(contexts)

    print("\nRetrieved Context:\n", context)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "Answer only using the given context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content
        print("\nAnswer:", answer)

    except Exception as e:
        print("LLM Error:", e)