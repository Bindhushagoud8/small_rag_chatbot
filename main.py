import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from sentence_transformers import SentenceTransformer

# Load API key
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Read document
with open("docs.txt", "r") as f:
    text = f.read()

# Split into chunks
chunks = text.split("\n")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()
question = input("Ask a question: ")

question_embedding = model.encode(question).tolist()
# Setup Qdrant (memory database)
qdrant = QdrantClient(":memory:")

qdrant.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
)

# Store embeddings
points = []
for i, chunk in enumerate(chunks):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": chunk},
        )
    )

qdrant.upsert(collection_name="docs", points=points)

# Ask user question
question = input("Ask a question: ")

question_embedding = model.encode(question).tolist()

# Search in Qdrant
search_result = qdrant.query_points(
    collection_name="docs",
    query=question_embedding,
    limit=1
)

context = search_result.points[0].payload["text"]
# Send to LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "Answer using the given context only."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
    ]
)

print("\nAnswer:", response.choices[0].message.content)