import chromadb
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")
client=chromadb.PersistentClient(path="./chroma_db")

def embed_and_store(chunks,collection_name="resume"):
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection=client.create_collection(collection_name)
    embeddings=model.encode(chunks).tolist()


    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunks_{i}" for i in range(len(chunks))]

    )
    print(f"stored {len(chunks)} chunks in ChromaDB")
    return collection


