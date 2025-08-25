import chromadb
from typing import List
from config import CHROMA_DB_PATH
from services.embedding import NvidiaEmbeddingFunction
import traceback

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_function = NvidiaEmbeddingFunction()

    def get_or_create_collection(self, collection_name: str):
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, collection_name:str, documents: List[str], metadatas: List[dict], ids: List[str]):
        if not (len(documents) == len(metadatas) == len(ids)):
            print(f"[VectorStore Error] Mismatched lengths: documents({len(documents)}), metadatas({len(metadatas)}), ids({len(ids)})")
            return
        
        try:
            collection = self.get_or_create_collection(collection_name)
            batch_size=100
            for i in range(0, len(documents), batch_size):
                collection.add(
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
                print(f"[VectorStore] Added batch {i//batch_size + 1} to collection '{collection_name}'")
            
            print(f"[VectorStore] Successfully added {len(documents)} documents to '{collection_name}'")

        except Exception as e:
            print(f"[VectorStore Error] {e}")
            traceback.print_exc()

    def count_documents(self, collection_name: str) -> int:
        try: 
            collection = self.get_or_create_collection(collection_name)
            return collection.count()
        except Exception as e:
            print(f"Error counting documents: {e}")
            traceback.print_exc()
            return 0  