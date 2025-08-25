import chromadb
import traceback
from typing import List
from config import CHROMA_DB_PATH, COHERE_API_KEY
from langchain_cohere import CohereRerank
from services.embedding import NvidiaEmbeddingFunction
from langchain_core.documents import Document



class Retriever:
    def __init__(self, top_k: int = 5, initial_k: int = 20):
        self.top_k = top_k
        self.initial_k = initial_k
        self.client = chromadb.PersistentClient(path = CHROMA_DB_PATH)

        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not set in environment variables.")
        self.reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0", top_n=self.top_k )

    def retrieve(self, query: str, collection_name: str) -> List[Document]:

        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=NvidiaEmbeddingFunction()
            )

            if collection.count() == 0:
                print(f"[Retriever] Collection '{collection_name}' is empty.")
                return []
            
            results = collection.query(
                query_texts=[query],
                n_results=self.initial_k
            )

            initial_docs_content = results['documents'][0]
            if not initial_docs_content:
                print(f"[Retriever] No documents found for the query in collection '{collection_name}'.")
                return []
            langchain_docs = [Document(page_content=doc) for doc in initial_docs_content]
            reranked_docs = self.reranker.compress_documents(
                documents=langchain_docs,
                query=query
            )
            print(f"[Retriever] Reranking complete. Returning {len(reranked_docs)} documents.")
            return reranked_docs
        
        except Exception as e:
            print(f"Error occurred while retrieving documents: {e}")
            return []