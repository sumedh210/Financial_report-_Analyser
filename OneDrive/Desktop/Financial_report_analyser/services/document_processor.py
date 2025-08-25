import os
import traceback
from typing import List

from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from services.vector_store import VectorStore


class DocumentProcessor:
    def __init__(self):
        self.vector_store = VectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"[DocumentProcessor] Extracting text from: {pdf_path}")
        
        try:
            elements = partition_pdf(pdf_path)
            return "\n\n".join([str(el) for el in elements])
        except Exception as e:
            print(f"[DocumentProcessor Error] {e}")
            traceback.print_exc()
            return ""
  
    def process_and_store(self, pdf_path: str, collection_name:str):
        full_text = self.extract_text_from_pdf(pdf_path)
        if not full_text:
            print(f"[DocumentProcessor] No text extracted from {pdf_path}")
            return

        chunks = self.text_splitter.split_text(full_text)
        if not chunks:
            print(f"[DocumentProcessor] No chunks created from text in {pdf_path}")
            return
        print(f"[DocumentProcessor] Created {len(chunks)} chunks.")

        source_filename = os.path.basename(pdf_path)
        metadatas = [{"source": source_filename} for _ in chunks]
        ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]



        self.vector_store.add_documents(
            collection_name=collection_name,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[DocumentProcessor] Processed and stored {len(chunks)} chunks from {pdf_path} into collection '{collection_name}'")