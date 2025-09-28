import streamlit as st
import os
from services.document_processor import DocumentProcessor
from services.hyde_generator import HydeGenerator
from services.retriever import Retriever    
from groq import Groq
from config import MODEL_KEY

# --- Configuration ---
COLLECTION_NAME = "financial_reports"
PDF_SOURCE_FOLDER = "C:\\Users\\sumed\\OneDrive\\Desktop\\Financial_report_analyser\\data"
st.set_page_config(page_title="Financial Report Analyzer", page_icon="📈")

# --- Initialize Service Classes ---
@st.cache_resource
def get_services():
    print("Initializing services...")
    hyde = HydeGenerator(model="llama-3.1-8b-instant")
    retriever = Retriever(top_k=5)
    groq_client = Groq(api_key=MODEL_KEY)
    processor = DocumentProcessor()
    return hyde, retriever, groq_client, processor

hyde_generator, retriever, groq_client, processor = get_services()

# --- Main Application Logic ---
st.title("📈 Advanced Financial Report Analyzer")
st.write("Ask a question about your financial documents, and the AI will find the answer using an advanced RAG pipeline (HyDE + Reranker).")

# --- Sidebar for File Uploading ---
st.sidebar.header("Upload a Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    save_path = os.path.join(PDF_SOURCE_FOLDER, uploaded_file.name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully.")
    
    with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
        processor.process_and_store(save_path, COLLECTION_NAME)
    
    st.sidebar.success("Ingestion complete! You can now ask questions about the new document.")

# --- Main Query Interface ---
query = st.text_input("Ask your question:", placeholder="e.g., What was the total revenue last quarter?")

if query:
    st.write("---")
    
    with st.spinner("Step 1/4: Generating hypothetical answer (HyDE)..."):
        hypothetical_doc = hyde_generator.generate_hypothetical_answer(query)
    
    st.info("🔹 **Hypothetical Answer (for search):**")
    st.write(hypothetical_doc)

    with st.spinner("Step 2/4: Retrieving and reranking relevant documents..."):
        retrieved_docs = retriever.retrieve(hypothetical_doc, COLLECTION_NAME)

    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # --- THIS IS THE CORRECTED PROMPT ---
        final_prompt = (
            "You are a financial analyst assistant. Answer the user's question based only on the following context.\n"
            "If the answer is not in the context, state that you cannot find the answer in the provided documents.\n\n"
            "--- CONTEXT ---\n"
            f"{context}\n\n"
            "--- QUESTION ---\n"
            f" {query}\n\n" 
            "--- ANSWER ---"
        )
        
        with st.spinner("Step 3/4: Generating final answer..."):
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.1
            )
            final_answer = response.choices[0].message.content

        with st.spinner("Step 4/4: Finalizing response..."):
            st.success("✅ **Final Answer:**")
            st.write(final_answer)

            with st.expander("📚 Show Sources"):
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")
    else:
        st.error("Could not retrieve any relevant documents from the database. Please try another query or check if documents have been ingested.")