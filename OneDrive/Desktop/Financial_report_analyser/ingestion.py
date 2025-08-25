import os 
from services.document_processor import DocumentProcessor

PDF_SOURCE = "C:\\Users\\sumed\\OneDrive\\Desktop\\Financial_report_analyser\\data"

COLLECTION_NAME = "financial_reports"
def run_ingestion():
    processor = DocumentProcessor()

    try:
        files_in_folder = os.listdir(PDF_SOURCE)
    except FileNotFoundError:
        print(f"[Ingestion] PDF source folder not found: {PDF_SOURCE}")
        return
    
    pdf_files = [f for f in files_in_folder if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"[Ingestion] No PDF files found in {PDF_SOURCE}")
        return
    
    for  file in pdf_files:
        pdf_path = os.path.join(PDF_SOURCE, file)
        print(f"[Ingestion] Processing file: {pdf_path}")
        processor.process_and_store(pdf_path, COLLECTION_NAME)

if __name__ == "__main__":
    if not os.path.exists(PDF_SOURCE):
        print(f"[Ingestion] PDF source folder does not exist: {PDF_SOURCE}")
    else:
        run_ingestion()