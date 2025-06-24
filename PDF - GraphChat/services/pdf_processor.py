from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(pdf_files):
    """
    Extracts all text content from a list of uploaded PDF files.
    """
    text = ""
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between page texts
            print(f"DEBUG (pdf_processor.py): Extracted text from {pdf_file.filename}")
        except Exception as e:
            print(f"ERROR (pdf_processor.py): Failed to extract text from {pdf_file.filename}: {e}")
            # Optionally, you might want to raise an exception or store error for the specific file
            pass # Continue processing other files if one fails
    return text

def chunk_text(text):
    """
    Splits a large text into smaller, overlapping chunks suitable for LLM processing.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_text(text)
    print(f"DEBUG (pdf_processor.py): Text chunked into {len(chunks)} pieces.")
    return chunks