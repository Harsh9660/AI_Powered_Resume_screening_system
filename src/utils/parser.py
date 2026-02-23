from pdfminer.high_level import extract_text
import io

def parse_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    Handles: path strings, raw bytes, and FastAPI UploadFile objects.
    """
    try:
        if isinstance(pdf_file, str):
            # File path
            text = extract_text(pdf_file)
        elif isinstance(pdf_file, bytes):
            text = extract_text(io.BytesIO(pdf_file))
        else:
            # FastAPI UploadFile — use the underlying SpooledTemporaryFile
            # .read() is a coroutine on UploadFile; access .file for sync read
            raw = pdf_file.file.read()
            pdf_file.file.seek(0)  # reset for any future reads
            text = extract_text(io.BytesIO(raw))

        text = text.strip()
        if not text:
            print(f"Warning: extracted empty text from {getattr(pdf_file, 'filename', 'unknown')}")
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return ""

if __name__ == "__main__":
  print("PDF Parser Utility ready.")
