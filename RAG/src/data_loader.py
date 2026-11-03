from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory.

    Supported formats:
    - PDF
    - TXT
    - CSV
    - Excel (.xlsx, .xls)
    - Word (.docx)
    - JSON
    """

    data_path = Path(data_dir).resolve()

    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    # ---------------- PDF ----------------
    pdf_files = list(data_path.glob("**/*.pdf"))

    print(f"[DEBUG] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            print(f"[DEBUG] Loaded {len(docs)} pages")

            documents.extend(docs)

        except Exception as e:
            print(f"[ERROR] Failed to load {pdf_file}: {e}")

    print(f"[INFO] Total documents loaded: {len(documents)}")

    return documents