from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

## Example usage

if __name__ == "__main__":
    docs=load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    store.build_from_documents(docs)


