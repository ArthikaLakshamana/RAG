from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")

        return chunks

    def embed_documents(self, documents: List[Any]) -> np.ndarray:
        """
        Split documents into chunks and generate embeddings.
        """

        chunks = self.chunk_documents(documents)

        embeddings = self.model.encode(
            [chunk.page_content for chunk in chunks],
            show_progress_bar=True
        )

        print(f"[INFO] Created embeddings for {len(embeddings)} chunks.")

        return embeddings

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Generate embeddings for already chunked documents.
        """

        texts = [chunk.page_content for chunk in chunks]

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        return embeddings


if __name__ == "__main__":
    docs = load_all_documents()

    pipeline = EmbeddingPipeline()

    chunks = pipeline.chunk_documents(docs)

    embeddings = pipeline.embed_chunks(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")