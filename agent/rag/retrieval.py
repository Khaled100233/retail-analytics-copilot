"""
RAG Retrieval System using TF-IDF
Chunks documents and retrieves relevant passages based on query similarity
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentChunk:
    """Represents a chunk of text from a document"""
    def __init__(self, chunk_id: str, content: str, source: str, metadata: Dict = None):
        self.id = chunk_id
        self.content = content
        self.source = source
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata
        }


class DocumentRetriever:
    """TF-IDF based document retriever"""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        self.tfidf_matrix = None
        
        # Load and chunk documents
        self._load_documents()
        
    def _load_documents(self):
        """Load all markdown files from docs directory and chunk them"""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        
        for file_path in self.docs_dir.glob("*.md"):
            self._chunk_document(file_path)
        
        if not self.chunks:
            raise ValueError("No documents found to index")
        
        # Fit TF-IDF vectorizer on all chunks
        chunk_texts = [chunk.content for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        
        print(f"Loaded {len(self.chunks)} chunks from {len(list(self.docs_dir.glob('*.md')))} documents")
    
    def _chunk_document(self, file_path: Path):
        """Chunk a document into paragraphs"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines (paragraphs) or headers
        paragraphs = []
        current_para = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
            else:
                current_para.append(line)
        
        # Add last paragraph
        if current_para:
            paragraphs.append('\n'.join(current_para))
        
        # Create chunks
        source_name = file_path.stem  # filename without extension
        for idx, para in enumerate(paragraphs):
            if para.strip():  # Skip empty paragraphs
                chunk_id = f"{source_name}::chunk{idx}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=para,
                    source=source_name,
                    metadata={"file": str(file_path)}
                )
                self.chunks.append(chunk)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            
        Returns:
            List of dicts with keys: id, content, source, score
        """
        if not self.chunks:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "score": float(similarities[idx])
            })
        
        return results
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return all chunks (useful for debugging)"""
        return [chunk.to_dict() for chunk in self.chunks]


# Singleton instance
_retriever_instance = None


def get_retriever(docs_dir: str = "docs") -> DocumentRetriever:
    """Get or create the global retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DocumentRetriever(docs_dir)
    return _retriever_instance


if __name__ == "__main__":
    # Test the retriever
    print("Testing Document Retriever...")
    
    retriever = get_retriever()
    
    # Test query
    test_queries = [
        "return policy for beverages",
        "summer beverages campaign dates",
        "average order value formula"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        results = retriever.retrieve(query, top_k=2)
        for result in results:
            print(f"  [{result['id']}] (score: {result['score']:.3f})")
            print(f"  {result['content'][:100]}...")
            print()