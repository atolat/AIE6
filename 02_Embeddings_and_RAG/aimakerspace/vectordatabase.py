import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import re
from datetime import datetime


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.metadata = defaultdict(dict)
        # Store document-level metadata separately
        self.document_metadata = {}

    def add_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Add or update document-level metadata."""
        self.document_metadata[doc_id] = metadata

    def insert(
        self, 
        key: str, 
        vector: np.array, 
        chunk_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> None:
        """
        Insert a vector with metadata into the database.
        
        Args:
            key: The text or identifier for this chunk
            vector: The embedding vector
            chunk_metadata: Section-level metadata specific to this chunk
            doc_id: Document identifier to link with document-level metadata
        """
        self.vectors[key] = vector
        
        # Combine document and chunk metadata
        metadata = {}
        if doc_id and doc_id in self.document_metadata:
            metadata.update(self.document_metadata[doc_id])
        if chunk_metadata:
            metadata.update(chunk_metadata)
            
        self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        filter_fn: Callable[[dict], bool] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors with metadata.
        Returns list of tuples (key, score, metadata).
        """
        scores = [
            (key, distance_measure(query_vector, vector), self.metadata[key])
            for key, vector in self.vectors.items()
        ]
        if filter_fn:
            scores = [score for score in scores if filter_fn(score[2])]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        filter_fn: Callable[[dict], bool] = None,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar texts with metadata.
        Returns list of tuples (text, score, metadata) if return_as_text is False,
        otherwise returns list of texts.
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, filter_fn)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict[str, Any]]:
        """Retrieve a vector and its metadata by key."""
        return self.vectors.get(key, None), self.metadata.get(key, {})

    async def abuild_from_list(
        self, list_of_text: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> "VectorDatabase":
        """
        Build the vector database from a list of texts.
        If metadata_list is not provided, metadata will be extracted from each text.
        """
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = None
            if metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]
            self.insert(text, np.array(embedding), metadata)
        return self

    async def abuild_from_chunks(
        self, 
        chunks: List[Tuple[str, Dict[str, Any]]], 
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> "VectorDatabase":
        """
        Build the vector database from a list of text chunks with their metadata.
        
        Args:
            chunks: List of tuples (text, chunk_metadata)
            doc_id: Document identifier
            doc_metadata: Document-level metadata
        """
        if doc_id and doc_metadata:
            self.add_document_metadata(doc_id, doc_metadata)
            
        texts = [chunk[0] for chunk in chunks]
        embeddings = await self.embedding_model.async_get_embeddings(texts)
        
        for (text, chunk_metadata), embedding in zip(chunks, embeddings):
            self.insert(text, np.array(embedding), chunk_metadata, doc_id)
            
        return self


if __name__ == "__main__":
    # Example usage with PMarca blog data
    doc_metadata = {
        "source": "PMarca Blog Archives",
        "author": "Marc Andreessen",
        "date_range": "2007-2009",
        "copyright": "Andreessen Horowitz"
    }
    
    # Example chunks with section metadata
    chunks = [
        (
            "Why not to do a startup...",  # chunk text
            {
                "guide_type": "THE PMARCA GUIDE TO STARTUPS",
                "part_number": "Part 1",
                "section_title": "Why not to do a startup",
                "page_number": 2,
                "topic_category": "STARTUPS"
            }
        ),
        (
            "When the VCs say 'no'...",  # chunk text
            {
                "guide_type": "THE PMARCA GUIDE TO STARTUPS",
                "part_number": "Part 2",
                "section_title": "When the VCs say 'no'",
                "page_number": 10,
                "topic_category": "STARTUPS"
            }
        )
    ]
    
    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_chunks(
        chunks=chunks,
        doc_id="pmarca_blog",
        doc_metadata=doc_metadata
    ))
    
    # Example search with metadata filtering
    def filter_startup_guide(metadata: dict) -> bool:
        return metadata.get("guide_type") == "THE PMARCA GUIDE TO STARTUPS"
    
    results = vector_db.search_by_text(
        "startup funding advice",
        k=2,
        filter_fn=filter_startup_guide
    )
    print("Search results:", results)
