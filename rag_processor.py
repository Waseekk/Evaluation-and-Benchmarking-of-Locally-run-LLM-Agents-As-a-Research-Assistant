# rag_processor.py

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class RAGProcessor:
    """Handles RAG processing for research paper analysis."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ";"],
            length_function=len
        )
        self.vector_store = None
    
    def process_paper(self, text: str) -> None:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input: text must be a non-empty string")
        try:
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                raise ValueError("No valid chunks created from input text")
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"chunk_id": i, "source": "research_paper"}
                )
                for i, chunk in enumerate(chunks)
            ]
            if not documents:
                raise ValueError("No valid documents created")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            raise RuntimeError(f"Failed to process paper: {str(e)}")
    
    def get_relevant_chunks(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise ValueError("No paper has been processed yet. Call process_paper first.")
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        return results
    
    def create_enhanced_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        sorted_chunks = sorted(chunks, key=lambda x: x['similarity_score'], reverse=True)
        context = "\n\n".join([
            f"[Excerpt {i+1} (Similarity: {chunk['similarity_score']:.2f})]\n{chunk['content']}"
            for i, chunk in enumerate(sorted_chunks)
        ])
        prompt = f"""Answer the following question based on the provided research paper excerpts.
Question: {query}

Relevant Paper Excerpts:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Uses specific information from the provided excerpts
3. Maintains academic tone and precision
4. Cites specific excerpts when appropriate
5. Acknowledges if information might be incomplete

Answer:"""
        return prompt
