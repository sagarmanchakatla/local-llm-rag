import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Core libraries
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

# Document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store and embeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Reranking 
from sentence_transformers import CrossEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for RAG system"""
    openrouter_api_key: str
    model_name: str = "microsoft/wizardlm-2-8x22b"  # Default model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    temperature: float = 0.1
    max_tokens: int = 2000
    vector_store_path: str = "./chroma_db"

class OptimizedRAGSystem:
    """Optimized RAG system with OpenRouter integration"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.embeddings = None
        self.vectorstore = None
        self.reranker = None
        self.text_splitter = None
        self.chain = None
        
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all system components"""
        try:
            # Setup OpenRouter client
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.config.openrouter_api_key,
            )
            
            # Setup embeddings with caching
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'}, # Use CPU for embeddings
                encode_kwargs={'normalize_embeddings': True} # Normalize embeddings (unit-length, imp for cosine similarity)
            )
            
            # Setup reranker - ranks the retrieved documents on their relavance to a query, A reranker reorders these documents, placing the most relevant ones at the top. It filters out less relevant documents, ensuring the LLM focuses only on the most useful context. Most modern rerankers use a cross-encoder model. Unlike embeddings (which encode queries and documents separately), a cross-encoder jointly encodes the query and each document, capturing their interaction and producing a relevance score.
            self.reranker = CrossEncoder(self.config.reranker_model)
            
            # Setup text splitter with optimized parameters. Used for splitting documents into smaller chunks.
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            logger.info("RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up components: {e}")
            raise
    
    def load_and_process_documents(self, file_paths: List[str]) -> List[Document]:
        """Load and process documents with optimized chunking"""
        all_documents = []
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents = loader.load_and_split()
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                # Enhanced preprocessing
                processed_docs = self._preprocess_documents(documents)
                
                # Split documents with optimized chunking
                split_docs = self.text_splitter.split_documents(processed_docs)
                
                # Add metadata enrichment
                for doc in split_docs:
                    doc.metadata.update({
                        'source_file': Path(file_path).name,
                        'chunk_length': len(doc.page_content)
                    })
                
                all_documents.extend(split_docs)
                logger.info(f"Processed {len(split_docs)} chunks from {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return all_documents
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Enhanced document preprocessing"""
        processed = []
        
        for doc in documents:
            # Clean text
            text = doc.page_content
            
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Remove very short pages (likely headers/footers)
            if len(text) < 50:
                continue
            
            # Update document
            doc.page_content = text
            processed.append(doc)
        
        return processed
    
    def create_vectorstore(self, documents: List[Document], force_rebuild: bool = False):
        """Create or load vector store with persistence"""
        vector_store_path = Path(self.config.vector_store_path)
        
        if not force_rebuild and vector_store_path.exists():
            # Load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=str(vector_store_path),
                embedding_function=self.embeddings
            )
            logger.info("Loaded existing vector store")
        else:
            # Create new vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(vector_store_path)
            )
            logger.info(f"Created new vector store with {len(documents)} documents")
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank retrieved documents using cross-encoder"""
        if not documents:
            return documents
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = self.reranker.predict(pairs)
        
        # Sort documents by relevance score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k reranked documents
        return [doc for doc, score in scored_docs[:self.config.top_k_rerank]]
    
    def _query_openrouter(self, messages: List[Dict[str, str]]) -> str:
        """Query OpenRouter API with error handling and retry logic"""
        max_retries = 3
        # print(messages)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def setup_chain(self):
        """Setup the RAG chain with optimized retrieval and generation"""
        
        # Enhanced prompt template
        template = """You are a helpful assistant that answers questions based on the provided context. 
        Use only the information from the context to answer the question. If the context doesn't contain 
        enough information to answer the question fully, say so clearly.

        Context:
        {context}

        Question: {question}

        Instructions:
        1. Provide a comprehensive answer based on the context
        2. If information is missing, clearly state what cannot be answered
        3. Use specific details and quotes from the context when relevant
        4. Structure your answer clearly with bullet points or paragraphs as appropriate

        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(f"Source: {doc.metadata.get('source_file', 'Unknown')}\n{doc.page_content}" for doc in docs)
        
        def enhanced_retrieval(question):
            # Initial retrieval
            docs = self.vectorstore.similarity_search(
                question, 
                k=self.config.top_k_retrieval
            )
            
            # Rerank documents
            reranked_docs = self._rerank_documents(question, docs)
            
            return reranked_docs
        
        def generate_response(inputs):
            context = format_docs(inputs["context"])
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.format(context=context, question=inputs["question"])}
            ]
            
            return self._query_openrouter(messages)
        
        # Build the chain
        self.chain = RunnableParallel({
            "context": lambda x: enhanced_retrieval(x["question"]),
            "question": RunnablePassthrough()
        }) | generate_response
        
        logger.info("RAG chain setup complete")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system and return detailed response"""
        if not self.chain:
            raise ValueError("Chain not setup. Call setup_chain() first.")
        
        try:
            # Get retrieved documents for transparency
            retrieved_docs = self.vectorstore.similarity_search(question, k=self.config.top_k_retrieval)
            reranked_docs = self._rerank_documents(question, retrieved_docs)
            
            # Generate response
            response = self.chain.invoke({"question": question})
            
            return {
                "answer": response,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in reranked_docs
                ],
                "num_sources": len(reranked_docs)
            }
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {e}",
                "sources": [],
                "num_sources": 0
            }
    
    async def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries asynchronously"""
        tasks = [asyncio.create_task(asyncio.to_thread(self.query, q)) for q in questions]
        return await asyncio.gather(*tasks)

# Usage example and configuration
def main():
    # Configuration
    config = RAGConfig(
        openrouter_api_key="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",
        model_name="microsoft/wizardlm-2-8x22b",  # or any other model from OpenRouter
        chunk_size=800,  # Optimized chunk size
        chunk_overlap=100,
        top_k_retrieval=8,
        top_k_rerank=4,
        temperature=0.1
    )
    
    # Initialize RAG system
    rag = OptimizedRAGSystem(config)
    
    # Load documents
    documents = rag.load_and_process_documents(["Blackbook-Sagar.pdf"])
    
    # Create vector store
    rag.create_vectorstore(documents)
    
    # Setup chain
    rag.setup_chain()
    
    # Example queries
    questions = [
        "What are the roles and responsibilities of each personnel?",
        "What projects did Sagar work on during the internship?",
        "What is Afility Engineering's mission and vision?"
        "What comapies does afility work with?"
    ]
    
    # Single query
    # print("Single Query Example:")
    # result = rag.query(questions[0])
    # print(f"Question: {questions[0]}")
    # print(f"Answer: {result['answer']}")
    # print(f"Sources used: {result['num_sources']}")
    # print()
    
    # Batch queries (async)
    print("Batch Query Example:")
    async def run_batch():
        results = await rag.batch_query(questions)
        for i, result in enumerate(results):
            print(f"Q{i+1}: {questions[i]}")
            print(f"A{i+1}: {result['answer']}")
            print()
    
    # Run batch queries
    asyncio.run(run_batch())

if __name__ == "__main__":
    main()