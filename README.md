# 📚 RAG Pipeline with Llama 2 (Ollama + LangChain)

This repository demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using a locally hosted **Llama 2** model with [Ollama](https://ollama.ai/) and [LangChain](https://www.langchain.com/).

The pipeline loads documents, creates embeddings, and retrieves context-relevant passages to enhance LLM responses.

---

## 🚀 Features

- Run **Llama 2 (llama2:latest)** locally with Ollama.
- Load and split PDFs into chunks using `PyPDFLoader`.
- Generate embeddings with `OllamaEmbeddings`.
- Store and search vectors in `DocArrayInMemorySearch`.
- Build a **retriever → prompt → LLM → parser** chain.
- Answer questions with context-aware responses.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create and activate a Python environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and Llama 2

Follow [Ollama setup instructions](https://github.com/jmorganca/ollama).  
Then pull the Llama 2 model:

```bash
ollama pull llama2
```

---

## 📖 Usage

### Run the Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

### Example Workflow

1. Load a PDF:
   ```python
   from langchain_community.document_loaders import PyPDFLoader
   loader = PyPDFLoader("sagar-manchakatla-resume.pdf")
   pages = loader.load_and_split()
   ```
2. Create embeddings & vector store:
   ```python
   from langchain_community.vectorstores import DocArrayInMemorySearch
   vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding)
   retriever = vectorstore.as_retriever()
   ```
3. Build the RAG chain:
   ```python
   from operator import itemgetter
   chain = (
       {
           "context": itemgetter("question") | retriever,
           "question": itemgetter("question"),
       }
       | prompt
       | model
       | parser
   )
   ```
4. Ask questions:
   ```python
   chain.invoke({"question": "What projects has Sagar worked on?"})
   ```

---

## 📂 Project Structure

```
.
├── notebook.ipynb         # Jupyter notebook with the RAG pipeline
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── data/                  # (Optional) Store PDFs or documents here
```

---

## 🔮 Next Steps

- Add support for multiple document types (CSV, TXT, web pages).
- Swap `DocArrayInMemorySearch` with a persistent vector store (like FAISS, Pinecone, Weaviate).
- Extend the prompt template for multi-turn conversations.
