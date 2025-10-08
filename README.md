# Rag Based Daraz Product Recommender Chatbot

An end‑to‑end Retrieval‑Augmented Generation (RAG) system that recommends **Daraz** products from a local CSV, stores embeddings in **AstraDB**, reasons with **Groq LLMs**, and serves a clean **Flask** chat UI.  

---

## ✨ Highlights

* **RAG with chat memory**: history‑aware retriever rewrites follow‑ups into standalone queries.
* **AstraDB Vector Store**: scalable semantic search with metadata filters (Category, SubCategory).
* **Local embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (no external HF endpoint needed).
* **LLM**: Groq via `ChatGroq` for low‑latency generation.
* **Frontend**: HTML and CSS `Most of the code snippets taken from chatgtp`.
* **Backend**: Flask app as the backend serving the RAG pipeline.

---

## 🧭 Architecture

![Ragchain](/Data/RAG_chain.png)


---

## 📂 Project Structure

```
.
├─ app.py                      # Flask entrypoint
├─ templates/
│  └─ index.html               # Chat UI (Bootstrap, jQuery)
├─ static/
│  ├─ css/style.css
│  └─ images/daraz_logo.png    # Served via url_for('static', ...)
├─ Data/
│  └─ Daraz_product.csv        # Source data
└─ daraz_folder/
   ├─ __init__.py
   ├─ config.py                # Reads env; central config
   ├─ data_converter.py        # CSV -> LangChain Documents
   ├─ data_ingestion.py        # Embeddings + AstraDB ingestion (Online)
   └─ rag_chain.py             # History-aware RAG chain also prompt is declared
```

---

## 🧱 Data Model (Document Schema)
* The dataset is downloaded from Kaggle (Sorry I have lost the link but you can download the dataset from **Data** folder)

* The dataset has total 5 colums ```(Category,SubCategory,Title,Original Price,Positive Seller Ratings)``` and total 12,908 instances are present. Among them I am embedded only the Title column and declared other columns as meta data. 
```python
# data_converter.py
Document(
  page_content=row["Title"],
  metadata={
    "Category": row["Category"],
    "SubCategory": row["SubCategory"],
    "Original Price": row["Original Price"],
    "Positive Seller Ratings": row["Positive Seller Ratings"],
  }
)
```

---

## 🧠 RAG & Prompting Strategy

### 1. **create_history_aware_retriever(...)**

The role of this chain is not to directly answer questions but to prepare a clean and well-formatted query for the retriever. It refines raw user input and fetches the top relevant documents for the next steps.

#### Input:
- **Raw User Input:** The initial question posed by the user (Human: "tell me about the battery life of iPhone 14").
- **Chat History (if any):** Any previous context or interactions with the user.
- **Context Prompt:** System instructions for rephrasing and cleaning the query.

#### Process:
1. The raw user query is passed to the model for rewriting.
2. The model refines the question. For example:
   - Original input: "tell me about the battery life i-phone 14"
   - Rewritten query: "What is the battery life of iPhone 14?"
3. The refined query is passed to the retriever with additional search parameters (e.g., `k=5` to fetch the top 5 relevent documents).

#### Output:
- **Retrieved Documents:** The top 5 most relevant documents are fetched from the database.
- **Cleaned Question:** The rewritten user query is passed along for the next chain.

---

### 2. **create_stuff_documents_chain(...)**

This chain uses the cleaned query and documents to generate a context-based answer.

#### Input:
- **Cleaned User Question:** The refined user query (e.g., "What is the battery life of iPhone 14?").
- **Retrieved Context:** The documents or data retrieved by the history-aware retriever.
- **QA Prompt:** This promt will pass System messange and if any history is present in the **MessagesPlaceholder( )**.

#### Process:
1. The QA prompt is filled with the retrieved documents and the user’s cleaned query.
2. The model generates a response based on the input.
   - Example template:
     ```
     System: You're an e-commerce bot answering product-related queries...
     CONTEXT: [retrieved docs]
     QUESTION: What is the battery life of iPhone 14? --> rewritten user question
     ```
3. The model generates a concise and context-aware answer.

#### Output:
- **Final Answer:** A response generated based on the cleaned query and retrieved context.
  - Example answer: "The iPhone 14 offers up to 20 hours of video playback according to reviews."

---

## Overall Workflow

1. **User Input:** "tell me about the battery life of iPhone 14"
2. **History-Aware Retriever:** Cleans the question to: "What is the battery life of iPhone 14?"
   - Retrieves top 3 documents related to iPhone 14 battery life.
3. **Stuff Documents Chain:** Uses the retrieved documents, rewritten user question, system messange and if any chat history is present to generate a final answer.
   - Output: "The iPhone 14 offers up to 20 hours of video playback according to reviews."

---

## ⚙️ Tech Stack

* **Python** 3.11+
* **Flask**, **HTML, CSS**
* **LangChain** (core), `langchain_groq`, `langchain_astradb`
* **AstraDB** (DataStax) via `AstraDBVectorStore`
* **Embeddings**: `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`
* **LLM**: Groq (model set in `Config.RAG_MODEL`)
* **Monitoring**: `prometheus_client`

---

## 🚀 Quickstart

### 1) Clone & Install

```bash
1. Open Command Prompt 
2. Install all the necessary libraries that is included in the requirements.txt file. 
```

### 2) Environment

Create a `.env` in the repo root:

```dotenv
# Astra DB
ASTRA_DB_API_ENDPOINT= Use your own key
ASTRA_DB_APPLICATION_TOKEN= use your own key
ASTRA_DB_KEYSPACE="default_keyspace" <-- write as it is. 

# For LLM Model
GROQ_API_KEY=xxxxxxxxxxxx <-- use your key

# Embedding Model
Use any embedding model you want (suggestion: Use hugging face api, its free) I didn't use for this project because i have crossed my monthly usages limitations :)) so i am downloaded embedding model locally in my machine. 
```

> **Note**: Embeddings are **local** (no HF token required). If you wish to switch to HF Inference, swap to `HuggingFaceEndpointEmbeddings` and add `HUGGINGFACEHUB_API_TOKEN`.

### 3) One‑time Ingestion
**Important:** Put load_existing=False if you are loading your dataset to AstraDB for the first time. 
```python
# Run once in a Python shell to populate AstraDB
from daraz_folder.data_ingestion import DataIngestor
DataIngestor().ingest(load_existing=False)
```
after the loading is complete then change load_existing=True everytime you are using the database. 
* Uses `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
set chunk_overlap as you needed. 
* Batched upserts: `batch_size=200` change based on your dataset. 

### 4) Run the App

```bash
python app.py
# open http://localhost:5000
```

---

## 💬 Overview 

![daraz](/Data/daraz%20testcase.png)
