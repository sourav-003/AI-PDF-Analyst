# Intelligent PDF Summarizer & QA Bot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Frontend-Gradio-orange?logo=gradio&logoColor=white)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-8E75B2?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-00d1b2)

An AI-powered tool designed to ingest PDF documents (including scanned images), generate structured summaries, and answer user questions with precise citations. Built for the **AI Intern Assessment (Use Case 3)**.

##  Live Demo
[**Click here to open the Hugging Face Space**](https://huggingface.co/spaces/Sourav-003/PDF-Analyst-Bot)


---

## Overview

This project implements an **Intelligent Agent** capable of analyzing complex documents. Unlike simple keyword searches, it uses **Retrieval-Augmented Generation (RAG)** to "read" the document, understand the context, and provide grounded answers.

### Key Capabilities
* **Summarization:** Instantly generates a hierarchical summary of the document's key themes.
* **Q&A with Citations:** Answers questions based *strictly* on the document content, citing the specific page numbers where the information was found.
* **OCR Support:** Automatically detects scanned pages (images) and uses **Tesseract OCR** to extract text.
* **Streaming Interface:** Provides a real-time, typewriter-style chat experience.

---

## Architecture & Tech Stack

The agent follows a linear RAG pipeline:

1.  **Ingestion:** `PyMuPDF` parses text; `pytesseract` handles OCR for images.
2.  **Chunking:** `LangChain` splits text into semantic chunks (1000 chars with overlap).
3.  **Embedding:** `SentenceTransformers` (`all-MiniLM-L6-v2`) converts text to vectors.
4.  **Indexing:** `FAISS` (Facebook AI Similarity Search) stores vectors for fast retrieval.
5.  **Reasoning:** `Google Gemini 2.0 Flash` synthesizes the final answer from retrieved context.
6.  **UI:** `Gradio` provides the web interface.

---

##  Project Structure

```plaintext
.
├── app.py                  # Main application logic
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (for Hugging Face)
└── README.md               # Project documentation
```

---

## Testing the Agent

To verify the bot's reasoning capabilities, try asking:

- **Fact Extraction:**  
  *"What are the specific deliverables listed in the document?"*

- **Reasoning:**  
  *"Compare Use Case 1 and Use Case 3. How are the agentic elements different?"*

- **Negative Constraint:**  
  *"Does this document mention Java support?"*  
  → It should correctly respond **No / Not explicitly**.
