import os
import io
import time
import fitz
import pytesseract
from PIL import Image
import numpy as np
import gradio as gr
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CONFIGURATION
# 1. Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Google API Key not found in Environment Variables.")
    gemini_model = None
else:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini API Configured.")
    except Exception as e:
        print(f"Error configuring API: {e}")
        gemini_model = None

# 2. Load Embedding Model
print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded.")

# CORE FUNCTIONS
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    print(f"Extracting text from {doc.page_count} pages...")

    for i, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            # OCR Fallback
            try:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            except Exception as e:
                print(f"OCR Error on page {i+1}: {e}")
                text = ""
        full_text += f"\n--- PAGE {i+1} ---\n{text}"

    return full_text

def smart_chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [{"text": chunk, "id": i} for i, chunk in enumerate(chunks)]

def build_vector_store(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# CHAT LOGIC
def process_file(file):
    if not file:
        return "Please upload a file.", None
    
    text = extract_text_from_pdf(file.name)
    chunks = smart_chunking(text)
    index, chunk_data = build_vector_store(chunks)
    
    if not gemini_model:
        return "Error: API Key not configured.", (index, chunk_data)

    # Generate Summary
    prompt = f"Summarize this document in 5 bullet points:\n\n{text[:50000]}"
    try:
        response = gemini_model.generate_content(prompt)
        summary_text = response.text
    except Exception as e:
        summary_text = f"Error generating summary: {e}"
    
    return summary_text, (index, chunk_data)

def chat_response(message, history, pdf_state):
    if not pdf_state:
        yield "Please upload and process a PDF first."
        return

    if not gemini_model:
        yield "API Key error. Please check Space settings."
        return

    index, chunks = pdf_state
    
    # 1. Retrieve Context
    q_emb = embed_model.encode([message], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=5)
    
    context_text = "\n\n".join([chunks[i]["text"] for i in I[0]])
    
    # 2. Format History
    history_context = ""
    for user_msg, bot_msg in history:
        history_context += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    
    # 3. Prompt
    system_prompt = f"""
    You are a helpful PDF assistant. Use the context below to answer the user's question.
    If the answer is not in the context, say you don't know.
    
    CONTEXT FROM PDF:
    {context_text}
    
    CONVERSATION HISTORY:
    {history_context}
    
    CURRENT QUESTION:
    {message}
    """
    
    # 4. Stream Response
    try:
        response_stream = gemini_model.generate_content(system_prompt, stream=True)
        partial_text = ""
        for chunk in response_stream:
            partial_text += chunk.text
            yield partial_text
            time.sleep(0.05)
            
    except Exception as e:
        if "429" in str(e):
            yield "Rate limit exceeded. Please wait 30 seconds."
        else:
            yield f"Error: {str(e)}"

# UI LAUNCH
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## AI PDF Analyst")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process Document", variant="primary")
            summary_box = gr.Textbox(label="Document Summary", lines=8)
            pdf_state = gr.State()

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_response,
                additional_inputs=[pdf_state],
                examples=[
                    ["What is the main conclusion?"],
                    ["Summarize page 1"],
                    ["Explain the methodology"]
                ],
                title="Chat with your PDF"
            )

    process_btn.click(
        fn=process_file,
        inputs=[pdf_input],
        outputs=[summary_box, pdf_state]
    )

if __name__ == "__main__":
    demo.launch()