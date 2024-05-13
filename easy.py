import os
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.readers import PDFReader
from llama_index import StorageContext, load_index_from_storage
from transformers import pipeline
from llama_index.service_context import ServiceContext

import pickle
from sentence_transformers import SentenceTransformer
import faiss

def build_index(data, index_path=None):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    if index_path and os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        index = load_index(index_path)
    else:
        print("Building index")
        embeddings = embedder.encode(data)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        if index_path:
            save_index(index, index_path)
    return index, embedder, data

def search(index, embedder, data, query, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [(data[idx], distance) for idx, distance in zip(indices[0], distances[0])]

def save_index(index, index_path):
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)

def load_index(index_path):
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print(f"Building index {index_name}")
        # Create a custom ServiceContext without an LLM
        service_context = ServiceContext.from_defaults(llm_predictor=None)
        # Create a GPT Vector Store Index from the data
        index = GPTVectorStoreIndex.from_documents(data, service_context=service_context, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))
    return index

pdf_path = "data.pdf"
loaded_pdf = PDFReader().load_data(file=pdf_path)

# Load the Hugging Face model and tokenizer
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

text = [doc.get_text() for doc in loaded_pdf]

index_path = "pdf_index.pkl"
index, embedder, data = build_index(text, index_path)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    # Retrieve relevant text chunks
    results = search(index, embedder, data, prompt, k=5)
    context = ' '.join([text for text, distance in results])

    # Generate answer using the question-answering model
    qa_input = {
        'question': prompt,
        'context': context
    }
    result = qa_pipeline(qa_input)
    print(f"Answer: {result['answer']}")