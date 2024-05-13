import os
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.readers import PDFReader
from llama_index import StorageContext, load_index_from_storage
from transformers import pipeline
from llama_index.service_context import ServiceContext

import pickle
from sentence_transformers import SentenceTransformer
import faiss
from typing import List


class CustomVectorIndex:
    def __init__(self, data: List[str], index_path: str = None):
        self.data = data
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        if index_path and os.path.exists(index_path):
            print(f"Loading index from {index_path}")
            self.index = self.load_index(index_path)
        else:
            print("Building index")
            self.index = self.build_index()
            if index_path:
                self.save_index(index_path)

    def build_index(self):
        embeddings = self.embedder.encode(self.data)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def search(self, query: str, k: int = 5):
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        results = [(self.data[idx], distance) for idx, distance in zip(indices[0], distances[0])]
        return results

    def save_index(self, index_path: str):
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)

    def load_index(self, index_path: str):
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

pdf_path = "Canada.pdf"
loaded_pdf = PDFReader().load_data(file=pdf_path)

# Load the Hugging Face model and tokenizer
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

text = [doc.get_text() for doc in loaded_pdf]

index_path = "pdf_index.pkl"
custom_index = CustomVectorIndex(text, index_path)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    # Retrieve relevant text chunks
    results = custom_index.search(prompt, k=5)
    context = ' '.join([text for text, distance in results])

    # Generate answer using the question-answering model
    qa_input = {
        'question': prompt,
        'context': context
    }
    result = qa_pipeline(qa_input)
    print(f"Answer: {result['answer']}")
