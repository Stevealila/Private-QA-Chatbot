import os
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the embeddings and language model
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI()

# Load documents from the specified directory
directory = "./data"
def load_docs(path):
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

documents = load_docs(directory)

# Split documents into smaller chunks
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# Convert documents to embeddings
doc_texts = [doc.page_content for doc in docs]  
doc_embeddings = embeddings_model.embed_documents(doc_texts)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# Create and populate the FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Load the QA chain
chain = load_qa_chain(llm, chain_type="stuff")

# Function to get an answer to a query
def get_answer(query):
    query_embedding = np.array(embeddings_model.embed_query(query)).astype('float32').reshape(1, -1)
    _, I = index.search(query_embedding, k=2)
    similar_docs = [docs[i] for i in I[0]]
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer
# Streamlit app
st.title("Private Q&A chatbot")

prompt = st.text_input("Question: ")

if prompt:
    st.markdown(get_answer(prompt))
