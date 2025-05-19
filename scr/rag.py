import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

CHROMA_DIR = "chromadb"

def cargar_pdf_y_generar_vectores(ruta_pdf: str):
    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_documents(paginas)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(fragmentos, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

def cargar_vectores_existentes():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def construir_rag(vectordb, temperature=0.7, top_p=0.95, top_k=40):
    llm = Ollama(
        model="llama3.2:latest",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
