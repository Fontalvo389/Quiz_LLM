import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain.globals import set_verbose
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.callbacks import tracing_v2_enabled
from langsmith import Client

load_dotenv()
set_verbose(True)
client = Client()

CHROMA_DIR = "chromadb"
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Chat RAG PDF", layout="wide")
st.title("🧠 Chatbot RAG basado en PDF")

with st.sidebar:
    st.header("⚙️ Parámetros del modelo")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95)
    top_k = st.slider("Top-k", 1, 100, 40)



if "historial" not in st.session_state:
    st.session_state.historial = []

if "modelo_inicializado" not in st.session_state:
    st.session_state.modelo_inicializado = False

if "cadena_rag" not in st.session_state:
    st.session_state.cadena_rag = None


def cargar_pdf_y_generar_vectores(ruta_pdf: str):
    try:
        loader = PyPDFLoader(ruta_pdf)
        paginas = loader.load()
        if len(paginas) == 0:
            raise ValueError("PyPDFLoader no cargó contenido")
        st.success(f"📄 PyPDFLoader cargó {len(paginas)} paginas")
    except Exception as e:
        st.warning(f"⚠️ PyPDFLoader falló: {e}")
        loader = UnstructuredPDFLoader(ruta_pdf)
        paginas = loader.load()
        st.success(f"📄 UnstructuredPDFLoader cargó {len(paginas)} fragmentos")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_documents(paginas)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(fragmentos, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

def cargar_vectores_existentes():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def construir_rag(vectordb):
    llm = Ollama(
        model="llama3.2:latest",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)



archivo_pdf = st.file_uploader("📄 sube un archivo PDF", type="pdf")

if archivo_pdf:
    ruta_pdf = DATA_DIR / archivo_pdf.name
    with open(ruta_pdf, "wb") as f:
        f.write(archivo_pdf.getbuffer())

    st.success("✅ archivo PDF cargado correctamente")

    if not Path(CHROMA_DIR).exists():
        with st.spinner("🧠 generando vectores..."):
            vectordb = cargar_pdf_y_generar_vectores(str(ruta_pdf))
    else:
        vectordb = cargar_vectores_existentes()

    with tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT")):
        st.session_state.cadena_rag = construir_rag(vectordb)
        st.session_state.modelo_inicializado = True
        st.success("✅ Modelo RAG inicializado")

# =======================
# Chat con el modelo
# =======================

if st.session_state.modelo_inicializado:
    pregunta_usuario = st.chat_input("Haz una pregunta sobre el documento...")

    if pregunta_usuario:
        st.chat_message("user").write(pregunta_usuario)
        st.session_state.historial.append(("user", pregunta_usuario))

        with st.spinner("🤔 Pensando..."):
            try:
                inicio = time.time()
                resultado = st.session_state.cadena_rag.invoke({"query": pregunta_usuario})
                fin = time.time()
                respuesta = resultado["result"]
                documentos_fuente = resultado.get("source_documents", [])
                elapsed = round(fin - inicio, 2)

                
                print("🧠 Respuesta del modelo:", respuesta)
                for i, doc in enumerate(documentos_fuente):
                    print(f"\n📚 Fuente {i+1}:\n{doc.page_content[:500]}")

                #LangSmith
                runs = client.list_runs(
                    project_name=os.getenv("LANGCHAIN_PROJECT"),
                    execution_order=1,
                    order_by="desc",
                    limit=1,
                )
                trace_url = None
                for run in runs:
                    trace_url = f"https://smith.langchain.com/public/{run.id}"
                    break

            except Exception as e:
                st.error(f"❌ Error al procesar la pregunta: {e}")
                respuesta = "Lo siento, ocurrió un error procesando tu pregunta."
                documentos_fuente = []
                trace_url = None

        st.chat_message("assistant").write(respuesta)
        st.session_state.historial.append(("assistant", respuesta))

        if documentos_fuente:
            with st.expander("📚 Documentos fuente utilizados"):
                for i, doc in enumerate(documentos_fuente):
                    st.markdown(f"*Fragmento {i+1}:*")
                    st.code(doc.page_content[:1000])

        if trace_url:
            st.markdown(f"[🔗 Ver trace en LangSmith]({trace_url})")
            print("🔗 Trace LangSmith:", trace_url)



for rol, mensaje in st.session_state.historial:
    if rol == "user":
        st.chat_message("user").write(mensaje)
    elif rol == "assistant":
        st.chat_message("assistant").write(mensaje)
