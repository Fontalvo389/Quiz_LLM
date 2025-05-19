# 📚 Quiz\_LLM — Chatbot RAG basado en PDF con LangChain y Ollama

Este proyecto implementa un **chatbot de preguntas y respuestas basado en PDF**, utilizando **RAG (Retrieval-Augmented Generation)**. Aprovecha tecnologías de última generación como:

* 🧠 **LangChain** + **Ollama** para generación de respuestas contextuales
* 💂️ **ChromaDB** para almacenamiento vectorial
* 🌐 **Streamlit** como interfaz gráfica
* 💪 **LangSmith** para trazabilidad y depuración

---

## 🚀 Requisitos

* Python **3.12+**
* [Ollama](https://ollama.com) instalado y ejecutándose localmente
* [uv](https://github.com/astral-sh/uv) instalado (`pip install uv`)
* `.env` con variable `LANGCHAIN_PROJECT`

---

## ⚙️ Instalación

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/tu-usuario/Quiz_LLM.git
   cd Quiz_LLM
   ```

2. **Instala dependencias y crea entorno:**

   ```bash
   uv venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

   O directamente con `uv` y el `pyproject.toml`:

   ```bash
   uv pip install -e .
   ```

3. **Configura el entorno:**

   Crea un archivo `.env` en la raíz:

   ```env
   LANGCHAIN_PROJECT=NombreDeTuProyectoLangSmith
   ```

4. **Asegúrate de que Ollama esté corriendo:**

   ```bash
   ollama run llama3
   ```

---

## ▶️ Ejecutar la aplicación

Utiliza el `Makefile` para ejecutar la app:

```bash
make app
```

Esto ejecutará internamente:

```bash
uv run streamlit run app.py
```

La interfaz se abrirá en tu navegador para cargar PDFs y hacer preguntas.

---

## 📜 Estructura del Proyecto

```
Quiz_LLM/
├── scr/
│   ├── app.py             # Interfaz principal (Streamlit)
│   ├── rag.py             # Lógica de carga y RAG
│   ├── data/              # PDFs subidos (excluido en .gitignore)
│   └── chroma/            # Vector store persistente
├── data/                  # Carpeta raíz para PDFs
├── .env                   # Variables de entorno (no subir)
├── .gitignore
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🛆 Dependencias

Las dependencias están gestionadas desde `pyproject.toml`:

```toml
[project]
name = "quiz"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "chromadb>=1.0.9",
    "community>=1.0.0b1",
    "dotenv>=0.9.9",
    "langchain>=0.3.25",
    "langchain-chroma>=0.2.4",
    "langchain-community>=0.3.24",
    "langchain-ollama>=0.3.3",
    "langsmith>=0.3.42",
    "ollama>=0.4.8",
    "pypdf>=5.5.0",
    "streamlit>=1.45.1",
]
```

---



