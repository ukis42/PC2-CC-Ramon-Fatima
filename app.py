import streamlit as st
import pymongo
import google.generativeai as genai
from PyPDF2 import PdfReader
import b2sdk.v2 as b2
import base64
import os
import cohere
import time

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]
COHERE_API_KEY = st.secrets["app"]["COHERE_API_KEY"]

B2_READ_KEY_ID = st.secrets["b2"]["B2_READ_KEY_ID"]
B2_READ_APPLICATION_KEY = st.secrets["b2"]["B2_READ_APPLICATION_KEY"]
B2_WRITE_KEY_ID = st.secrets["b2"]["B2_WRITE_KEY_ID"]
B2_WRITE_APPLICATION_KEY = st.secrets["b2"]["B2_WRITE_APPLICATION_KEY"]
B2_BUCKET_NAME = st.secrets["b2"]["B2_BUCKET_NAME"]

USER = st.secrets["app"].get("USER", "")

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan GOOGLE_API_KEY o MONGODB_URI en secrets")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

# MongoDB
client = pymongo.MongoClient(MONGODB_URI)
db = client["pdf_embeddings_db"]
collection = db["pdf_vectors"]


def crear_indice_vectorial():
  from pymongo.operations import SearchIndexModel

  # Conexión a MongoDB Atlas
  client = pymongo.MongoClient(MONGODB_URI)
  db = client.pdf_embeddings_db
  collection = db.pdf_vectors
  collection.insert_one({"a":"sample"})

  existing_indexes = [index['name'] for index in collection.list_search_indexes()]
  if "vector_index" in existing_indexes:
    print("El índice 'vector_index' ya existe. No se crea nuevamente.")
    return

  # Create your index model, then create the search index
  search_index_model = SearchIndexModel(
    definition = {
      "fields": [
        {
          "type": "vector",
          "path": "embedding",
          "similarity": "cosine",
          "numDimensions": 768
        }
      ]
    },
    name="vector_index",
    type="vectorSearch"
  )

  collection.create_search_index(model=search_index_model)
  time.sleep(20)

crear_indice_vectorial()
# =======================
# BACKBLAZE CONEXIONES
# =======================

def conectar_b2_lectura():
    """Devuelve bucket con permisos de lectura."""
    info = b2.InMemoryAccountInfo()
    api = b2.B2Api(info)
    api.authorize_account("production", B2_READ_KEY_ID, B2_READ_APPLICATION_KEY)
    return api.get_bucket_by_name(B2_BUCKET_NAME)

def conectar_b2_escritura():
    """Devuelve bucket con permisos de escritura."""
    info = b2.InMemoryAccountInfo()
    api = b2.B2Api(info)
    api.authorize_account("production", B2_WRITE_KEY_ID, B2_WRITE_APPLICATION_KEY)
    return api.get_bucket_by_name(B2_BUCKET_NAME)

# =======================
# FUNCIONES PDF + EMBEDDING
# =======================

def leer_pdf(archivo):
    reader = PdfReader(archivo)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"
    return texto.strip()

def crear_embedding(texto):
    """Genera embeddings usando Cohere (modelo multilenguaje)."""
    resp = co.embed(
        model="multilingual-22-12",
        texts=[texto]
    )
    return resp.embeddings[0]

def procesar_pdf(archivo_pdf, nombre_pdf):
    """Lee PDF, genera embeddings, guarda en MongoDB y sube PDF a Backblaze."""
    st.info("📄 Leyendo PDF...")

    texto = leer_pdf(archivo_pdf)
    if not texto:
        st.error("El PDF no contiene texto.")
        return None

    trozos = [texto[i:i + 1000] for i in range(0, len(texto), 1000)]

    documentos = []
    for i, chunk in enumerate(trozos):
        embedding = crear_embedding(chunk)
        documentos.append({
            "pdf": nombre_pdf,
            "id": i,
            "texto": chunk,
            "embedding": embedding
        })

    # Guardar en MongoDB
    collection.insert_many(documentos)

    # Subir a Backblaze usando la clave de escritura
    st.info("📤 Subiendo PDF a Backblaze...")

    bucket = conectar_b2_escritura()
    bucket.upload_bytes(
        archivo_pdf.getvalue(),
        file_name=nombre_pdf,
        content_type="application/pdf"
    )

    return len(documentos)

# =======================
# VISOR PDF DESDE BACKBLAZE
# =======================

def obtener_pdf(nombre_pdf):
    """Devuelve la URL pública o autenticada del PDF en Backblaze."""
    bucket = conectar_b2_lectura()
    bucket.download_file_by_name(nombre_pdf).save_to("_"+nombre_pdf)

def mostrar_pdf(nombre_pdf):
    """Muestra PDF en un iframe dentro de Streamlit."""
    with open(nombre_pdf, "rb") as f:
        pdf_bytes = f.read()

    b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

# =======================
# VECTOR SEARCH + CHAT
# =======================

def buscar_similares(embedding, k=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    return list(collection.aggregate(pipeline))

def generar_respuesta(pregunta, contextos):
    modelo = genai.GenerativeModel("gemini-flash-latest")
    contexto = "\n\n".join([c["texto"] for c in contextos])

    prompt = f"""
Usa el contexto para responder la pregunta.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde en español, de forma clara.
"""
    respuesta = modelo.generate_content(prompt)
    return respuesta.text


# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(page_title="ChatBot", page_icon="📚")
st.title("📚 Chat con PDFs almacenados en Backblaze + MongoDB + Gemini + Cohere: "+USER)

# Subida del PDF
archivo_pdf = st.file_uploader("📤 Sube un PDF", type=["pdf"])

if archivo_pdf:
    if st.button("Procesar y guardar PDF"):
        with st.spinner("Procesando PDF..."):
            cantidad = procesar_pdf(archivo_pdf, archivo_pdf.name)
            st.success(f"Procesado: {cantidad} fragmentos generados y PDF guardado.")

        st.info("📖 Vista previa del PDF desde Backblaze:")

        obtener_pdf(archivo_pdf.name)
        mostrar_pdf("_"+archivo_pdf.name)

# ---------------- Chat ----------------

st.subheader("💬 Pregunta sobre el contenido del PDF")

if "historial" not in st.session_state:
    st.session_state.historial = []

pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    with st.spinner("Buscando en el PDF..."):
        emb = crear_embedding(pregunta)
        similares = buscar_similares(emb)

        if not similares:
            respuesta = "No encontré información relevante."
        else:
            respuesta = generar_respuesta(pregunta, similares)

        st.session_state.historial.append({"rol": "usuario", "texto": pregunta})
        st.session_state.historial.append({"rol": "bot", "texto": respuesta})

for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])
