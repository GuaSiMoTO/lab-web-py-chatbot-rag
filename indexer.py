import os
import glob
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# 1. Configuración de clientes (Apuntando a LM Studio)
# LM Studio por defecto corre en el puerto 1234
client_llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Inicializar ChromaDB (Persistente para no perder los datos al cerrar)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Usaremos la función de embeddings por defecto de Chroma o la de LM Studio.
# Nota: Asegúrate de tener un modelo de embeddings cargado en LM Studio si usas su API.
# Para simplificar y asegurar que funcione sin doble modelo en LM Studio,
# usaremos el embedding por defecto de Chroma (Sentence Transformers en local).
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="documentos_empresa", embedding_function=embedding_fn)

def chunk_text(text, max_chars=500):
    """Función de chunking simple por caracteres"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def indexar_documentos():
    archivos = glob.glob("docs/*.txt")
    total_chunks = 0
    total_tokens_estimados = 0
    
    print(f"Leyendo {len(archivos)} documentos...")
    
    for ruta_archivo in archivos:
        nombre_archivo = os.path.basename(ruta_archivo)
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            contenido = f.read()
        
        # Fragmentar
        chunks = chunk_text(contenido)
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{nombre_archivo}_chunk_{idx}"
            
            # Estimación burda de tokens (1 token ≈ 4 caracteres en inglés/español aprox)
            tokens_chunk = len(chunk) // 4
            total_tokens_estimados += tokens_chunk
            
            # Guardar en ChromaDB
            collection.add(
                documents=[chunk],
                metadatas=[{"fuente": nombre_archivo, "chunk_id": chunk_id}],
                ids=[chunk_id]
            )
            total_chunks += 1

    # Resumen solicitado por el LAB
    print("\n--- RESUMEN DE INDEXACIÓN ---")
    print(f"Documentos procesados: {len(archivos)}")
    print(f"Total de chunks creados: {total_chunks}")
    print(f"Tokens estimados procesados: {total_tokens_estimados}")
    print("Coste estimado: $0.00 (¡Estás usando LM Studio local!)")

if __name__ == "__main__":
    indexar_documentos()