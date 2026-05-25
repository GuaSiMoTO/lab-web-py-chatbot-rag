from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# 1. Conexión con LM Studio (Simulando la API de OpenAI)
# Apuntamos al localhost en el puerto 1234 que configuraste en LM Studio
client_llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 2. Conexión con nuestra base de datos local ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="documentos_empresa", embedding_function=embedding_fn)

# 3. Historial de conversación en memoria
# Usamos un diccionario donde la clave será el session_id y el valor la lista de mensajes
HISTORIALES = {}

# 4. SYSTEM PROMPT (Las instrucciones y restricciones del LAB)
SYSTEM_PROMPT = """Eres un asistente de atención interno honesto, riguroso y respetuoso con la privacidad.
Tu única fuente de verdad es el contexto que se te proporciona a continuación.

REGLAS CRÍTICAS QUE DEBES CUMPLIR:
1. Responde SOLO con la información disponible en el contexto provisto.
2. Si el contexto no contiene la respuesta o no es relevante para responder la pregunta, di exactamente: "No tengo información sobre eso".
3. NO inventes datos, fechas, nombres, extensiones telefónicas ni políticas bajo ninguna circunstancia (prohibido alucinar).
4. No asumas nada que no esté escrito explícitamente.
"""

def chat(pregunta: str, session_id: str) -> dict:
    # Si es una sesión nueva, le creamos su historial vacío
    if session_id not in HISTORIALES:
        HISTORIALES[session_id] = []
        
    # --- PASO 1: Recuperar los 3 fragmentos más relevantes ---
    resultados = collection.query(
        query_texts=[pregunta],
        n_results=3
    )
    
    # Extraemos los textos de los fragmentos y sus metadatos
    fragmentos = resultados['documents'][0] if resultados['documents'] else []
    metadatos = resultados['metadatas'][0] if resultados['metadatas'] else []
    
    # Obtenemos una lista única de los archivos de origen usados como fuentes
    fuentes_usadas = list(set([meta['fuente'] for meta in metadatos if 'fuente' in meta]))
    
    # --- PASO 2: Construir el prompt con contexto ---
    contexto_str = "\n\n".join([f"[Fragmento {i+1}]: {f}" for i, f in enumerate(fragmentos)])
    
    # --- PASO 3: Mantener el historial de la conversación ---
    # Construimos la estructura de mensajes para el LLM
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Añadimos el historial previo de esta sesión específica (limitado a los últimos 6 mensajes para no saturar)
    mensajes.extend(HISTORIALES[session_id][-6:])
    
    # Añadimos la pregunta actual formateada con el contexto recuperado
    mensajes.append({
        "role": "user", 
        "content": f"CONTEXTO DISPONIBLE:\n\"\"\"\n{contexto_str}\n\"\"\"\n\nPREGUNTA DEL USUARIO: {pregunta}"
    })

    # Llamada a LM Studio
    try:
        # Usamos temperature=0.0 para que el modelo sea totalmente determinista y no invente nada
        response = client_llm.chat.completions.create(
            model="local-model", # LM studio acepta cualquier string aquí si solo tienes un modelo cargado
            messages=mensajes,
            temperature=0.0 
        )
        respuesta_llm = response.choices[0].message.content
    except Exception as e:
        respuesta_llm = f"Error al conectar con el servidor local de LM Studio: {str(e)}"

    # Guardamos la interacción actual en el historial de la sesión (para la próxima pregunta)
    HISTORIALES[session_id].append({"role": "user", "content": pregunta})
    HISTORIALES[session_id].append({"role": "assistant", "content": respuesta_llm})

    # --- PASO 4: Estructura de retorno requerida por el LAB ---
    return {
        "respuesta": respuesta_llm,
        "fuentes": fuentes_usadas,
        "session_id": session_id,
        "fragmentos_usados": len(fragmentos)
    }