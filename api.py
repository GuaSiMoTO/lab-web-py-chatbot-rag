from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import time
import re
import logging
from chatbot import chat, HISTORIALES, collection

# Configuración del Logging (Parte 4: No registrará el contenido de los docs, solo info de control)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="RAG Chatbot API",
    description="API con FastAPI para interactuar con nuestro sistema RAG local mediante LM Studio."
)

# Diccionario en memoria para controlar el Rate Limiting { ip: [timestamps] }
CLIENTES_RATE_LIMIT = {}

# --- MODELOS DE DATOS (Pydantic) ---
class ChatRequest(BaseModel):
    # Parte 4: Validación de input (longitud máxima de pregunta: 500 caracteres)
    pregunta: str = Field(..., max_length=500, description="Pregunta para el chatbot (Máx 500 caracteres)")
    session_id: str = Field(..., description="ID único para mantener el historial de la conversación")

# --- FUNCIONES DE SEGURIDAD Y PRIVACIDAD (Parte 4) ---

def verificar_rate_limit(ip: str):
    """Implementa un Rate limiting básico: Máx 10 peticiones por minuto por IP"""
    ahora = time.time()
    if ip not in CLIENTES_RATE_LIMIT:
        CLIENTES_RATE_LIMIT[ip] = []
    
    # Limpiar marcas de tiempo que tengan más de 60 segundos
    CLIENTES_RATE_LIMIT[ip] = [t for t in CLIENTES_RATE_LIMIT[ip] if ahora - t < 60]
    
    # Si ya hay 10 peticiones en el último minuto, lanzamos un error 429
    if len(CLIENTES_RATE_LIMIT[ip]) >= 10:
        raise HTTPException(
            status_code=429, 
            detail="Demasiadas peticiones. Límite de 10 consultas por minuto excedido."
        )
    
    # Registrar la petición actual
    CLIENTES_RATE_LIMIT[ip].append(ahora)

def detectar_informacion_personal(texto: str) -> bool:
    """Detecta si la pregunta contiene correos electrónicos o estructuras de nombres"""
    patron_email = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # Detecta frases típicas como "me llamo X", "mi nombre es X", "mi correo es X"
    patron_nombre = r'(?i)\b(mi\s+nombre\s+es|me\s+llamo|mi\s+correo\s+es)\b'
    
    if re.search(patron_email, texto) or re.search(patron_nombre, texto):
        return True
    return False

# --- ENDPOINTS (Parte 3) ---

@app.post("/chat", summary="Enviar una pregunta al chatbot RAG")
async def post_chat(request_data: ChatRequest, request: Request):
    ip_cliente = request.client.host
    
    # 1. Aplicar Rate Limiting
    verificar_rate_limit(ip_cliente)
    
    # 2. Medida de Privacidad: Validar Información Personal (PII)
    if detectar_informacion_personal(request_data.pregunta):
        raise HTTPException(
            status_code=400,
            detail="Privacidad: Se ha detectado información personal (Nombre o Email) en tu consulta. Por seguridad, elimínala antes de enviarla al modelo."
        )
        
    # 3. Logging seguro de la llamada (Sin registrar el texto completo del documento)
    logging.info(f"IP: {ip_cliente} | Sesión: {request_data.session_id} | Longitud Pregunta: {len(request_data.pregunta)} chars")
    
    # Ejecutar la lógica del RAG que programamos en chatbot.py
    resultado = chat(request_data.pregunta, request_data.session_id)
    return resultado

@app.get("/chat/history/{session_id}", summary="Obtener el historial de una sesión")
async def get_historial(session_id: str):
    if session_id not in HISTORIALES:
        return {"session_id": session_id, "historial": [], "mensaje": "No hay historial para esta sesión."}
    return {"session_id": session_id, "historial": HISTORIALES[session_id]}

@app.get("/documentos", summary="Listar los documentos que han sido indexados en ChromaDB")
async def get_documentos():
    # Recuperamos los datos guardados en la colección de Chroma
    datos = collection.get()
    if not datos or not datos.get('metadatas'):
        return {"documentos_indexados": []}
    
    # Extraemos los nombres únicos de los archivos fuente de los metadatos
    archivos_unicos = list(set([meta['fuente'] for meta in datos['metadatas'] if 'fuente' in meta]))
    return {"documentos_indexados": archivos_unicos}