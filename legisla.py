import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import Document
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import re
from bs4 import BeautifulSoup
import fitz  # PyMuPDF para procesar PDFs
import pandas as pd  # Para procesar archivos Excel
import openai  # Para utilizar la API de OpenAI

# Cargar variables de entorno
load_dotenv()

SALUDOS = "Hola en que puedo asistirte, ¡Saludos! ¿Cómo puedo ayudarte hoy?, ¡Bienvenido/a! ¿Cómo puedo asistirte?, ¡Qué gusto verte por aquí! ¿Cómo puedo ayudarte hoy? "
NOMBRE_DE_LA_EMPRESA = "Corporación Write"
NOMBRE_AGENTE = "Kliofer"
prompt_inicial = f"""
Soy {NOMBRE_AGENTE}, parte del equipo de {NOMBRE_DE_LA_EMPRESA}, un asistente inteligente diseñado para resolver casos basado en normas ISO. 
"""

# Inicializar memoria en session_state si no existe
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Inicializar variable response en session_state para evitar error
if 'response' not in st.session_state:
    st.session_state.response = None

# Conexión a MongoDB
def conexion_a_mongo():
    """ Conecta a MongoDB """
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(MONGO_URI)
    db = client["db-historial-chats"]
    collection = db["coleccion-histochats"]
    return collection

# Función para guardar chat en MongoDB
def guardar_chat_en_mongo(user, message, response, case_use=None, user_solution1=None, user_solution2=None):
    """ Guarda chats en MongoDB junto con las soluciones propuestas. """
    collection = conexion_a_mongo()
    chat_data = {
        "user": user,
        "message": message,
        "response": response,
        "case_use": case_use,
        "user_solution1": user_solution1,
        "user_solution2": user_solution2,
        "timestamp": datetime.now()
    }
    collection.insert_one(chat_data)

# Función principal
def main():
    st.sidebar.markdown("<h3 style='text-align: center; font-size: 36px;'>LEGISLACIÓN</h3>",unsafe_allow_html=True)
    st.sidebar.markdown("**Autores:**\n- *Gabriela Tumbaco*\n- *Gabriel Ruales*\n- *Nicolas Liberio*")
    st.sidebar.image('bott.jpg', width=250)  # o bot.jpg
    st.markdown('<h1 style="color: #FFD700;">ISOCOMPARA</h1>', unsafe_allow_html=True)

    # 🔽 Inicializamos la base de conocimiento en session_state para evitar errores 🔽
    if "knowledgeBase" not in st.session_state:
        st.session_state["knowledgeBase"] = None

    uploaded_files = st.file_uploader("Sube archivos (PDFL)", type=["pdf"], accept_multiple_files=True)
    text = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            if file_type in ["text/html", "application/xml"]:
                soup = BeautifulSoup(uploaded_file, 'html.parser' if file_type == "text/html" else 'xml')
                text += soup.get_text()
            elif file_type == "application/pdf":
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text += "".join([page.get_text("text") for page in doc])
            elif file_type == "text/csv":
                df = pd.read_csv(uploaded_file)
                text += df.to_string(index=False) + "\n"

    # 🔹 Crear Tabs para separar Chat, Costos
    tab1, tab2 = st.tabs(["💬 Chat", "📜 Historial & Costos"])

    with tab1:
        st.markdown("## 💬 Chat con InfoBot")

        # Cajas de texto para el caso de uso y las soluciones
        case_use = st.text_area("Caso de uso a resolver", height=150)
        user_solution1 = st.text_area("Posible solución de Usuario 1", height=150)
        user_solution2 = st.text_area("Posible solución de Usuario 2", height=150)

        tema = st.text_input('Tema del caso de uso')
        iso = st.text_input('Norma ISO a aplicar')

        # Botón para cancelar
        cancel_button = st.button('Cancelar')
        if cancel_button:
            st.stop()


        # Botón para resolver
        resolve_button = st.button('Resolver Caso')
        if resolve_button:
                if case_use and user_solution1 and user_solution2:
                # Crear el prompt dinámico con el contexto del caso de uso, las soluciones y el tema/ISO
                 prompt = f"""
                Eres un analista experto en gestión de riesgos, ciberseguridad y normativas ISO.

                Analiza el siguiente caso de uso, incluyendo el tema y la norma ISO relevante. Evalúa también las soluciones propuestas por dos usuarios y responde siguiendo estrictamente la siguiente estructura. No agregues secciones extra. Solo responde en ese formato.

                **Contexto del problema**:
                Describe el problema utilizando el caso de uso [{case_use}], el tema [{tema}], y la norma ISO aplicada [{iso}].

                **Solución propuesta (solución de la IA después de analizar todo lo que tenía a su disposición)**:
                Aquí debes proponer una solución ideal basada en el análisis de los elementos anteriores. Puedes combinar ideas de los usuarios o mejorarlas, siempre con base en el caso, el tema y la norma ISO indicada.

                **Conclusión y recomendación**:
                Haz una conclusión corta y una recomendación clara sobre cuál solución es más viable o adecuada en el contexto dado.

                **Evaluación de soluciones del [{user_solution1}] y del [{user_solution2}] y la de la IA  (porcentaje de efectividad/pertinencia)**:
                Asigna un porcentaje a cada solución con base en:
                - Relevancia para el caso de uso
                - Concordancia con la norma ISO [{iso}]
                - Aplicabilidad técnica y claridad
                
                Usa exactamente este formato, sin agregar explicaciones ni texto adicional:

                - Usuario 1: [porcentaje]%
                - Usuario 2: [porcentaje]%
                - Solución IA: [porcentaje]%

                Ejemplo:
                - Usuario 1: 75%
                - Usuario 2: 85%
                - Solución IA: 95%

                Responde en markdown con títulos en negrita como se indica. No salgas de ese formato.
                """

                # Concatenar el texto cargado si es que lo hay
                knowledgeBase = st.session_state.get("knowledgeBase", None)
                context = None  # Inicializamos context con un valor por defecto
                if knowledgeBase:
                    docs = knowledgeBase.similarity_search(prompt)
                    context = "\n".join([doc.page_content for doc in docs]) if docs else "No hay información relevante."

                # Obtener historial de la conversación desde la memoria
                history_messages = st.session_state.memory.load_memory_variables({}).get("history", [])
                messages = [{"role": "system", "content": prompt_inicial}]

                # Agregar los mensajes anteriores de la conversación
                for message in history_messages:
                    messages.append({"role": message['role'], "content": message['content']})

                # Agregar la respuesta anterior del asistente si existe un historial
                if history_messages:
                    last_message = history_messages[-1]
                    messages.append({"role": "assistant", "content": last_message['content']})

                # Agregar el nuevo caso de uso
                messages.append({"role": "user", "content": case_use}) 
                # Incluir el contexto relevante si existe
                if context:
                    messages.append({"role": "system", "content": context})
                else:
                    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": case_use}]

                # Llamar a la API de OpenAI para obtener la respuesta del bot
                with get_openai_callback() as obtienec:
                    start_time = datetime.now()
                    st.session_state.response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=messages,
                        api_key=os.environ.get("OPENAI_API_KEY"),
                    )
                    end_time = datetime.now()
                    answer = st.session_state.response['choices'][0]['message']['content'] if st.session_state.response.get('choices') else "Lo siento, no pude obtener una respuesta."

                # Guardar el chat en la base de datos MongoDB
                guardar_chat_en_mongo("user1", case_use, answer, case_use, user_solution1, user_solution2)

                st.write(answer)


                # 🔻 BOTÓN PARA DESCARGAR EN PDF 🔻
                from fpdf import FPDF
                import base64

                class PDF(FPDF):
                    def header(self):
                        self.set_font('Arial', 'B', 12)
                        self.cell(0, 10, 'Respuesta del Caso Analizado', ln=True, align='C')

                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.cell(0, 10, f'Página {self.page_no()}', align='C')

                def generar_pdf(respuesta_texto):
                    pdf = PDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)
                    lines = respuesta_texto.split('\n')
                    for line in lines:
                        pdf.multi_cell(0, 10, line)
                    return pdf.output(dest='S').encode('latin1')

                if st.session_state.response:
                    pdf_bytes = generar_pdf(answer)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="respuesta_IA.pdf">📄 Descargar respuesta en PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)




    with tab2:
        st.markdown("## 📜 Historial de Conversación y Costos")
        st.write("### 🔹 Historial de Conversación")
        st.write(st.session_state.memory.buffer)
        st.write("### 💰 Costos y Tokens")
        if st.session_state.response:
            response = st.session_state.response
            prompt_tokens = response['usage'].get('prompt_tokens', 0)
            completion_tokens = response['usage'].get('completion_tokens', 0)
            total_tokens = response['usage'].get('total_tokens', 0)
            costo_total = ((prompt_tokens * 0.00001) + (completion_tokens * 0.00003))
            st.write(f"🔹 Tokens de entrada: {prompt_tokens}")
            st.write(f"🔹 Tokens de salida: {completion_tokens}")
            st.write(f"🔹 Costo total: {costo_total} USD")
    
    
if __name__ == "__main__":
    main()
