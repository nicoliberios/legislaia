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
import fitz
import pandas as pd
import openai
from fpdf import FPDF
import base64
import json
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()

SALUDOS = "Hola en que puedo asistirte, ¡Saludos! ¿Cómo puedo ayudarte hoy?, ¡Bienvenido/a! ¿Cómo puedo asistirte?, ¡Qué gusto verte por aquí! ¿Cómo puedo ayudarte hoy? "
NOMBRE_DE_LA_EMPRESA = "Corporación Write"
NOMBRE_AGENTE = "Kliofer"
prompt_inicial = f"""Soy {NOMBRE_AGENTE}, parte del equipo de {NOMBRE_DE_LA_EMPRESA}, un asistente inteligente diseñado para resolver casos basado en normas ISO."""

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

if 'response' not in st.session_state:
    st.session_state.response = None

def conexion_a_mongo():
    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://liberionicolas:nnERjbqYVaA3U2rT@clusterlegislacion.ahirmsy.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLegislacion")
    client = MongoClient(MONGO_URI)
    db = client['db-art-iso-leydpdd']
    collection = db['collection-leydpdd']
    return collection

def guardar_chat_en_mongo(user, message, response, case_use=None, user_solution1=None, user_solution2=None):
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

def main():
    if "knowledgeBase_summary" not in st.session_state:
     st.session_state["knowledgeBase_summary"] = ""

    texto_pdf = st.session_state["knowledgeBase_summary"]
    st.sidebar.markdown("<h3 style='text-align: center; font-size: 36px;'>LEGISLACIÓN</h3>",unsafe_allow_html=True)
    st.sidebar.markdown("**Autores:**\n- *Gabriela Tumbaco*\n- *Gabriel Ruales*\n- *Nicolas Liberio*")
    st.sidebar.image('bott.jpg', width=250)
    st.markdown('<h1 style="color: #FFD700;">LEOPOLDO</h1>', unsafe_allow_html=True)

    mongo_uri = "mongodb+srv://liberionicolas:nnERjbqYVaA3U2rT@clusterlegislacion.ahirmsy.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLegislacion"
    db_name = "db-art-iso-leydpdd"
    collection_name = "collection-leydpdd"
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    if data:
        for doc in data:
            doc.pop("_id", None)
        df = pd.DataFrame(data)
        st.write("📊 **MATRIZ**")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("La colección está vacía.")

    if "knowledgeBase" not in st.session_state:
        st.session_state["knowledgeBase"] = None

    tab1, tab2 = st.tabs(["📜 Aplicabilidad", "💬 Bot"])

    with tab1:
        #st.markdown("## 📜 Costos y Tokens")
        if st.session_state.response:
            response = st.session_state.response
            prompt_tokens = response['usage'].get('prompt_tokens', 0)
            completion_tokens = response['usage'].get('completion_tokens', 0)
            total_tokens = response['usage'].get('total_tokens', 0)
            costo_total = ((prompt_tokens * 0.00001) + (completion_tokens * 0.00003))
            st.write(f"🔹 Tokens de entrada: {prompt_tokens}")
            st.write(f"🔹 Tokens de salida: {completion_tokens}")
            st.write(f"🔹 Costo total: {costo_total} USD")

        #st.markdown("## 💬 Historial de Conversación")
        st.write(st.session_state.memory.buffer)

        st.markdown("## 📋 Análisis de Caso de Uso según la Ley de Protección de Datos")
        caso_uso = st.text_area("🔍 Pega aquí el caso de uso que deseas analizar", height=200)

        if st.button("📊 Analizar Aplicabilidad de la Ley"):
            if not caso_uso.strip():
                st.warning("Por favor ingresa un caso de uso para analizar.")
            else:
                st.info("🔎 Analizando aplicabilidad con GPT...")
                capitulos = [f"Capítulo {i}" for i in range(1, 13)]
                # Supongamos que "texto_pdf" contiene el resumen o los fragmentos más relevantes extraídos del PDF
                texto_pdf = st.session_state["knowledgeBase_summary"]  # o el nombre que uses para guardar el contenido

                prompt_ley = f"""
Usa esta información oficial extraída de la Ley Orgánica de Protección de Datos Personales de Ecuador (LOPDP):

{texto_pdf}

Analiza el siguiente caso de uso de acuerdo con los 12 capítulos de la ley.
Para cada capítulo, indica qué tan aplicable es en una escala del 0 al 100%.

Luego, en un solo párrafo, indica cuál es el capítulo con menor aplicabilidad y sugiere recomendaciones concretas, basadas en el contenido arriba, para fortalecer la protección de datos en ese capítulo.

Caso de uso:
{caso_uso}

Devuelve la respuesta en este formato JSON:

{{
  "Capítulo 1": 85,
  "Capítulo 2": 60,
  ...
  "Capítulo 12": 95
}}

"""
#Resumen: El capítulo que menos se aplica es el Capítulo X, se recomienda fortalecer con [recomendaciones].


                response_json = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "Eres un analista legal experto en la Ley de Protección de Datos Personales del Ecuador."},
                        {"role": "user", "content": prompt_ley}
                    ],
                    api_key=os.getenv("OPENAI_API_KEY"),
                )

                try:
                    contenido = response_json['choices'][0]['message']['content']
                    match = re.search(r"\{[\s\S]+\}", contenido)
                    if match:
                        data = json.loads(match.group())
                        st.success("✅ Análisis completado.")

                        # Mostrar tabla y gráfico igual que antes
                        df_resultado = pd.DataFrame(list(data.items()), columns=["Capítulo", "Aplicabilidad (%)"])
                        st.dataframe(df_resultado, use_container_width=True)

                        labels = list(data.keys())
                        values = list(data.values())
                        values += values[:1]
                        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                        angles += angles[:1]

                        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                        ax.plot(angles, values, 'o-', linewidth=2, label='Aplicabilidad')
                        ax.fill(angles, values, alpha=0.25)
                        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
                        ax.set_title("Grado de Aplicabilidad por Capítulo", size=14)
                        ax.set_ylim(0, 100)
                        st.pyplot(fig)
                    else:
                        st.error("❌ No se pudo interpretar correctamente la respuesta del modelo.")
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar la respuesta: {e}")

   
    with tab2:
        uploaded_files = st.file_uploader("📂 Sube archivos PDF", type=["pdf"], accept_multiple_files=True)
        text = ""
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    text += "".join([page.get_text("text") for page in doc])

            st.markdown("### 📝 Texto extraído de los PDF cargados:")
            st.write(text[:1000] + "..." if len(text) > 1000 else text)

            pregunta = st.text_input("💡 Haz tu pregunta respecto al contenido cargado")

            if st.button("🤖 Preguntar"):
                if not pregunta.strip():
                    st.warning("Por favor escribe una pregunta válida.")
                else:
                    prompt = f"""
Eres un asistente experto en normas ISO y gestión de riesgos. Utiliza el siguiente texto extraído de documentos para responder la pregunta que se te haga.

Texto extraído:
{text}

Pregunta:
{pregunta}

Responde con claridad y precisión.
"""

                    messages = [{"role": "system", "content": prompt_inicial}]
                    messages.append({"role": "user", "content": prompt})

                    with get_openai_callback() as obtienec:
                        st.session_state.response = openai.ChatCompletion.create(
                            model="gpt-4-turbo",
                            messages=messages,
                            api_key=os.getenv("OPENAI_API_KEY"),
                        )

                    answer = st.session_state.response['choices'][0]['message']['content'] if st.session_state.response.get('choices') else "Lo siento, no pude obtener una respuesta."
                    guardar_chat_en_mongo("user1", pregunta, answer)

                    st.markdown("### ✅ Respuesta:")
                    st.write(answer)

                    pdf_bytes = generar_pdf(answer)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="respuesta_IA.pdf">📄 Descargar respuesta en PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
