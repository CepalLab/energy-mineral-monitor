import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

def parse_embedding(embedding_str):
    try:
        cleaned_str = embedding_str.strip().replace('[', '').replace(']', '').replace('\n', '')
        return np.fromstring(cleaned_str, sep=' ', dtype=np.float32)
    except Exception:
        return np.zeros(384, dtype=np.float32)

@st.cache_data
def load_data_and_process_embeddings():
    try:
        df = pd.read_csv("df_news_final.csv")
        embeddings_list = [parse_embedding(emb) for emb in df['embedding']]
        df['embedding_vector'] = embeddings_list
        return df
    except FileNotFoundError:
        st.error("Error: El archivo 'df_news_final.csv' no fue encontrado. Asegúrate de que esté en el mismo directorio que la aplicación.")
        return None

def run_search():
    st.header("Buscador Semántico Avanzado")
    st.markdown("Este buscador utiliza embeddings vectoriales para encontrar noticias basadas en el significado y contexto de tu consulta. Ajusta el umbral de relevancia para mayor precisión semántica (valores hacia 1)")

    with st.spinner("Cargando modelo de lenguaje y base de datos... Esto puede tardar un momento la primera vez."):
        model = load_embedding_model()
        data = load_data_and_process_embeddings()

    if data is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_input("Buscar noticias:", value="", placeholder="Ejemplo: energía nuclear en Sudamérica")
        with col2:
            relevance_threshold = st.slider("Umbral de Relevancia", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        if user_query:
            with st.spinner('Realizando búsqueda semántica...'):
                query_embedding = model.encode(user_query, convert_to_tensor=True)
                news_embeddings = torch.tensor(np.vstack(data['embedding_vector'].values), dtype=torch.float)
                cosine_scores = util.cos_sim(query_embedding, news_embeddings)
                results_df = data.copy()
                results_df['relevance'] = cosine_scores[0].cpu().numpy()
                relevant_results_df = results_df[results_df['relevance'] >= relevance_threshold]

                if not relevant_results_df.empty:
                    relevant_results_df['publishDate'] = pd.to_datetime(relevant_results_df['publishDate'], errors='coerce')
                    final_results_df = relevant_results_df.sort_values(by='publishDate', ascending=False, na_position='last')
                    st.success(f"Se encontraron **{len(final_results_df)}** resultados con una relevancia mayor a **{relevance_threshold}**.")

                    for index, row in final_results_df.iterrows():
                        st.subheader(row['title'])
                        date_display = row['publishDate'].strftime('%Y-%m-%d') if pd.notna(row['publishDate']) else 'Fecha no disponible'
                        st.markdown(f"**Relevancia:** {row['relevance']:.2f} | **Fuente:** {row['source']} | **Fecha:** {date_display}")
                        content_snippet = row['content']
                        if isinstance(content_snippet, str):
                            st.write(content_snippet[:500] + ("..." if len(content_snippet) > 500 else ""))
                        st.markdown(f"[Leer noticia completa]({row['url']})", unsafe_allow_html=True)
                        st.write("---")
                else:
                    st.warning("No se encontraron noticias que superen el umbral de relevancia seleccionado.")
    else:
        st.info("La carga de datos ha fallado. Por favor, verifica el error de arriba.")

