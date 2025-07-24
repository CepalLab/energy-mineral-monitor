import streamlit as st
import pandas as pd

# Function to get keywords from the user's query
def get_keywords_from_query(query):
    """
    Extracts keywords from the user query.
    """
    return query.lower().split()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("df_news_final.csv")
        df['search_text'] = (df['cleaned_title'].astype(str) + ' ' + df['cleaned_content'].astype(str)).str.lower()
        return df
    except FileNotFoundError:
        st.error("Error: El archivo 'df_news_final.csv' no fue encontrado. Asegúrate de que esté en el mismo directorio que la aplicación.")
        return None

def run_search():
    st.header("Buscador Inteligente de Noticias")
    st.markdown("Escribe tu consulta en lenguaje natural. Por ejemplo: `órdenes ejecutivas de Trump sobre minerales`.")

    data = load_data()

    if data is not None:
        user_query = st.text_input("Buscar noticias:",value="", placeholder="Ejemplo: energía nuclear en Sudamérica")

        if user_query:
            with st.spinner('Buscando noticias relevantes...'):
                keywords = get_keywords_from_query(user_query)
                masks = [data['search_text'].str.contains(keyword, na=False) for keyword in keywords]
                combined_mask = pd.concat(masks, axis=1).any(axis=1)
                results_df = data[combined_mask].copy()

                if not results_df.empty:
                    results_df['relevance'] = results_df['search_text'].apply(lambda text: sum(keyword in text for keyword in keywords))
                    results_df = results_df.sort_values(by='relevance', ascending=False)
                    st.success(f"Se encontraron **{len(results_df)}** resultados para tu búsqueda.")

                    for index, row in results_df.head(15).iterrows():
                        st.subheader(row['title'])
                        st.markdown(f"**Fuente:** {row['source']} | **Fecha de Publicación:** {row['publishDate']}")
                        content_snippet = row['content']
                        if isinstance(content_snippet, str):
                            st.write(content_snippet[:500] + "..." if len(content_snippet) > 500 else content_snippet)
                        st.markdown(f"[Leer noticia completa]({row['url']})", unsafe_allow_html=True)
                        st.write("---")
                else:
                    st.warning("No se encontraron noticias que coincidan con tu búsqueda. Intenta con otros términos.")
    else:
        st.info("La carga de datos ha fallado. Por favor, verifica el error de arriba.")
