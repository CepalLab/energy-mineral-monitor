import streamlit as st
import pandas as pd
import plotly.express as px
import collections
from datetime import date
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Monitor de Noticias de Energía y Minerales",
    page_icon="📊",
    layout="wide",
)

# --- Data Loading ---
@st.cache_data
def load_data(path):
    """Loads the news data from a CSV file."""
    try:
        df = pd.read_csv(path)
        df['publishDate'] = pd.to_datetime(df['publishDate'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: No se pudo encontrar el archivo en la ruta: {path}")
        return None

df_news = load_data('df_news_final.csv')

if df_news is not None:
    # --- Sidebar Navigation ---
    st.sidebar.title("Navegación")
    analysis_option = st.sidebar.radio(
        "Selecciona una sección:",
        (
            "Metodología",
            "Distribuciones",
            "Top 20 Noticias Relevantes",
            "Noticias Anómalas",
            "Análisis de Palabras Clave",
            "Base de Datos",
        ),
    )

    # --- Semantic Search Option ---
    st.sidebar.subheader("🔎 Opciones de Búsqueda Semántica")
    
    semantic_search_mode = st.sidebar.selectbox(
        "Selecciona una opción (para volver a secciones superiores selecciona opción 'Desactivada'):",
        ["Desactivada", "Palabras Clave", "Embeddings Avanzados"]
    )

    # --- Main Content ---
    st.title("Monitor de Noticias de Energía y Minerales")
    st.markdown("Resultados preliminares de proceso experimental de captura y análisis de noticias.")

    # --- Integración de buscadores semánticos ---
    if semantic_search_mode != "Desactivada":
        if semantic_search_mode == "Palabras Clave":
            import semantic_search_app
            semantic_search_app.run_search()
        elif semantic_search_mode == "Embeddings Avanzados":
            import semantic_search_pro
            semantic_search_pro.run_search()
        st.stop()

    # Cargar los títulos de los topics
    with open('topic_titles.json', encoding='utf-8') as f:
        topic_titles = json.load(f)

    if analysis_option == "Metodología":
        st.header("Metodología de Recolección y Procesamiento de Noticias")

        st.markdown("""
Se diseñó un proceso de búsqueda automatizada a partir de un conjunto de consultas predefinidas en inglés y español, orientadas a temas relacionados con minerales críticos, energía y geopolítica de éstos. Las consultas se ejecutaron en un metabuscador, el cual permite buscar simultáneamente en varios motores de búsqueda. Para cada set de consultas, se especificó el idioma correspondiente y se configuró la búsqueda para obtener únicamente los primeros resultados de cada consulta, sin restricción de fecha (aunque es posible ampliar la búsqueda a más páginas de resultados, en este caso experimental se limitó a la primera página de resultados por cada consulta).

*Para este ejercicio experimental se ejecutó el proceso el 13 de julio de 2025, obteniendo resultados de noticias más recientes hasta esa fecha.*
                    
**Consultas utilizadas en español:**
- 'órdenes ejecutivas de Trump respecto a minerales críticos'
- 'minerales críticos'
- 'aranceles estados unidos a minerales críticos'
- 'órdenes ejecutivas de Trump respecto a energía, seguridad energética, autosuficiencia energética'
- 'decisiones y órdenes ejecutivas de estados unidos respecto a desregulación del sector energético'
- 'políticas y decisiones de fomento a combustibles fósiles'
- 'restricciones a la exportación de minerales críticos China'
- 'incentivos a la producción de minerales estratégicos Estados Unidos Unión Europea'
- 'geopolítica de minerales críticos'
- 'tierras raras'
- 'minería de aguas profundas'

**Consultas utilizadas en inglés:**
- 'Trump executive orders on critical minerals'
- 'critical minerals'
- 'U.S. tariffs on critical minerals'
- 'Trump executive orders on energy, energy independence, and energy security'
- 'U.S. executive decisions and orders on energy sector deregulation'
- 'policies and decisions supporting fossil fuel development'
- 'China export restrictions on critical minerals OR rare earths'
- 'US or EU incentives for critical minerals production OR supply chain resilience'
- 'geopolitics of critic minerals'
- 'rare earth'
- 'deep sea mining'

En total, se realizaron 11 consultas por cada idioma, obteniendo solo la primera página de resultados para cada una.

### Resultados

El proceso arrojó un total de **1617 registros**. Dado que los motores de búsqueda tienden a priorizar la información más reciente, aproximadamente el **60% de los datos corresponden al año 2025** (y el 90% de los datos con fecha no vacía, ya que existen más de 500 registros sin fecha en los metadatos).

La base de datos resultante contiene 23 columnas, entre las que destacan:  
- **url**: enlace a la noticia  
- **title**: título  
- **content**: contenido  
- **engine**: motor de búsqueda  
- **source**: fuente  
- **publishDate**: fecha de publicación  
- **query**: consulta utilizada  
- **language**: idioma  
- **topic_id**: identificador temático  
- **enhanced_relevance_score**: puntaje de relevancia  
- **anomaly_label**: etiqueta de anomalía  
- ...entre otras variables útiles para el análisis exploratorio y temático.
    """)

    if analysis_option == "Distribuciones":
        st.header("Análisis de Distribuciones")
        st.write("Visualización de la distribución de variables en el conjunto de datos. Se incluyen variables categóricas como idioma, consulta realizada, motor de búsqueda, fuente. Además se presentan agrupaciones automáticas por clúster temático, así como la distribución por fecha de publicación.")

        # --- Categorical Variables ---
        st.subheader("Variables Categóricas")
        categorical_cols = ["language", "query", "engine", "source"]
        for col in categorical_cols:
            if col == "query":
                # Contar ocurrencias y asociar idioma principal
                query_counts = (
                    df_news.groupby('query')
                    .agg(count=('query', 'size'), language=('language', lambda x: x.mode()[0] if not x.mode().empty else ''))
                    .reset_index()
                )
                max_label_length = 30
                query_counts['query_trunc'] = query_counts['query'].apply(
                    lambda x: x[:max_label_length] + '...' if len(x) > max_label_length else x
                )
                color_map = {
                    'es': '#4FC3F7',      # azul claro
                    'español': '#4FC3F7',
                    'en': '#1565C0',      # azul oscuro
                    'inglés': '#1565C0'
                }
                fig = px.bar(
                    query_counts,
                    x='query_trunc',
                    y='count',
                    color='language',
                    color_discrete_map=color_map,
                    title="Distribución por query",
                    labels={'query_trunc': 'Query', 'count': 'Cantidad', 'language': 'Idioma'},
                    hover_data=['query', 'language'],
                )
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig)
            elif col == "source":
                # Tu código actual para source (barras simples, sin color por idioma)
                counts = df_news[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                max_label_length = 30
                counts[f'{col}_trunc'] = counts[col].apply(
                    lambda x: x[:max_label_length] + '...' if len(x) > max_label_length else x
                )
                fig = px.bar(
                    counts,
                    x=f'{col}_trunc',
                    y='count',
                    title=f"Distribución por {col}",
                    labels={f'{col}_trunc': col.capitalize(), 'count': 'Cantidad'},
                    hover_data=[col],
                )
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig)
            else:
                fig = px.pie(df_news, names=col, title=f"Distribución por {col}", hole=0.3)
                st.plotly_chart(fig)

        # --- Numerical Variables ---
        st.subheader("Clústeres Temáticos")
        st.write("Visualización grupos de noticias a través de clustering automático.")
        # Crear columna descriptiva para topic_id
        df_news['topic_label'] = df_news['topic_id'].astype(str).map(topic_titles)

        # Calcular el conteo por tema y ordenar de mayor a menor
        topic_counts = df_news['topic_label'].value_counts().sort_values(ascending=False)
        ordered_labels = topic_counts.index.tolist()

        # Graficar el histograma con el eje x ordenado
        fig_topic = px.histogram(
            df_news,
            x="topic_label",
            category_orders={"topic_label": ordered_labels},
            title="Distribución por Tema (agrupación no supervisada)",
            labels={"topic_label": "Tema", "count": "Cantidad"}
        )
        fig_topic.update_xaxes(type='category', categoryorder='array', categoryarray=ordered_labels)
        fig_topic.update_layout(bargap=0.2)
        st.plotly_chart(fig_topic)

        st.markdown("### Distribución por fecha de publicación")
        df_filtered_date = df_news.dropna(subset=['publishDate']).copy()
        df_filtered_date['year_month'] = df_filtered_date['publishDate'].dt.to_period('M').astype(str)
        df_filtered_date['date_str'] = df_filtered_date['publishDate'].dt.strftime('%Y-%m-%d')

        # Definir el rango de fechas para el slider (convertir a date)
        min_date = df_filtered_date['publishDate'].min().date()
        max_date = df_filtered_date['publishDate'].max().date()

        # Slider para seleccionar el rango de fechas
        date_range = st.slider(
            "Selecciona el rango de fechas:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )

        # Filtrar el DataFrame según el rango seleccionado
        df_filtered_date = df_filtered_date[
            (df_filtered_date['publishDate'].dt.date >= date_range[0]) &
            (df_filtered_date['publishDate'].dt.date <= date_range[1])
        ]

        # Decidir el nivel de agregación
        range_days = (date_range[1] - date_range[0]).days
        if range_days > 60:
            x_col = 'year_month'
            x_title = "Año-Mes"
        else:
            x_col = 'date_str'
            x_title = "Fecha"

        # Ordenar el DataFrame por fecha
        df_filtered_date = df_filtered_date.sort_values('publishDate')

        # Graficar
        fig_date = px.histogram(
            df_filtered_date,
            x=x_col,
            title=f"Distribución por Fecha de Publicación ({x_title})",
            labels={x_col: x_title, "count": "Cantidad"}
        )
        fig_date.update_xaxes(type='category', categoryorder='category ascending')
        st.plotly_chart(fig_date)


    elif analysis_option == "Top 20 Noticias Relevantes":
        st.header("Top 20 Noticias por Relevancia (ver detalles de scoring)")

        with st.expander("ℹ️ Detalles del scoring"):
            st.markdown("""
## Sistema experimental de Scoring para Noticias

El sistema de scoring evalúa la relevancia de noticias según múltiples criterios ponderados, generando un puntaje normalizado entre 0 y 1 para priorizar información estratégica.

---

## Componentes del Sistema de Scoring

### 1. **Criterios de Evaluación y Pesos**

| **Categoría** | **Peso** | **Descripción** | **Ejemplos de Términos** |
|---|---|---|---|
| **🌎 América Latina** | 3.0 | Enfoque geográfico prioritario | "latin america", "chile", "américa latina", "caribe" |
| **📋 Órdenes Ejecutivas** | 2.8 | Decisiones presidenciales directas | "executive order", "orden ejecutiva", "presidential directive" |
| **⚡ Minerales Críticos** | 2.5 | Recursos estratégicos clave | "critical minerals", "lithium", "minerales críticos", "litio" |
| **🔋 Energías Renovables** | 2.2 | Transición energética | "renewable energy", "solar", "energía renovable", "eólica" |
| **🏛️ Políticas Generales** | 2.0 | Políticas estadounidenses | "policy change", "tariff", "cambio de política", "arancel" |
| **⚙️ Política Energética** | 2.0 | Estrategias energéticas | "energy policy", "energy security", "política energética" |
| **⛏️ Actividades Mineras** | 1.8 | Operaciones extractivas | "mining operations", "extraction", "operaciones mineras" |
| **🛢️ Combustibles Fósiles** | 1.5 | Energías tradicionales | "fossil fuels", "oil production", "combustibles fósiles" |
| **🥉 Bonificación Cobre** | 1.2 | Mineral específico de interés | "copper", "cobre" |

---

### 2. **Autoridad de Fuentes**

| **Nivel** | **Peso** | **Tipo de Fuente** | **Ejemplos** |
|---|---|---|---|
| **Tier 1** | 2.0 | Gubernamental y financiera premium | whitehouse.gov, reuters.com, bloomberg.com |
| **Tier 2** | 1.5 | Medios establecidos | nytimes.com, wsj.com, bbc.com, cnn.com |
| **Tier 3** | 1.2 | Fuentes especializadas | mining.com, energycentral.com, law360.com |

---

### 3. **Relevancia Temporal**

| **Antigüedad** | **Multiplicador** | **Descripción** |
|---|---|---|
| **0-7 días** | 0.9-1.0 | Máxima relevancia |
| **8-15 días** | 0.7-0.9 | Alta relevancia |
| **16-30 días** | 0.4-0.7 | Relevancia media |
| **30+ días** | 0.1-0.4 | Relevancia mínima |
| **Fecha faltante** | 0.7 | Penalización por dato incompleto |

---

### 4. **Factores Adicionales**

| **Factor** | **Peso** | **Condición** |
|---|---|---|
| **Coincidencia con Query** | +0.8 | Términos relevantes en consulta original |
| **Multiidioma** | N/A | Detección automática inglés/español |
| **Normalización** | ÷10 | Score final entre 0-1 |

---

## 🔍 Metodología de Búsqueda

### **Técnicas Implementadas:**
- **Word Boundaries**: Evita coincidencias parciales (`\b palabra \b`)
- **Normalización**: Conversión a minúsculas y limpieza
- **Búsqueda Multiidioma**: Términos en inglés y español
- **Validación de Datos**: Manejo robusto de valores nulos

---

## 📈 Interpretación de Resultados

### **Rangos de Score:**
| **Rango** | **Interpretación** | **Acción Recomendada** |
|---|---|---|
| **0.8 - 1.0** | Altamente relevante | Prioridad máxima |
| **0.6 - 0.8** | Muy relevante | Revisión prioritaria |
| **0.4 - 0.6** | Moderadamente relevante | Revisión secundaria |
| **0.2 - 0.4** | Poco relevante | Monitoreo |
| **0.0 - 0.2** | Baja relevancia | Archivo |

---

## 🎯 Ejemplo de Cálculo

### **Noticia hipotética:**
*"Trump firma orden ejecutiva sobre minerales críticos en América Latina"*

| **Componente** | **Coincidencia** | **Peso** | **Puntos** |
|---|---|---|---|
| América Latina | ✅ | 3.0 | 3.0 |
| Órdenes Ejecutivas | ✅ | 2.8 | 2.8 |
| Minerales Críticos | ✅ | 2.5 | 2.5 |
| Fuente (Reuters) | ✅ | 2.0 | 2.0 |
| Relevancia Temporal | 2 días | x0.95 | - |
| **Score Bruto** | | | **9.8** |
| **Score Normalizado** | | ÷10 x0.95 | **0.93** |

---

## ⚙️ Configuración del Sistema

### **Parámetros Ajustables:**
- **Time Decay Factor**: 0.05 (menos agresivo)
- **Max Days Relevant**: 30 días
- **Date Penalty**: 0.7 (70% del score)
- **Query Bonus**: 0.8 puntos adicionales

### **Ventajas del Sistema:**
✅ **Objetividad**: Criterios cuantificables y transparentes  
✅ **Flexibilidad**: Pesos ajustables según prioridades  
✅ **Escalabilidad**: Procesamiento automatizado de grandes volúmenes  
✅ **Trazabilidad**: Registro detallado de componentes del score  

---

## 📋 Outputs del Sistema

### **Métricas Generadas:**
- Score de relevancia normalizado (0-1)
- Desglose por componentes
- Estadísticas descriptivas
- Ranking de noticias prioritarias
- Alertas por calidad de datos

### **Casos de Uso:**
1. **Priorización diaria** de noticias para analistas
2. **Alertas automáticas** para eventos críticos
3. **Dashboards ejecutivos** con métricas clave
4. **Análisis histórico** de tendencias informativas
            """)

        top_20_news = df_news.nlargest(20, 'enhanced_relevance_score')
        st.data_editor(top_20_news[['title', 'content', 'url','enhanced_relevance_score']],
                       column_config={
                           "url": st.column_config.LinkColumn("URL")
                       },
                       hide_index=True
                       )


    elif analysis_option == "Noticias Anómalas":
        st.header("Listado de Noticias 'Anómalas'")
        st.warning("Estas noticias fueron etiquetadas como anómalas por un modelo de detección de anomalías. Se definen así porque pueden ser noticias poco comunes y relevantes a las consultas o noticias que no tienen relación con los temas estudiadas recuperadas erroneámente por los motores de búsqueda. Estas requieren revisión manual para determinar su relevancia.")
        anomalous_news = df_news[df_news['anomaly_label'] == -1].copy()
        anomalous_news = anomalous_news[['url', 'title']]
        st.data_editor(
            anomalous_news,
            column_config={
                "url": st.column_config.LinkColumn("URL")
            },
            hide_index=True
        )


    elif analysis_option == "Análisis de Palabras Clave":
        st.header("Top Palabras Clave por Tema")
        st.subheader("Análisis de set de palabras clave predifinidas")
        st.write("Para este análisis se utilizaron sets de palabras clave predefinidas para identificar términos relevantes en el texto completo de las noticias. Estos sets fueron diseñados para capturar temas relacionados con minería y energía, puede mejorarse con sets de palabras clave más específicos o personalizados según el contexto de interés.")

        # --- Keyword Processing para texto completo ---
        def find_terms_in_text(text, terms):
            """Busca términos en un texto y retorna los que aparecen."""
            if not isinstance(text, str):
                return []
            text_lower = text.lower()
            found_terms = [term for term in terms if term in text_lower]
            return found_terms

        # --- Mining Keywords ---
        st.subheader("Minería")
        mining_terms = [
            "minería", "mining", "mineral", "mineral", "cobre", "copper",
            "extracción minera", "mining extraction", "oro", "gold",
            "plata", "silver", "litio", "lithium"
        ]
        df_news['mining_keywords'] = df_news['preprocessed_text_for_embeddings'].apply(
            lambda x: find_terms_in_text(x, mining_terms)
        )
        mining_keywords_list = [kw for sublist in df_news['mining_keywords'] for kw in sublist]
        mining_keyword_counts = collections.Counter(mining_keywords_list)

        if mining_keyword_counts:
            df_mining_keywords = pd.DataFrame(mining_keyword_counts.most_common(20), columns=['Keyword', 'Count'])
            fig_mining = px.bar(df_mining_keywords, x='Keyword', y='Count', title="Top 20 Palabras Clave de Minería")
            st.plotly_chart(fig_mining)
        else:
            st.write("No se encontraron palabras clave de minería con los términos de búsqueda.")

        # --- Energy Keywords ---
        st.subheader("Energía")
        energy_terms = [
            "energía", "energy", "energía verde", "green energy", "renovables", "renewables",
            "solar", "solar", "eólica", "wind", "wind power", "combustibles fósiles", "fossil fuels",
            "petróleo", "oil", "petroleum", "carbón", "coal"
        ]
        df_news['energy_keywords'] = df_news['preprocessed_text_for_embeddings'].apply(
            lambda x: find_terms_in_text(x, energy_terms)
        )
        energy_keywords_list = [kw for sublist in df_news['energy_keywords'] for kw in sublist]
        energy_keyword_counts = collections.Counter(energy_keywords_list)

        if energy_keyword_counts:
            df_energy_keywords = pd.DataFrame(energy_keyword_counts.most_common(20), columns=['Keyword', 'Count'])
            fig_energy = px.bar(df_energy_keywords, x='Keyword', y='Count', title="Top 20 Palabras Clave de Energía")
            st.plotly_chart(fig_energy)
        else:
            st.write("No se encontraron palabras clave de energía con los términos de búsqueda.")

    elif analysis_option == "Base de Datos":
        st.header("Base de Datos de Noticias")
        st.write("Esta sección permite explorar la base de datos de noticias procesadas. Se exluyen noticias sin fecha de publicación y se muestran las columnas más relevantes para el análisis.")

        # Crear columna descriptiva para topic_id
        df_news['topic_label'] = df_news['topic_id'].astype(str).map(topic_titles)

        # Inicializar filtros en session_state
        if "title_filter" not in st.session_state:
            st.session_state.title_filter = ""
        if "date_filter" not in st.session_state:
            min_date, max_date = df_news['publishDate'].min(), df_news['publishDate'].max()
            st.session_state.date_filter = (min_date.date(), max_date.date())
        if "topic_filter" not in st.session_state:
            st.session_state.topic_filter = "Todos"

        # Botón para limpiar filtros
        if st.button("Limpiar filtros"):
            st.session_state.title_filter = ""
            min_date, max_date = df_news['publishDate'].min(), df_news['publishDate'].max()
            st.session_state.date_filter = (min_date.date(), max_date.date())
            st.session_state.topic_filter = "Todos"

        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            title_filter = st.text_input("Buscar por título (contiene):", st.session_state.title_filter, key="title_filter")
        with col2:
            min_date, max_date = df_news['publishDate'].min(), df_news['publishDate'].max()
            date_filter = st.date_input(
                "Filtrar por fecha de publicación:",
                value=st.session_state.date_filter,
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="date_filter"
            )
        with col3:
            topic_options = ["Todos"] + [f"{k}: {v}" for k, v in topic_titles.items()]
            topic_filter = st.selectbox("Filtrar por Topic:", topic_options, key="topic_filter")

        # Aplicar filtros
        df_filtered = df_news.copy()
        if st.session_state.title_filter:
            df_filtered = df_filtered[df_filtered['title'].str.contains(st.session_state.title_filter, case=False, na=False)]
        if st.session_state.date_filter:
            df_filtered = df_filtered[
                (df_filtered['publishDate'].dt.date >= st.session_state.date_filter[0]) &
                (df_filtered['publishDate'].dt.date <= st.session_state.date_filter[1])
            ]
        if st.session_state.topic_filter != "Todos":
            topic_id_selected = st.session_state.topic_filter.split(":")[0]
            df_filtered = df_filtered[df_filtered['topic_id'].astype(str) == topic_id_selected]

        # Seleccionar columnas a mostrar
        columns_to_show = [
            "url", "title", "content","publishDate", "source", "enhanced_relevance_score", "topic_id", "topic_label"
        ]
        df_filtered = df_filtered[columns_to_show]

        # Texto dinámico de resultados
        st.markdown(f"**Mostrando {len(df_filtered)} resultados.**")

        # Mostrar la tabla
        st.data_editor(
            df_filtered,
            column_config={
                "url": st.column_config.LinkColumn("URL"),
                "topic_label": st.column_config.TextColumn("Tema")
            },
            hide_index=True
        )

else:
    st.warning("No se pudieron cargar los datos. Por favor, verifica la ruta del archivo.")

