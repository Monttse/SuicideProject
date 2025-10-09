import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gdown
import json
import os
import pydeck as pdk

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Perfiles de Riesgo de Suicidio MX", layout="wide")

# --- VARIABLES Y ARCHIVOS ---
K_OPTIMO = 5 
PERFILES_PATH = 'perfiles.csv'       
TSNE_DATA_PATH = 'tsne_3d_data.json' # Nuevo archivo de datos 3D
GEOJSON_PATH = 'mexico.json'
DF_FILE_ID = '1li-MLpM6vpkgwLkvv2TLRqNhR_kqDWnp' # ID de Google Drive para el DataFrame principal

# --- MAPEOS ---
ESTADO_NOMBRES = {
    '01': 'Aguascalientes', '02': 'Baja California', '03': 'Baja California Sur', 
    '04': 'Campeche', '05': 'Coahuila', '06': 'Colima', '07': 'Chiapas', 
    '08': 'Chihuahua', '09': 'Ciudad de México', '10': 'Durango', '11': 'Guanajuato', 
    '12': 'Guerrero', '13': 'Hidalgo', '14': 'Jalisco', '15': 'México', 
    '16': 'Michoacán', '17': 'Morelos', '18': 'Nayarit', '19': 'Nuevo León', 
    '20': 'Oaxaca', '21': 'Puebla', '22': 'Querétaro', '23': 'Quintana Roo', 
    '24': 'San Luis Potosí', '25': 'Sinaloa', '26': 'Sonora', '27': 'Tabasco', 
    '28': 'Tamaulipas', '29': 'Tlaxcala', '30': 'Veracruz', '31': 'Yucatán', 
    '32': 'Zacatecas'
}

MES_NOMBRES = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 
               8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}


CLUSTER_NOMBRES = {
    0: 'Adulto Joven - riesgo nocturno',
    1: 'Adulto Joven - riesgo vespertino',
    2: 'Adulto Joven - riesgo no especificado',
    3: 'Adulto Joven - sin ocupación',
    4: 'Adulto Mayor - analfabetismo' 
}

CLUSTER_DESCRIPCIONES = {
    0: 'Adulto promedio (32 años), ocupado, con riesgo concentrado en la **madrugada** (00-05).',
    1: 'Adulto promedio (32 años), ocupado, con alto riesgo en la **noche** (18-23) al finalizar la jornada.',
    2: 'Adulto promedio (30 años), con gran cantidad de datos **no especificados**, y riesgo en la madrugada (00-05).',
    3: '**Joven** (22 años), **desempleado** o inactivo. El foco principal de riesgo, concentrado en la **noche** (18-23).',
    4: '**Adulto mayor** (60 años), ocupado, con riesgo concentrado en la **tarde** (12-17).'
}


# -----------------------------------------------------------
# --- FUNCIONES DE CARGA DE DATOS ---
# -----------------------------------------------------------

@st.cache_data
def load_geojson(path):
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error fatal al cargar GeoJSON desde GitHub: {e}") 
        return None

@st.cache_data
def load_data(file_id):
    output_path_df = "temp_df.parquet"
    with st.spinner('Cargando datos principales desde Google Drive...'):
        try:
            gdown.download(id=file_id, output=output_path_df, quiet=True, fuzzy=True)
            df = pd.read_parquet(output_path_df)
            df['ent_resid'] = df['ent_resid'].astype(str).str.zfill(2) 
            os.remove(output_path_df)
            return df
        except Exception as e:
            st.error(f"Error fatal al cargar el DataFrame principal: {e}")
            return None

@st.cache_data
def load_tsne_data(path):
    try:
        # Usamos read_csv si es más fácil de generar en tu entorno, 
        # o un método de lectura de JSON más simple si el 'orient' falla.
        # Asumiremos que si lo cambias a un CSV funciona mejor:
        # Recomiendo guardar en el script de modelado como: df_tsne_3d.to_csv('tsne_3d_data.csv', index=False)
               
        # SI LO GUARDASTE COMO JSON:
        df = pd.read_json(path, orient='records')
        
        # Corregir el tipo de dato de la columna de cluster
        if 'cluster_nombre' in df.columns:
             df['cluster_nombre'] = df['cluster_nombre'].astype(str)
        
        return df
        
    except FileNotFoundError:
        st.warning(f"Advertencia: No se encontró el archivo de datos 3D en {path} o {path.replace('.json', '.csv')}. Intenta generarlo como CSV.")
        return None
    except Exception as e:
        st.error(f"Error al cargar datos t-SNE 3D: {e}")
        return None

# --- LLAMADA INICIAL DE DATOS ---
df_final = load_data(DF_FILE_ID)
# Ajusta la ruta si decidiste guardar como CSV
TSNE_DATA_PATH = 'tsne_3d_data.csv' 
df_tsne_3d = load_tsne_data(TSNE_DATA_PATH)


# ====================================================================
# --- COMIENZA LA INTERFAZ (UI) ---
# ====================================================================

st.title("Sistema de Identificación de Perfiles de Riesgo de Suicidio (2020-2023)")
st.subheader("Modelado no Supervisado (K-Means) en Casos de Suicidio en México")
st.markdown("---")

# --------------------------------------------------------------------------------
# --- SECCIÓN 1: VISUALIZACIONES DESCRIPTIVAS (NUEVA POSICIÓN) ---
# --------------------------------------------------------------------------------
st.header("1. Visualizaciones Descriptivas Clave")
st.markdown("Gráficas que contextualizan las características generales de la población de estudio (2020-2023).")

col1, col2, col3 = st.columns(3)

# GRÁFICA 1: Defunciones Totales
with col1:
    st.subheader("Distribución Mensual")
    try:
        st.image('01.Dist_defunciones.PNG', use_container_width=True)
        st.caption("Distribución de los casos, mostrando la estacionalidad (ej. picos en Marzo y Septiembre).")
    except FileNotFoundError:
        st.warning("No se encontró la imagen: 01.Dist_defunciones.PNG")

# GRÁFICA 2: Edad y Género
with col2:
    st.subheader("Distribución por Edad y Género")
    try:
        st.image('03.Dist_edad_genero_suicide.PNG', use_container_width=True)
        st.caption("Comparativa por grupos de edad, resaltando la mayor vulnerabilidad en el género masculino joven.")
    except FileNotFoundError:
        st.warning("No se encontró la imagen: 03.Dist_edad_genero_suicide.PNG")

# GRÁFICA 3: Nivel Educativo
with col3:
    st.subheader("Distribución por Nivel Educativo")
    try:
        st.image('10.Dist_nivel_educativo_suicide.PNG', use_container_width=True)
        st.caption("Concentración de casos por el nivel educativo formal alcanzado (Primaria, Secundaria, etc.).")
    except FileNotFoundError:
        st.warning("No se encontró la imagen: 10.Dist_nivel_educativo_suicide.PNG")

st.markdown("---")


# --------------------------------------------------------------------------------
# --- SECCIÓN 2: PERFILES DE RIESGO (TABLA) ---
# --------------------------------------------------------------------------------
st.header("2. Perfiles de Riesgo de Suicidio (K=5)")
st.markdown("Tabla resumen que describe las características principales de cada grupo.")

try:
    df_perfiles = pd.read_csv(PERFILES_PATH)
    
    # 1. Mapeo y Renombre
    if 'Mes Frecuente' in df_perfiles.columns:
        df_perfiles['Mes Frecuente'] = df_perfiles['Mes Frecuente'].map(MES_NOMBRES)

    df_perfiles['cluster'] = df_perfiles['cluster'].map(CLUSTER_NOMBRES)
    df_perfiles.rename(columns={'cluster': 'Perfil de Riesgo'}, inplace=True)
    
    # Intenta usar el nombre correcto. Si falla, usa 'tamano_temp' o el que exista.
    try:
        subset_col = 'Tamaño' 
        df_perfiles.style.background_gradient(cmap='YlOrRd', subset=[subset_col])
    except KeyError:
        # Fallback si el nombre 'Tamaño del Cluster' no está en el CSV
        subset_col = 'Tamaño' 


    st.dataframe(
        df_perfiles.style.background_gradient(cmap='YlOrRd', subset=[subset_col]),
        hide_index=True,
        use_container_width=True
    )
    st.caption("Los 5 perfiles identificados por K-Means. El tamaño indica la cantidad de casos en cada grupo.")
except FileNotFoundError:
    st.error(f"⚠️ Error: No se pudo cargar la tabla de perfiles en {PERFILES_PATH}.")
except Exception as e:
    st.error(f"Error al mostrar la tabla de perfiles: {e}")
    
st.markdown("---")


# --------------------------------------------------------------------------------
# --- SECCIÓN 3: MAPA Y ANÁLISIS GEOGRÁFICO ---
# --------------------------------------------------------------------------------
st.header("3. Foco de Intervención Geográfica")

try:
    mx_geojson = load_geojson(GEOJSON_PATH) 

    if df_final is not None and mx_geojson is not None:
        
        # --- SELECTOR DE CLUSTER PARA EL MAPA ---
        st.subheader("Selección de Perfil de Riesgo")
        
        cluster_seleccionado_id = st.selectbox(
            'Visualizar Concentración de:',
            options=list(CLUSTER_NOMBRES.keys()),
            format_func=lambda x: f"Cluster {x}: {CLUSTER_NOMBRES[x]}"
        )
        
        cluster_seleccionado_nombre = CLUSTER_NOMBRES[cluster_seleccionado_id]
        
        # --- DESCRIPCIÓN DEL CLUSTER (Añadido) ---
        st.info(f"**Descripción del Perfil {cluster_seleccionado_id}:** {CLUSTER_DESCRIPCIONES[cluster_seleccionado_id]}")
        
        st.subheader(f"Mapa de Concentración del Perfil: {cluster_seleccionado_nombre}")
        
        # --- LÓGICA DEL MAPA (SIN CAMBIOS) ---
        df_conteo_total = df_final.groupby('ent_resid').size().reset_index(name='Total Casos')
        df_conteo_cluster = df_final[df_final['cluster'] == cluster_seleccionado_id].groupby('ent_resid').size().reset_index(name='Casos Cluster')
        
        df_mapa = pd.merge(df_conteo_total, df_conteo_cluster, on='ent_resid', how='left').fillna(0)
        df_mapa['Porcentaje Cluster'] = (df_mapa['Casos Cluster'] / df_mapa['Total Casos']) * 100
        df_mapa.rename(columns={'ent_resid': 'CVE_ENT'}, inplace=True)
        
        max_porcentaje = df_mapa['Porcentaje Cluster'].max()
        df_mapa['color_intensity'] = (df_mapa['Porcentaje Cluster'] / max_porcentaje * 255).astype(int)

        # Mapeo de colores RGB basado en el Cluster ID (Ajustado a 5 clusters)
        if cluster_seleccionado_id == 0: color_formula = "[0, 0, properties.color_intensity, 160]" # Azul
        elif cluster_seleccionado_id == 1: color_formula = "[0, properties.color_intensity, 0, 160]" # Verde
        elif cluster_seleccionado_id == 2: color_formula = "[properties.color_intensity, properties.color_intensity, 0, 160]" # Amarillo
        elif cluster_seleccionado_id == 3: color_formula = "[properties.color_intensity, 0, 0, 160]" # Rojo (Foco)
        else: color_formula = "[255, 0, properties.color_intensity, 160]" # Magenta (Cluster 4)


        # --- FUSIÓN DE DATOS EN EL GEOJSON Y PYDECK (SIN CAMBIOS) ---
        color_lookup = df_mapa.set_index('CVE_ENT')['color_intensity'].to_dict()
        geojson_data = json.loads(json.dumps(mx_geojson)) 
        
        for feature in geojson_data['features']:
            entidad_id = feature['properties']['CVE_ENT'] 
            color_value = color_lookup.get(entidad_id, 0) 
            feature['properties']['color_intensity'] = color_value
            
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            get_fill_color=color_formula, 
            get_line_color=[0, 0, 0], 
            line_width_min_pixels=1,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=23.6345,
            longitude=-102.5528,
            zoom=4.5, min_zoom=4, max_zoom=10, pitch=0,
        )

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9", 
            initial_view_state=view_state,
            layers=[geojson_layer],
        ))
        
        st.caption(f"El color en el mapa indica la concentración porcentual del Perfil {cluster_seleccionado_id} ({cluster_seleccionado_nombre}) por entidad.")
        st.markdown("---")
        
        # --- ANÁLISIS DETALLADO POR ENTIDAD (Gráfico Plotly) ---
        st.subheader("Análisis Detallado por Entidad")
        
        lista_codigos = sorted(df_final['ent_resid'].unique().tolist())
            
        entidad_seleccionada_codigo = st.selectbox(
            'Selecciona una Entidad de Residencia:',
            options=lista_codigos,
            format_func=lambda codigo: f"{ESTADO_NOMBRES.get(codigo, f'Entidad {codigo}')}"  
        )
        # Definición de nombre_estado antes de usarse
        nombre_estado = ESTADO_NOMBRES.get(entidad_seleccionada_codigo, f'Entidad {entidad_seleccionada_codigo}')

        df_filtrado = df_final[df_final['ent_resid'] == entidad_seleccionada_codigo]
        distribucion_cluster = df_filtrado['cluster'].value_counts(normalize=True).mul(100).sort_index()

        distribucion_cluster.index = distribucion_cluster.index.map(CLUSTER_NOMBRES)

        # Gráfico Plotly
        fig_bar = px.bar(
            distribucion_cluster,
            y=distribucion_cluster.values, 
            x=distribucion_cluster.index,
            labels={'y': 'Porcentaje de Casos (%)', 'x': 'Perfil de Riesgo'},
            title=f"Distribución de Perfiles en {nombre_estado}",
            color_discrete_sequence=['#CC0000'] 
        )
        
        fig_bar.update_layout(xaxis={'categoryorder':'array', 'categoryarray': list(CLUSTER_NOMBRES.values())})
        
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(f"Distribución porcentual de los 5 perfiles en {nombre_estado}.")


    elif df_final is None:
         st.error("No se pudo cargar el DataFrame principal.")
    elif mx_geojson is None:
         st.error("Error al cargar el GeoJSON.")
         
except Exception as e:
    st.error(f"Error crítico al generar la sección geográfica: {e}")
    
st.markdown("---")


# --------------------------------------------------------------------------------
# --- SECCIÓN 4: VALIDACIÓN DEL MODELO (t-SNE 3D INTERACTIVO) ---
# --------------------------------------------------------------------------------
st.header("4. Validación del Modelo (t-SNE 3D)")

if df_tsne_3d is not None and 'cluster_nombre' in df_tsne_3d.columns:
    
    # 1. Mapeo de Colores Fijo para Plotly
    # Plotly puede asignar colores aleatorios. Usaremos un mapeo fijo.
    # Los nombres de los clusters deben coincidir con las claves del diccionario.
    color_map = {
        'Adulto Joven - riesgo nocturno': '#ffc000',     # Amarillo
        'Adulto Joven - riesgo vespertino': '#00ff00', # Verde
        'Adulto Joven - riesgo no especificado': '#00ffff',     # Cian
        'Adulto Joven - sin ocupación': '#ff00ff',    # Magenta (FOCO)
        'Adulto Mayor - analfabetismo': '#0000ff'  # Azul
    }

    # --- CÓDIGO PARA GENERAR PLOTLY 3D ---
    fig_3d = px.scatter_3d(
        df_tsne_3d,
        x='Componente_1',
        y='Componente_2',
        z='Componente_3',
        color='cluster_nombre', 
        symbol='cluster_nombre',
        hover_data=['cluster_nombre'],
        title='Visualización de Clusters con t-SNE (3D)',
        # Aplicamos el mapa de colores
        color_discrete_map=color_map 
    )

    # Ajuste de tamaño
    fig_3d.update_layout(
        height=700,
        legend_title_text='Perfil de Riesgo'
    )

    # Mostrar el gráfico interactivo
    st.plotly_chart(fig_3d, use_container_width=True) 
    st.caption("Gráfico interactivo de t-SNE que valida la separación clara de los 5 perfiles.")
    
else:
    st.warning("No se pudo generar la visualización 3D. Verifica que el archivo de datos ('tsne_3d_data.csv' o '.json') exista y contenga la columna 'cluster_nombre'.")

st.markdown("---")



