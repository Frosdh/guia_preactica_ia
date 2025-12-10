import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Práctica IA - Predicción Académica", layout="wide", page_icon="🎓")

# Título principal
st.title("🎓 Sistema de Predicción de Rendimiento Académico")
st.markdown("### Modelos de Machine Learning: Supervisado y No Supervisado")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("📊 Navegación")
st.sidebar.markdown("Selecciona la sección que deseas explorar:")
opcion = st.sidebar.radio(
    "",
    ["🏠 Inicio", "📂 Exploración de Datos", "🧹 Preparación de Datos", "🤖 Modelo Supervisado", "🔍 Clustering", "📈 Comparación de Modelos"],
    label_visibility="collapsed"
)

# Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv('data/academic_performance_master.csv')
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'academic_performance_master.csv' en la carpeta 'data/'")
        st.info("📁 Por favor, asegúrate de que el archivo esté en: `data/academic_performance_master.csv`")
        return None

# Cargar datos
df = cargar_datos()

if df is not None:
    
    # ============= SECCIÓN: INICIO =============
    if opcion == "🏠 Inicio":
        st.header("Bienvenido al Sistema de Análisis Académico")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total de Estudiantes", len(df))
        with col2:
            st.metric("📋 Variables Analizadas", len(df.columns))
        with col3:
            if 'Nota_final' in df.columns:
                promedio = df['Nota_final'].mean()
                st.metric("📈 Promedio General", f"{promedio:.2f}")
        
        st.markdown("""
        ### 📋 Objetivo de la Práctica
        
        Desarrollar modelos de Machine Learning para:
        
        1. **🎯 Modelo Supervisado (Clasificación)**
           - Predecir si un estudiante aprobará o reprobará
           - Utilizar Regresión Logística
           - Evaluar con accuracy, matriz de confusión y métricas
        
        2. **🔍 Modelo No Supervisado (Clustering)**
           - Agrupar estudiantes según patrones de rendimiento
           - Aplicar K-means con 2-4 clusters
           - Identificar perfiles de estudiantes
        
        ### 🎯 Actividades Desarrolladas
        
        ✅ **Carga y exploración de datos**
        - Estructura, tipos de datos, estadísticas
        - Identificación de problemas de calidad
        
        ✅ **Preparación del dataset**
        - Limpieza y estandarización
        - Creación de variable objetivo (Aprobado/Reprobado)
        - Codificación de variables categóricas
        - Normalización de datos
        
        ✅ **Modelo Supervisado**
        - Entrenamiento con parámetros ajustables
        - Accuracy, matriz de confusión, reporte
        - Interpretación de resultados
        
        ✅ **Modelo No Supervisado**
        - K-means con 2-4 clusters
        - Visualización de clusters y centroides
        - Análisis de perfiles de estudiantes
        
        ### 🧭 Navegación Rápida
        
        Usa el menú lateral para explorar cada sección.
        """)
        
        st.info("💡 **Sugerencia:** Comienza por 'Exploración de Datos' para entender el dataset.")
    
    # ============= SECCIÓN: EXPLORACIÓN =============
    elif opcion == "📂 Exploración de Datos":
        st.header("1️⃣ Carga y Exploración de Datos")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Vista General", "📈 Estadísticas", "🔍 Calidad de Datos", "📉 Visualizaciones"])
        
        with tab1:
            st.subheader("Vista previa del dataset")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Información del Dataset")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📏 Dimensiones:**")
                st.write(f"- Filas: {df.shape[0]}")
                st.write(f"- Columnas: {df.shape[1]}")
            with col2:
                st.write("**📋 Tipos de datos:**")
                tipos = df.dtypes.value_counts()
                for tipo, count in tipos.items():
                    st.write(f"- {tipo}: {count} columnas")
        
        with tab2:
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe(), use_container_width=True)
            
            if 'Nota_final' in df.columns:
                st.subheader("Análisis de la Nota Final")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📉 Mínimo", f"{df['Nota_final'].min():.2f}")
                with col2:
                    st.metric("📊 Promedio", f"{df['Nota_final'].mean():.2f}")
                with col3:
                    st.metric("📍 Mediana", f"{df['Nota_final'].median():.2f}")
                with col4:
                    st.metric("📈 Máximo", f"{df['Nota_final'].max():.2f}")
        
        with tab3:
            st.subheader("Análisis de Calidad de Datos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔍 Valores Nulos:**")
                nulos = df.isnull().sum()
                if nulos.sum() > 0:
                    st.warning(f"⚠️ Se encontraron {nulos.sum()} valores nulos en total")
                    nulos_df = pd.DataFrame({'Columna': nulos[nulos > 0].index, 
                                            'Nulos': nulos[nulos > 0].values})
                    st.dataframe(nulos_df, use_container_width=True)
                else:
                    st.success("✅ No se encontraron valores nulos")
            
            with col2:
                st.write("**🔍 Valores Duplicados:**")
                duplicados = df.duplicated().sum()
                if duplicados > 0:
                    st.warning(f"⚠️ Se encontraron {duplicados} filas duplicadas")
                else:
                    st.success("✅ No se encontraron filas duplicadas")
            
            st.write("**📋 Resumen de Tipos:**")
            tipos_info = []
            for col in df.columns:
                tipos_info.append({
                    'Columna': col,
                    'Tipo': str(df[col].dtype),
                    'Valores únicos': df[col].nunique(),
                    'Nulos': df[col].isnull().sum()
                })
            st.dataframe(pd.DataFrame(tipos_info), use_container_width=True)
        
        with tab4:
            st.subheader("Distribuciones de Variables Clave")
            
            if 'Nota_final' in df.columns:
                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                
                # Histograma
                ax[0].hist(df['Nota_final'].dropna(), bins=25, color='skyblue', edgecolor='black', alpha=0.7)
                ax[0].axvline(df['Nota_final'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
                ax[0].axvline(df['Nota_final'].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
                ax[0].set_title('Distribución de Notas Finales', fontsize=14, fontweight='bold')
                ax[0].set_xlabel('Nota Final')
                ax[0].set_ylabel('Frecuencia')
                ax[0].legend()
                ax[0].grid(alpha=0.3)
                
                # Boxplot
                ax[1].boxplot(df['Nota_final'].dropna())
                ax[1].set_title('Boxplot de Notas Finales', fontsize=14, fontweight='bold')
                ax[1].set_ylabel('Nota Final')
                ax[1].grid(alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            
            # Distribución de aprobados vs reprobados
            if 'Nota_final' in df.columns:
                umbral = st.slider("Ajustar umbral de aprobación:", 10.0, 16.0, 14.0, 0.5)
                
                aprobado = (df['Nota_final'] >= umbral).sum()
                reprobado = (df['Nota_final'] < umbral).sum()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#4CAF50', '#F44336']
                    ax.pie([aprobado, reprobado], labels=['Aprobado', 'Reprobado'], 
                           autopct='%1.1f%%', colors=colors, startangle=90,
                           textprops={'fontsize': 12, 'weight': 'bold'})
                    ax.set_title(f'Distribución con umbral = {umbral}', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.metric("✅ Aprobados", aprobado, f"{aprobado/len(df)*100:.1f}%")
                    st.metric("❌ Reprobados", reprobado, f"{reprobado/len(df)*100:.1f}%")
    
    # ============= SECCIÓN: PREPARACIÓN =============
    elif opcion == "🧹 Preparación de Datos":
        st.header("2️⃣ Preparación del Dataset")
        
        st.subheader("Limpieza y Estandarización")
        
        # Mostrar columnas originales
        st.write("**Columnas disponibles:**")
        st.write(list(df.columns))
        
        # Selección de columnas
        st.subheader("Selección de Variables")
        
        columnas_disponibles = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            nota_col = st.selectbox("Selecciona la columna de Nota Final:", 
                                   ['Nota_final'] if 'Nota_final' in columnas_disponibles else columnas_disponibles)
        
        with col2:
            # Calcular un umbral sugerido basado en la mediana
            if nota_col in df.columns:
                notas_limpias = df[nota_col].dropna()
                min_nota = float(notas_limpias.min())
                max_nota = float(notas_limpias.max())
                mediana_notas = float(notas_limpias.median())
                percentil_40 = float(notas_limpias.quantile(0.4))
                
                # Usar percentil 40 como umbral sugerido (40% reprueba, 60% aprueba)
                umbral_sugerido = round(percentil_40, 1)
                
                st.info(f"📊 Rango de notas: {min_nota:.1f} - {max_nota:.1f} | Mediana: {mediana_notas:.1f}")
            else:
                min_nota, max_nota = 0.0, 20.0
                umbral_sugerido = 14.0
            
            umbral_aprobacion = st.slider("Umbral de aprobación (Nota ≥ umbral):", 
                                         min_nota, max_nota, umbral_sugerido, 0.1)
            
            # Mostrar vista previa de la distribución
            if nota_col in df.columns:
                preview_aprobados = (df[nota_col] >= umbral_aprobacion).sum()
                preview_reprobados = (df[nota_col] < umbral_aprobacion).sum()
                preview_total = len(df[nota_col].dropna())
                
                if preview_aprobados > 0 and preview_reprobados > 0:
                    st.success(f"✅ {preview_aprobados} aprobados ({preview_aprobados/preview_total*100:.1f}%) | {preview_reprobados} reprobados ({preview_reprobados/preview_total*100:.1f}%)")
                else:
                    st.error(f"⚠️ {preview_aprobados} aprobados | {preview_reprobados} reprobados - AJUSTA EL UMBRAL")
        
        # Variables predictoras
        st.subheader("Variables Predictoras")
        vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if nota_col in vars_numericas:
            vars_numericas.remove(nota_col)
        
        # Filtrar variables inútiles
        vars_numericas = [v for v in vars_numericas if v not in ['Identificacion_Estudiante', 'Cedula_docente']]
        
        if len(vars_numericas) == 0:
            st.warning("⚠️ No hay variables numéricas para predecir. Usa variables categóricas.")
            variables_x = []
        else:
            variables_x = st.multiselect(
                "Selecciona las variables numéricas para predecir (X):",
                vars_numericas,
                default=['Asistencia'] if 'Asistencia' in vars_numericas else (vars_numericas[:1] if vars_numericas else [])
            )
        
        # Análisis de distribución antes de preparar
        if nota_col in df.columns:
            st.subheader("📊 Análisis de Distribución de Notas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df[nota_col].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(umbral_aprobacion, color='red', linestyle='--', linewidth=2, label=f'Umbral = {umbral_aprobacion}')
                ax.axvline(df[nota_col].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana = {df[nota_col].median():.1f}')
                ax.set_title('Distribución de Notas', fontsize=12, fontweight='bold')
                ax.set_xlabel('Nota Final')
                ax.set_ylabel('Frecuencia')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                aprobados_preview = (df[nota_col] >= umbral_aprobacion).sum()
                reprobados_preview = (df[nota_col] < umbral_aprobacion).sum()
                total_preview = len(df[nota_col].dropna())
                
                st.metric("Total de registros", total_preview)
                st.metric("✅ Aprobados", aprobados_preview, f"{aprobados_preview/total_preview*100:.1f}%")
                st.metric("❌ Reprobados", reprobados_preview, f"{reprobados_preview/total_preview*100:.1f}%")
                
                # Advertencia si solo hay una clase
                if aprobados_preview == 0 or reprobados_preview == 0:
                    st.error("⚠️ ADVERTENCIA: Solo hay una clase con este umbral!")
                    st.info(f"💡 Sugerencia: Ajusta el umbral entre {df[nota_col].min():.1f} y {df[nota_col].max():.1f}")
                elif min(aprobados_preview, reprobados_preview) / max(aprobados_preview, reprobados_preview) < 0.1:
                    st.warning("⚠️ Clases muy desbalanceadas. Considera ajustar el umbral.")
                else:
                    st.success("✅ Distribución aceptable")
        
        # Variables categóricas
        st.subheader("Codificación de Variables Categóricas")
        vars_categoricas = df.select_dtypes(include=['object']).columns.tolist()
        
        vars_cat_selec = []
        if len(vars_categoricas) > 0:
            st.write(f"**Variables categóricas detectadas:** {vars_categoricas}")
            codificar_cats = st.checkbox("Incluir variables categóricas", value=False)
            
            if codificar_cats:
                vars_cat_selec = st.multiselect("Selecciona variables categóricas:", vars_categoricas)
        
        if st.button("🔧 Preparar Dataset", type="primary"):
            if len(variables_x) == 0 and len(vars_cat_selec) == 0:
                st.error("❌ Debes seleccionar al menos una variable predictora")
            else:
                # Crear dataset limpio
                columnas_usar = [nota_col] + variables_x + vars_cat_selec
                df_prep = df[columnas_usar].copy()
                
                # Limpiar nulos
                nulos_antes = df_prep.isnull().sum().sum()
                df_prep = df_prep.dropna()
                
                if len(df_prep) == 0:
                    st.error("❌ No quedan datos después de eliminar nulos")
                    st.stop()
                
                st.success(f"✅ Limpieza completada: Eliminados {nulos_antes} valores nulos, quedan {len(df_prep)} registros")
                
                # Codificar categóricas
                if len(vars_cat_selec) > 0:
                    le = LabelEncoder()
                    for col in vars_cat_selec:
                        if col in df_prep.columns:
                            df_prep[col] = le.fit_transform(df_prep[col].astype(str))
                    st.success(f"✅ Variables categóricas codificadas: {vars_cat_selec}")
                
                # Crear variable objetivo
                df_prep['Aprobado'] = (df_prep[nota_col] >= umbral_aprobacion).astype(int)
                
                # Verificar distribución
                distribucion = df_prep['Aprobado'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    aprobados = distribucion.get(1, 0)
                    st.metric("✅ Aprobados (1)", aprobados, f"{aprobados/len(df_prep)*100:.1f}%")
                with col2:
                    reprobados = distribucion.get(0, 0)
                    st.metric("❌ Reprobados (0)", reprobados, f"{reprobados/len(df_prep)*100:.1f}%")
                with col3:
                    if len(distribucion) >= 2:
                        balance = min(aprobados, reprobados) / max(aprobados, reprobados)
                        st.metric("⚖️ Balance", f"{balance:.2%}")
                
                # Variables finales
                variables_finales = [v for v in variables_x + vars_cat_selec if v in df_prep.columns and v != nota_col]
                
                # Guardar en session state
                st.session_state['df_preparado'] = df_prep
                st.session_state['nota_col'] = nota_col
                st.session_state['variables_x'] = variables_finales
                st.session_state['umbral'] = umbral_aprobacion
                
                if len(distribucion) < 2:
                    st.error("❌ El dataset solo tiene una clase. Ajusta el umbral de aprobación.")
                elif len(variables_finales) == 0:
                    st.error("❌ No hay variables predictoras válidas")
                else:
                    st.success("✅ Dataset preparado correctamente. Puedes continuar al modelo supervisado.")
                    
                    # Mostrar preview
                    st.subheader("Vista Previa del Dataset Preparado")
                    st.dataframe(df_prep.head(10), use_container_width=True)
    
    # ============= SECCIÓN: MODELO SUPERVISADO =============
    elif opcion == "🤖 Modelo Supervisado":
        st.header("3️⃣ Modelo de Clasificación Supervisado")
        
        if 'df_preparado' not in st.session_state:
            st.warning("⚠️ Primero debes preparar el dataset en la sección 'Preparación de Datos'")
            st.stop()
        
        df_modelo = st.session_state['df_preparado']
        variables_x = st.session_state['variables_x']
        
        if len(variables_x) == 0:
            st.error("❌ No hay variables predictoras. Regresa a 'Preparación de Datos'")
            st.stop()
        
        st.subheader("Configuración del Modelo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algoritmo = st.selectbox("Algoritmo:", 
                                    ["Regresión Logística", "Árbol de Decisión", "Random Forest"])
        
        with col2:
            test_size = st.slider("% Datos de prueba:", 10, 40, 20) / 100
        
        with col3:
            random_state = st.number_input("Semilla aleatoria:", 0, 100, 42)
        
        if st.button("🚀 Entrenar Modelo", type="primary"):
            # Preparar datos
            X = df_modelo[variables_x]
            y = df_modelo['Aprobado']
            
            # Verificar clases
            if len(y.unique()) < 2:
                st.error("❌ Solo hay una clase en los datos. Ajusta el umbral de aprobación.")
                st.stop()
            
            try:
                # División
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Escalado
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Seleccionar modelo
                if algoritmo == "Regresión Logística":
                    modelo = LogisticRegression(random_state=random_state, max_iter=1000)
                elif algoritmo == "Árbol de Decisión":
                    modelo = DecisionTreeClassifier(random_state=random_state)
                else:
                    modelo = RandomForestClassifier(random_state=random_state, n_estimators=100)
                
                # Entrenar
                with st.spinner('Entrenando modelo...'):
                    modelo.fit(X_train_scaled, y_train)
                
                # Predicciones
                y_pred = modelo.predict(X_test_scaled)
                
                # Métricas
                st.success("✅ Modelo entrenado exitosamente")
                
                st.subheader("📊 Resultados del Entrenamiento")
                
                accuracy = accuracy_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("📊 Muestras entrenamiento", len(X_train))
                with col3:
                    st.metric("📊 Muestras prueba", len(X_test))
                
                # Interpretación del accuracy
                if accuracy >= 0.90:
                    st.success("🌟 Excelente desempeño del modelo")
                elif accuracy >= 0.80:
                    st.info("✅ Buen desempeño del modelo")
                elif accuracy >= 0.70:
                    st.warning("⚠️ Desempeño aceptable, podría mejorar")
                else:
                    st.error("❌ Desempeño bajo, considera ajustar el modelo")
                
                # Matriz de confusión
                st.subheader("📈 Matriz de Confusión")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Reprobado (0)', 'Aprobado (1)'],
                           yticklabels=['Reprobado (0)', 'Aprobado (1)'],
                           cbar_kws={'label': 'Frecuencia'},
                           annot_kws={'size': 16, 'weight': 'bold'})
                ax.set_title(f'Matriz de Confusión - {algoritmo}', fontsize=14, fontweight='bold')
                ax.set_ylabel('Valor Real', fontsize=12)
                ax.set_xlabel('Predicción', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Explicación de la matriz
                with st.expander("ℹ️ ¿Cómo interpretar la matriz de confusión?"):
                    st.markdown(f"""
                    - **Verdaderos Negativos (TN):** {cm[0,0]} - Reprobados correctamente predichos
                    - **Falsos Positivos (FP):** {cm[0,1]} - Reprobados predichos como aprobados (Error Tipo I)
                    - **Falsos Negativos (FN):** {cm[1,0]} - Aprobados predichos como reprobados (Error Tipo II)
                    - **Verdaderos Positivos (TP):** {cm[1,1]} - Aprobados correctamente predichos
                    """)
                
                # Reporte de clasificación
                st.subheader("📋 Reporte de Clasificación Detallado")
                report = classification_report(y_test, y_pred, 
                                              target_names=['Reprobado', 'Aprobado'],
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), 
                           use_container_width=True)
                
                with st.expander("ℹ️ Explicación de métricas"):
                    st.markdown("""
                    - **Precision:** De todos los que el modelo predijo como aprobados, ¿cuántos realmente lo son?
                    - **Recall:** De todos los que realmente aprobaron, ¿cuántos el modelo identificó correctamente?
                    - **F1-Score:** Media armónica entre Precision y Recall
                    - **Support:** Número de muestras de cada clase
                    """)
                
                # Importancia de características
                st.subheader("📊 Importancia de Variables")
                
                if algoritmo == "Regresión Logística":
                    importancias = pd.DataFrame({
                        'Variable': variables_x,
                        'Coeficiente': modelo.coef_[0],
                        'Importancia_Abs': np.abs(modelo.coef_[0])
                    }).sort_values('Importancia_Abs', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['green' if x > 0 else 'red' for x in importancias['Coeficiente']]
                    ax.barh(importancias['Variable'], importancias['Coeficiente'], color=colors, alpha=0.7)
                    ax.set_xlabel('Coeficiente', fontsize=12)
                    ax.set_title('Importancia de Variables\n(Verde: Influencia Positiva | Rojo: Influencia Negativa)', 
                                fontsize=14, fontweight='bold')
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.write("**Interpretación:**")
                    st.dataframe(importancias[['Variable', 'Coeficiente']], use_container_width=True)
                
                # Guardar resultados
                st.session_state['modelo_supervisado'] = {
                    'accuracy': accuracy,
                    'modelo': algoritmo,
                    'tipo': 'Supervisado - Clasificación',
                    'cm': cm,
                    'report': report
                }
                
                st.success("✅ Resultados guardados. Puedes continuar al clustering o comparar modelos.")
                
            except Exception as e:
                st.error(f"❌ Error al entrenar el modelo: {str(e)}")
                st.info("💡 Verifica que tengas suficientes datos de ambas clases.")
    
    # ============= SECCIÓN: CLUSTERING =============
    elif opcion == "🔍 Clustering":
        st.header("4️⃣ Análisis de Clustering (K-means)")
        
        st.subheader("Configuración del Clustering")
        
        # Selección de variables
        vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(vars_numericas) < 2:
            st.error("❌ Se necesitan al menos 2 variables numéricas para clustering")
            st.stop()
        
        col1, col2 = st.columns(2)
        with col1:
            var_x = st.selectbox("Variable X (horizontal):", vars_numericas, index=0)
        with col2:
            var_y_options = [v for v in vars_numericas if v != var_x]
            var_y = st.selectbox("Variable Y (vertical):", var_y_options, 
                                index=0 if var_y_options else 0)
        
        n_clusters = st.slider("Número de clusters (k):", 2, 6, 3)
        
        if st.button("🔍 Realizar Clustering", type="primary"):
            # Preparar datos
            df_cluster = df[[var_x, var_y]].dropna()
            
            if len(df_cluster) < n_clusters:
                st.error(f"❌ No hay suficientes datos. Se necesitan al menos {n_clusters} registros.")
                st.stop()
            
            # Escalado
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)
            
            # K-means
            with st.spinner('Aplicando K-means...'):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
            
            df_cluster['Cluster'] = clusters
            
            st.success(f"✅ Clustering completado con {n_clusters} clusters")
            
            # Visualización principal
            st.subheader("📊 Visualización de Clusters")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(df_cluster[var_x], df_cluster[var_y], 
                               c=clusters, cmap='viridis', alpha=0.6, s=100, 
                               edgecolors='black', linewidth=0.5)
            
            # Centroides
            centroides = scaler.inverse_transform(kmeans.cluster_centers_)
            ax.scatter(centroides[:, 0], centroides[:, 1], 
                      c='red', marker='X', s=500, edgecolors='black',
                      linewidths=3, label='Centroides', zorder=5)
            
            # Etiquetar centroides
            for i, (x, y) in enumerate(centroides):
                ax.annotate(f'C{i}', (x, y), fontsize=14, fontweight='bold', 
                           color='white', ha='center', va='center')
            
            ax.set_xlabel(var_x, fontsize=12, fontweight='bold')
            ax.set_ylabel(var_y, fontsize=12, fontweight='bold')
            ax.set_title(f'Clustering K-means (k={n_clusters})', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Análisis por cluster
            st.subheader("📋 Análisis de Cada Cluster")
            
            for i in range(n_clusters):
                with st.expander(f"📊 Cluster {i} - {len(df_cluster[df_cluster['Cluster']==i])} estudiantes"):
                    cluster_data = df_cluster[df_cluster['Cluster']==i]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(f"Promedio {var_x}", f"{cluster_data[var_x].mean():.2f}")
                        st.metric(f"Desv. Est. {var_x}", f"{cluster_data[var_x].std():.2f}")
                    
                    with col2:
                        st.metric(f"Promedio {var_y}", f"{cluster_data[var_y].mean():.2f}")
                        st.metric(f"Desv. Est. {var_y}", f"{cluster_data[var_y].std():.2f}")
                    
                    st.write("**Interpretación:**")
                    if cluster_data[var_y].mean() > df_cluster[var_y].mean():
                        if cluster_data[var_x].mean() > df_cluster[var_x].mean():
                            st.success("✅ Grupo de alto rendimiento con buena asistencia")
                        else:
                            st.info("📊 Grupo de buen rendimiento pero asistencia mejorable")
                    else:
                        if cluster_data[var_x].mean() < df_cluster[var_x].mean():
                            st.error("⚠️ Grupo de bajo rendimiento y baja asistencia (Requiere atención)")
                        else:
                            st.warning("📉 Grupo con asistencia aceptable pero bajo rendimiento")
            
            # Resumen estadístico
            st.subheader("📊 Resumen Estadístico por Cluster")
            resumen = df_cluster.groupby('Cluster')[[var_x, var_y]].agg(['mean', 'std', 'min', 'max'])
            st.dataframe(resumen.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'), 
                        use_container_width=True)
            
            # Guardar resultados
            st.session_state['clustering'] = {
                'n_clusters': n_clusters,
                'var_x': var_x,
                'var_y': var_y,
                'tipo': 'No Supervisado - Clustering',
                'centroides': centroides
            }
            
            st.success("✅ Análisis de clustering completado. Puedes ir a 'Comparación de Modelos'.")
    
    # ============= SECCIÓN: COMPARACIÓN =============
    elif opcion == "📈 Comparación de Modelos":
        st.header("5️⃣ Comparación de Modelos")
        
        if 'modelo_supervisado' not in st.session_state and 'clustering' not in st.session_state:
            st.warning("⚠️ Debes ejecutar al menos un modelo para ver la comparación")
            st.stop()
        
        st.subheader("🔍 Resumen de Modelos Implementados")
        
        # Modelo Supervisado
        if 'modelo_supervisado' in st.session_state:
            st.markdown("### 🤖 Modelo Supervisado (Clasificación)")
            
            modelo_sup = st.session_state['modelo_supervisado']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tipo de Modelo", modelo_sup['modelo'])
            with col2:
                st.metric("Accuracy", f"{modelo_sup['accuracy']:.2%}")
            with col3:
                st.metric("Tipo de Aprendizaje", modelo_sup['tipo'])
            
            st.markdown("**🎯 Objetivo:** Predecir si un estudiante aprobará o reprobará")
            st.markdown(f"**📊 Desempeño:** {'Excelente' if modelo_sup['accuracy'] >= 0.9 else 'Bueno' if modelo_sup['accuracy'] >= 0.8 else 'Aceptable'}")
            
            # Métricas detalladas
            with st.expander("📋 Ver métricas detalladas"):
                report_df = pd.DataFrame(modelo_sup['report']).transpose()
                st.dataframe(report_df, use_container_width=True)
            
            st.markdown("---")
        
        # Modelo No Supervisado
        if 'clustering' in st.session_state:
            st.markdown("### 🔍 Modelo No Supervisado (Clustering)")
            
            clustering = st.session_state['clustering']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Algoritmo", "K-means")
            with col2:
                st.metric("Número de Clusters", clustering['n_clusters'])
            with col3:
                st.metric("Tipo de Aprendizaje", clustering['tipo'])
            
            st.markdown("**🎯 Objetivo:** Agrupar estudiantes según patrones de rendimiento")
            st.markdown(f"**📊 Variables:** {clustering['var_x']} vs {clustering['var_y']}")
            
            st.markdown("---")
        
        # Comparación
        st.subheader("⚖️ Comparación y Conclusiones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 Modelo Supervisado")
            st.markdown("""
            **Ventajas:**
            - ✅ Predice resultados específicos (Aprobado/Reprobado)
            - ✅ Permite medir accuracy y métricas
            - ✅ Útil para predicciones futuras
            
            **Limitaciones:**
            - ❌ Requiere datos etiquetados
            - ❌ Depende de la calidad del umbral definido
            """)
        
        with col2:
            st.markdown("#### 🔍 Modelo No Supervisado")
            st.markdown("""
            **Ventajas:**
            - ✅ Descubre patrones ocultos
            - ✅ No requiere etiquetas previas
            - ✅ Identifica grupos naturales
            
            **Limitaciones:**
            - ❌ No predice valores específicos
            - ❌ Interpretación más subjetiva
            """)
        
        st.markdown("---")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones y Conclusiones")
        
        st.markdown("""
        ### 🎯 ¿Cuál modelo es mejor?
        
        **No hay un modelo "mejor" en términos absolutos**, sino que cada uno sirve para propósitos diferentes:
        
        1. **Usa el Modelo Supervisado cuando:**
           - Necesitas predecir si un estudiante aprobará o no
           - Quieres evaluar el impacto de variables específicas
           - Tienes datos históricos con resultados conocidos
        
        2. **Usa el Modelo No Supervisado cuando:**
           - Quieres descubrir grupos naturales de estudiantes
           - Buscas identificar perfiles o patrones no obvios
           - Necesitas segmentar estudiantes para intervenciones personalizadas
        
        ### 🔄 Uso Combinado (Recomendado)
        
        La mejor estrategia es usar **ambos modelos de forma complementaria**:
        
        1. **Clustering** para identificar grupos de riesgo o perfiles
        2. **Clasificación** para predecir resultados individuales
        3. **Intervenciones personalizadas** según el cluster y la predicción
        
        ### 📊 Aplicación Práctica
        
        **Ejemplo de uso:**
        - El clustering identifica un grupo de "estudiantes en riesgo" (baja asistencia + notas bajas)
        - El modelo supervisado predice qué estudiantes específicos reprobarán
        - La institución puede implementar tutorías focalizadas en ese grupo
        
        ### ✅ Resultados Obtenidos en esta Práctica
        """)
        
        if 'modelo_supervisado' in st.session_state:
            accuracy = st.session_state['modelo_supervisado']['accuracy']
            st.success(f"✅ Modelo Supervisado: {accuracy:.1%} de accuracy - {'Excelente' if accuracy >= 0.9 else 'Bueno' if accuracy >= 0.8 else 'Aceptable'} desempeño")
        
        if 'clustering' in st.session_state:
            n_clusters = st.session_state['clustering']['n_clusters']
            st.info(f"✅ Clustering: {n_clusters} grupos identificados con patrones diferenciados")
        
        st.markdown("""
        ### 🎓 Conclusión Final
        
        Los modelos de Machine Learning son herramientas complementarias que, cuando se usan en conjunto, 
        proporcionan una visión más completa del rendimiento académico y permiten implementar estrategias 
        de intervención más efectivas y personalizadas.
        """)

else:
    st.error("❌ No se pudo cargar el dataset. Verifica que el archivo exista en la carpeta 'data/'")
    st.stop()