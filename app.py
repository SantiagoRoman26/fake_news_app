# app.py
import streamlit as st
import joblib
import os
import time
from openai import OpenAI
from utils import limpiar_texto, safe_load_env_var

# ---------------------------
# Configuración UI
# ---------------------------
st.set_page_config(page_title="Detector Fake News (ES)", page_icon="📰", layout="centered")
st.title("📰 Detector de Fake News (es) con Resúmenes Inteligentes")
st.markdown("""
**Cómo funciona:** combinamos un modelo clásico (TF-IDF + LogisticRegression) para estimar la probabilidad 
de que una noticia sea falsa, y un modelo de lenguaje (OpenAI) para generar un resumen y observaciones.
""")

# ---------------------------
# Cargar modelo (cacheado)
# ---------------------------
@st.cache_resource
def cargar_modelo(path="model.pkl"):
    if not os.path.exists(path):
        st.error("No se encontró 'model.pkl'. Debes entrenar y generar este archivo con train_model.py")
        raise FileNotFoundError("model.pkl no encontrado")
    model = joblib.load(path)
    return model

# Intentamos cargar modelo, si no existe se mostrará error en la UI al analizar
try:
    modelo = cargar_modelo("model.pkl")
except Exception:
    modelo = None

# ---------------------------
# Input usuario
# ---------------------------
with st.form("form_analyze"):
    noticia = st.text_area("✍️ Pega aquí el texto de la noticia en español:", height=220)
    mostrar_fuente = st.checkbox("Mostrar texto original en resultados", value=False)
    submitted = st.form_submit_button("🔎 Analizar noticia")

if submitted:
    if not noticia or noticia.strip() == "":
        st.warning("Por favor pega un texto de noticia para analizar.")
    else:
        # Mostrar spinner mientras analiza
        with st.spinner("Analizando..."):
            # 1) Preprocesar y clasificar
            if modelo is None:
                st.error("Modelo no cargado. Genera 'model.pkl' corriendo train_model.py")
            else:
                noticia_clean = limpiar_texto(noticia)
                # predict_proba puede lanzar error si el pipeline no soporta
                try:
                    prob_fake = modelo.predict_proba([noticia_clean])[0][1]
                except Exception:
                    # fallback: predict único (0/1)
                    pred = modelo.predict([noticia_clean])[0]
                    prob_fake = float(pred)
                etiqueta = "❌ Posible Fake News" if prob_fake > 0.5 else "✅ Noticia Creíble"

                # Mostrar resultados ML
                st.subheader("📊 Resultado del clasificador")
                st.metric(label="Probabilidad de ser Fake", value=f"{prob_fake*100:.1f}%")
                if prob_fake > 0.5:
                    st.error(etiqueta)
                else:
                    st.success(etiqueta)

            # 2) Generar resumen con OpenAI (si hay clave)
            openai_key = safe_load_env_var("OPENAI_API_KEY")
            resumen = None
            if openai_key:
                client = OpenAI(api_key=openai_key)
                try:
                    # Llamada a OpenAI ChatCompletion (nueva sintaxis)
                    prompt = (
                        "Resume la siguiente noticia en 2-3 frases en español. "
                        "Luego, en 1 frase, indica si observas señales de manipulación (ej: falta de fuentes, lenguaje emocional, afirmaciones extraordinarias) y por qué.\n\n"
                        f"NOTICIA:\n{noticia}\n\nRESPUESTA:"
                    )
                    response = client.chat.completions.create(
                        model="gpt-5-nano",  # modelo pequeño y económico
                        messages=[
                            {"role": "system", "content": "Eres un experto en verificación de información y periodismo."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    resumen = response.choices[0].message.content.strip()
                except Exception as e:
                    resumen = f"⚠️ Error generando resumen: {e}"
            else:
                resumen = "No hay API key de OpenAI configurada. Para generar resumen activa OPENAI_API_KEY."

            # 3) Mostrar resumen
            st.subheader("📝 Resumen inteligente (LLM)")
            st.info(resumen)

            # 4) Mostrar texto original (opcional)
            if mostrar_fuente:
                st.subheader("🔎 Texto original")
                st.write(noticia)

        # Fin spinner
        st.success("Análisis completo ✅")
        # Pequeña pausa visual para UX
        time.sleep(0.3)
