# utils.py
import re
import os
import nltk

# Intentamos descargar stopwords solo si es necesario
try:
    from nltk.corpus import stopwords
    stopwords.words("spanish")
except Exception:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

STOPWORDS_ES = set(stopwords.words("spanish"))

def limpiar_texto(text: str) -> str:
    """
    Limpieza básica para texto en español:
    - Minusculas
    - Quitar URLs
    - Mantener letras (incluye tildes y ñ) y espacios
    - Quitar stopwords
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # mantener letras y tildes (áéíóúüñ)
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    tokens = [w for w in text.split() if w and w not in STOPWORDS_ES]
    return " ".join(tokens)

def safe_load_env_var(key: str, default: str = None) -> str:
    """Obtiene variable de entorno o valor por defecto."""
    return os.getenv(key, default)
