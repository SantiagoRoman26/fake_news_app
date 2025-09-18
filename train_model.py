# train_model.py
"""
Script para entrenar un clasificador ligero de Fake News en español.
Entrada: dataset.csv con columnas: 'class' y 'Text'
Salida: model.pkl (pipeline TF-IDF + LogisticRegression)
"""

import pandas as pd
import joblib
import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from utils import limpiar_texto

def cargar_y_preprocesar(path_csv):
    df = pd.read_csv(path_csv)

    # Verificar columnas
    if "Text" not in df.columns or "class" not in df.columns:
        raise ValueError("El dataset debe tener columnas 'class' y 'Text'.")

    # Limpiar y normalizar los valores de la columna 'class'
    df["class"] = df["class"].astype(str).str.strip().str.upper()
    
    # Mapear labels: TRUE/REAL/VERDADERO → 0 (real), FALSE/FAKE/FALSO → 1 (fake)
    # Mapeo más flexible para diferentes formatos
    true_values = ["TRUE", "REAL", "VERDADERO", "1", "V", "T", "YES", "SI"]
    false_values = ["FALSE", "FAKE", "FALSO", "0", "F", "NO"]
    
    def map_label(value):
        if value in true_values:
            return 0
        elif value in false_values:
            return 1
        else:
            # Para debugging: mostrar valores problemáticos
            print(f"Valor problemático en columna 'class': '{value}'")
            return None

    df["label"] = df["class"].apply(map_label)
    
    # Verificar si hay valores no mapeados
    if df["label"].isnull().any():
        valores_unicos = df["class"].unique()
        raise ValueError(f"La columna 'class' contiene valores no reconocidos: {valores_unicos}. "
                        f"Valores esperados: {true_values + false_values}")

    # Limpiar textos
    df["text_clean"] = df["Text"].astype(str).apply(limpiar_texto)

    return df

def entrenar(path_csv, salida_modelo="model.pkl", test_size=0.2, random_state=42):
    print("📥 Cargando dataset...")
    df = cargar_y_preprocesar(path_csv)
    X = df["text_clean"]
    y = df["label"].astype(int)

    # Manejo de datasets pequeños
    if len(df) < 20:
        stratify = None
        test_size = 0.5
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"🏋️ Entrenando con {len(X_train)} ejemplos, validando con {len(X_test)} ejemplos...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    print("📊 Evaluación en test set...")
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds, digits=4))
    try:
        auc = roc_auc_score(y_test, probs)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass

    print(f"💾 Guardando modelo en {salida_modelo} ...")
    joblib.dump(pipeline, salida_modelo)
    print("✅ Modelo guardado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.csv", help="Ruta al CSV con dataset Kaggle")
    parser.add_argument("--out", type=str, default="model.pkl", help="Ruta de salida del modelo")
    args = parser.parse_args()
    entrenar(args.data, args.out)
