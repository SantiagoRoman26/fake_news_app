# fetch_kaggle.py
"""
Script para descargar un dataset de Kaggle (ejemplo).
Editar la variable `kaggle_dataset` con el identificador del dataset que quieras.
Ejecutar: python fetch_kaggle.py --dataset owner/dataset-name --out dataset.csv
"""

import argparse
import subprocess
import os
import sys

def descargar(dataset: str, out: str):
    # requiere kaggle CLI instalado y configurado
    try:
        print(f"Descargando dataset {dataset} desde Kaggle...")
        # el comando crea una carpeta con los archivos del dataset
        subprocess.check_call(["kaggle", "datasets", "download", "-d", dataset, "-f", "", "-p", ".", "--unzip"])
        # si el dataset trae un csv con nombre conocido, puedes renombrar/moverlo:
        # Para simplicidad, intenta encontrar un csv en el directorio actual
        archivos_csv = [f for f in os.listdir(".") if f.endswith(".csv")]
        if not archivos_csv:
            print("No se encontraron archivos CSV después de descargar. Revisa manualmente.")
            return
        # usar el primero
        csv = archivos_csv[0]
        print(f"Archivo encontrado: {csv}")
        if csv != out:
            os.replace(csv, out)
        print(f"Archivo final: {out}")
    except subprocess.CalledProcessError as e:
        print("Error al ejecutar kaggle CLI. Asegúrate de tener 'kaggle' instalado y configurado.")
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Identificador de dataset de Kaggle owner/dataset-name")
    parser.add_argument("--out", type=str, default="dataset.csv", help="Nombre del CSV de salida")
    args = parser.parse_args()
    descargar(args.dataset, args.out)
