import re
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.inter_rater import fleiss_kappa

script_dir = Path(__file__).resolve().parent

INPUT_CSV = script_dir / "SurveyResponses.csv"
OUTPUT_CSV = script_dir / "samples_by_rater.csv"

# Captura columnas del tipo: "Muestra 12. ¿Qué emoción percibiste en el video?"
EMO_COL_RE = re.compile(r"^Muestra\s+(\d+)\.\s*¿Qué emoción percibiste en el video\?\s*$")

def main():
    # 1. GENERA LA TABLA DE SAMPLES por RATERS
    # 1.1. Leer CSV (M raters x N samples extraído desde el CSV descargado de GoogleForms)
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # 1.2. Detectar y ordenar columnas de emoción por número de samples
    emo_cols = []
    for col in df.columns:
        match = EMO_COL_RE.match(col.strip())
        if match:
            sample_idx = int(match.group(1))
            emo_cols.append((sample_idx, col))
    if not emo_cols:
        raise ValueError(
            "No encontré columnas con el patrón 'Muestra N. ¿Qué emoción percibiste en el video?'. "
            "Revisa el encabezado de tu archivo CSV."
        )
    emo_cols.sort(key=lambda x: x[0])

    # 1.3. Crear IDs de rater (1..M columnas)
    rater_ids = [f"rater_{i+1:02d}" for i in range(len(df))]

    # 1.4. Construir matriz: filas=samples, columnas=raters
    #      Para cada sample, tomamos la columna correspondiente y la usamos como fila
    out_rows = []
    sample_labels = []
    for sample_idx, col in emo_cols:
        out_rows.append(df[col].tolist())
        sample_labels.append(f"sample_{sample_idx}")

    out = pd.DataFrame(out_rows, index=sample_labels, columns=rater_ids)
    out.index.name = "sample"

    # 1.5. Guardar CSV de samples por raters
    out.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"OK -> {OUTPUT_CSV} generado con {out.shape[0]} samples (filas) y {out.shape[1]} raters (columnas).")

    # 2. CALCULAR EL FACTOR DE KAPPA DE FLEISS
    # 2.1. Cargar archivo de samples por raters
    df = pd.read_csv(script_dir / "samples_by_rater.csv")

    # 2.2. Quitar columna de identificador
    ratings = df.drop(columns=["sample"])

    # 2.3. Obtener categorías únicas (8 emociones)
    categories = sorted(ratings.stack().unique())

    # 2.4. Construir matriz de conteos (N samples × 8 emociones)
    count_matrix = []
    for _, row in ratings.iterrows():
        counts = {cat: 0 for cat in categories}
        for val in row:
            counts[val] += 1
        count_matrix.append([counts[cat] for cat in categories])
    count_matrix = np.array(count_matrix)
    
    # 2.5. Cálculo de Kappa de Fleiss global
    kappa_global = fleiss_kappa(count_matrix)

    # 2.6. Cálculo de Kappa de Fleiss por clase (one-vs-all)
    kappa_per_class = {}
    for cat in categories:
        binary_matrix = []
        for row in ratings.values:
            pos = sum(val == cat for val in row)
            neg = len(row) - pos
            binary_matrix.append([neg, pos])
        binary_matrix = np.array(binary_matrix)
        kappa_per_class[cat] = fleiss_kappa(binary_matrix)
    kappa_df = pd.DataFrame(
        list(kappa_per_class.items()),
        columns=["emoción", "kappa"]
    )

    # 2.7. Imprimir resultados
    print("Kappa de Fleiss global:", kappa_global)
    print("Kappa de Fleiss por emoción:")
    print(kappa_df)


if __name__ == "__main__":
    main()