import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LightSource

script_dir = Path(__file__).resolve().parent

figures_folder_name = 'figures'
figures_dir = script_dir / figures_folder_name
os.makedirs(figures_dir, exist_ok=True)

selected_cm = 'tab20'
TEXT_HEIGHT = 18
# Shared color palette
color_map = list(eval(f"plt.cm.{selected_cm}.colors"))

def get_colors(n):
    """Return n colors sampled from the colormap."""
    if n <= len(color_map):
        return color_map[:n]
    else:
        return [color_map[i % len(color_map)] for i in range(n)]



# =========================================================
# CONFIG
# =========================================================
RESPONSES_CSV = script_dir / "SurveyResponses.csv"
REAL_EMOTIONS_CSV = script_dir / "real_emotions.csv"

OUT_FIG1 = figures_dir / "fig3_3d_emotion_counts.png"
OUT_FIG2 = figures_dir / "fig4_polygon_categorical_proportions.png"
OUT_FIG3 = figures_dir / "fig5_pie_overall_intensity_likert5.png"
OUT_FIG4 = figures_dir / "fig6_grouped_intensity_by_emotion_likert5.png"

# Same philosophy as compute_inconsistency.py:
# detect repeated emotion/intensity columns by regex
EMO_COL_RE = re.compile(
    r"^Muestra\s+(\d+)\.\s*¿Qué emoción percibiste en el video\?\s*$",
    re.IGNORECASE
)
INT_COL_RE = re.compile(
    r"^Muestra\s+(\d+)\.\s*¿Qué tan intensa fue la emoción que percibiste\?\s*$",
    re.IGNORECASE
)

# English labels for plots
EMOTION_EN = {
    "Asco": "Disgust",
    "Felicidad": "Happiness",
    "Ira": "Anger",
    "Miedo": "Fear",
    "Neutral": "Neutral",
    "Sorpresa": "Surprise",
    "Tristeza": "Sadness",
    "NInguna de las anteriores": "None",
    "Ninguna de las anteriores": "None",
}

LIKERT_EN = {
    "Nada Intensa": "Not Intense",
    "Poco Intensa": "Slightly Intense",
    "Moderadamente Intensa": "Moderately Intense",
    "Intensa": "Intense",
    "Muy Intensa": "Very Intense",
}

# Real emotions do NOT include "None of the Above"
REAL_EMOTION_ORDER_ES = [
    "Asco", "Felicidad", "Ira", "Miedo",
    "Neutral", "Sorpresa", "Tristeza"
]
REAL_EMOTION_ORDER_EN = [EMOTION_EN[e] for e in REAL_EMOTION_ORDER_ES]

# Perceived emotions DO include "None of the Above"
PERCEIVED_EMOTION_ORDER_ES = [
    "Asco", "Felicidad", "Ira", "Miedo",
    "Neutral", "Sorpresa", "Tristeza",
    "NInguna de las anteriores"
]
PERCEIVED_EMOTION_ORDER_EN = [EMOTION_EN[e] for e in PERCEIVED_EMOTION_ORDER_ES]

LIKERT_ORDER_ES = [
    "Nada Intensa",
    "Poco Intensa",
    "Moderadamente Intensa",
    "Intensa",
    "Muy Intensa",
]
LIKERT_ORDER_EN = [LIKERT_EN[x] for x in LIKERT_ORDER_ES]


# =========================================================
# HELPERS
# =========================================================
def safe_emotion_en(x):
    return EMOTION_EN.get(x, str(x))


def detect_columns(df):
    emo_cols = {}
    int_cols = {}

    for col in df.columns:
        c = col.strip()
        m1 = EMO_COL_RE.match(c)
        m2 = INT_COL_RE.match(c)

        if m1:
            emo_cols[int(m1.group(1))] = col
        if m2:
            int_cols[int(m2.group(1))] = col

    sample_ids = sorted(set(emo_cols.keys()).intersection(int_cols.keys()))
    return emo_cols, int_cols, sample_ids


def build_long_df():
    responses = pd.read_csv(RESPONSES_CSV, encoding="utf-8-sig")
    real_df = pd.read_csv(REAL_EMOTIONS_CSV, encoding="utf-8-sig")

    emo_cols, int_cols, sample_ids = detect_columns(responses)

    # Assumes row order in real_emotions.csv corresponds to Muestra 1..N
    real_map = {i + 1: real_df.iloc[i]["real_emotion"] for i in range(len(real_df))}

    rows = []
    for rater_idx in range(len(responses)):
        for sid in sample_ids:
            emo = responses.iloc[rater_idx][emo_cols[sid]]
            intensity = responses.iloc[rater_idx][int_cols[sid]]

            if pd.isna(emo) or pd.isna(intensity):
                continue

            rows.append({
                "sample_num": sid,
                "rater": f"rater_{rater_idx+1:02d}",
                "perceived_emotion_es": emo,
                "perceived_emotion_en": safe_emotion_en(emo),
                "real_emotion_es": real_map.get(sid, None),
                "real_emotion_en": safe_emotion_en(real_map.get(sid, None)),
                "intensity_es": intensity,
                "intensity_en": LIKERT_EN.get(intensity, str(intensity)),
            })

    return pd.DataFrame(rows)


# =========================================================
# 1) 3D BAR CHART
#    Count of perceived emotions vs real emotions
# =========================================================
def plot_3d_counts(long_df):
    count_df = (
        long_df.groupby(["real_emotion_en", "perceived_emotion_en"])
        .size()
        .reset_index(name="count")
    )

    matrix = pd.DataFrame(
        0.0,
        index=REAL_EMOTION_ORDER_EN,
        columns=PERCEIVED_EMOTION_ORDER_EN,
        dtype=float
    )

    for _, row in count_df.iterrows():
        r = row["real_emotion_en"]
        p = row["perceived_emotion_en"]
        if r in matrix.index and p in matrix.columns:
            matrix.loc[r, p] = row["count"]

    # Convertir a proporciones por emoción real
    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0.0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=-70) #-70

    spacing = 2.0
    xpos, ypos = np.meshgrid(
        np.arange(len(matrix.columns)) * spacing, 
        np.arange(len(matrix.index)) * spacing
        )
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = np.full_like(xpos, 0.6, dtype=float)
    dy = np.full_like(ypos, 0.6, dtype=float)
    dz = matrix.values.flatten()

    # ==========================================
    # Colores claros: uno distinto por barra
    # ==========================================
    n_bars = len(dz)
    
    # Un color por fila (real emotion)
    row_colors = get_colors(len(matrix.index))

    # Asignar a cada barra el color de su fila
    colors = [row_colors[int(y / spacing)] for y in ypos]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, alpha=0.75)

    # Etiquetas de porcentaje encima de cada barra
    
    for x, y, z in zip(xpos, ypos, dz):
        if z > 0:
            is_diagonal = int(x) == int(y)
            ax.text(
                x,
                y + 0.6,
                z + 0.008,
                f"{z*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=0.5*TEXT_HEIGHT if is_diagonal else 0.46*TEXT_HEIGHT,
                fontweight="bold" if is_diagonal else "normal"
            )

    ax.set_xticks(np.arange(len(matrix.columns)) * spacing + dx[0] / 2)
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)) * spacing + dy[0] / 2)
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Perceived Emotion", labelpad=28)
    ax.set_ylabel("Real Emotion", labelpad=18)
    # ax.set_zlabel("Proportion", labelpad=10)
    # ax.set_title("Perceived vs Real Emotion Proportions", pad=20)

    # Separar label del eje X
    ax.xaxis.set_label_coords(0.5, -0.18)

    # Eje Z en porcentaje
    ax.set_zlim(0, 1.0)
    ax.set_zticks(np.linspace(0, 1, 6))
    ax.set_zticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])

    plt.tight_layout()
    plt.subplots_adjust(bottom=-1.5)
    plt.rcParams.update({
        "font.size": TEXT_HEIGHT,          # base
        "axes.titlesize": TEXT_HEIGHT,     # títulos
        "axes.labelsize": TEXT_HEIGHT,     # labels de ejes
        "xtick.labelsize": TEXT_HEIGHT,    # ticks eje X
        "ytick.labelsize": TEXT_HEIGHT,    # ticks eje Y
        "legend.fontsize": TEXT_HEIGHT,    # leyenda
        "legend.title_fontsize": TEXT_HEIGHT
    })
    plt.savefig(OUT_FIG1, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 2) POLYGONAL (RADAR-LIKE) PROPORTION CHART
#    Overall proportion of perceived categorical judgments
#    Includes "None of the Above"
# =========================================================
def plot_polygon_categorical(long_df):
    """
    Heptagonal / radar chart of categorical agreement percentages by real emotion.

    This uses ONLY the diagonal of the real-vs-perceived matrix:
        proportion(real=e and perceived=e) / total(real=e)

    It excludes 'None of the Above' from the radar axes, because the goal
    is to show agreement percentages for the 7 real emotions only.
    """

    # Keep only the 7 real emotions
    real_categories = REAL_EMOTION_ORDER_EN

    # Build counts real vs perceived
    count_df = (
        long_df.groupby(["real_emotion_en", "perceived_emotion_en"])
        .size()
        .reset_index(name="count")
    )

    # Matrix of counts: rows = real emotion, cols = perceived emotion
    matrix = pd.DataFrame(
        0.0,
        index=real_categories,
        columns=PERCEIVED_EMOTION_ORDER_EN,
        dtype=float
    )

    for _, row in count_df.iterrows():
        r = row["real_emotion_en"]
        p = row["perceived_emotion_en"]
        if r in matrix.index and p in matrix.columns:
            matrix.loc[r, p] = row["count"]

    # Diagonal proportions:
    # correct_count(real=e, perceived=e) / total_count(real=e)
    agreement = []
    for emo in real_categories:
        total_real = matrix.loc[emo].sum()
        correct = matrix.loc[emo, emo] if emo in matrix.columns else 0.0
        proportion = correct / total_real if total_real > 0 else 0.0
        agreement.append(proportion)

    agreement = np.array(agreement)

    # Close polygon
    values = np.concatenate([agreement, [agreement[0]]])
    angles = np.linspace(0, 2 * np.pi, len(real_categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    _ = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    color = get_colors(1)[0]
    ax.plot(angles, values, linewidth=4, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(real_categories)

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # ax.set_title("Categorical Agreement Percentages by Real Emotion")

    # Optional: annotate each vertex with its percentage
    for angle, val in zip(angles[:-1], agreement):
        ax.text(angle, val + 0.04, f"{val*100:.1f}%", ha="center", va="center", fontsize=TEXT_HEIGHT)

    plt.tight_layout()
    plt.rcParams.update({
        "font.size": TEXT_HEIGHT,          # base
        "axes.titlesize": TEXT_HEIGHT,     # títulos
        "axes.labelsize": TEXT_HEIGHT,     # labels de ejes
        "xtick.labelsize": TEXT_HEIGHT,    # ticks eje X
        "ytick.labelsize": TEXT_HEIGHT,    # ticks eje Y
        "legend.fontsize": TEXT_HEIGHT,    # leyenda
        "legend.title_fontsize": TEXT_HEIGHT
    })
    plt.savefig(OUT_FIG2, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 3) PIE CHART
#    Overall intensity distribution using current Likert 1–5
# =========================================================
def plot_pie_overall_intensity(long_df):
    counts = (
        long_df["intensity_en"]
        .value_counts()
        .reindex(LIKERT_ORDER_EN)
        .fillna(0)
    )

    colors = get_colors(len(counts))

    _, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90
    )
    # ax.set_title("Overall Intensity Distribution (Likert 1–5)")

    plt.tight_layout()
    plt.rcParams.update({
        "font.size": TEXT_HEIGHT,          # base
        "axes.titlesize": TEXT_HEIGHT,     # títulos
        "axes.labelsize": TEXT_HEIGHT,     # labels de ejes
        "xtick.labelsize": TEXT_HEIGHT,    # ticks eje X
        "ytick.labelsize": TEXT_HEIGHT,    # ticks eje Y
        "legend.fontsize": TEXT_HEIGHT,    # leyenda
        "legend.title_fontsize": TEXT_HEIGHT
    })
    plt.savefig(OUT_FIG3, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 4) GROUPED BAR CHART
#    Proportion of Likert 1–5 intensity by perceived emotion
#    Includes "None of the Above"
# =========================================================
def plot_grouped_likert_by_emotion(long_df):
    ctab = pd.crosstab(
        long_df["perceived_emotion_en"],
        long_df["intensity_en"],
        normalize="index"
    )

    ctab = ctab.reindex(REAL_EMOTION_ORDER_EN).fillna(0.0)
    ctab = ctab.reindex(columns=LIKERT_ORDER_EN).fillna(0.0)

    x = np.arange(len(ctab.index))
    width = 0.15

    _, ax = plt.subplots(figsize=(16, 7))

    colors = get_colors(len(ctab.columns))

    for i, col in enumerate(ctab.columns):
        ax.bar(
            x + (i - 2) * width,
            ctab[col].values,
            width=width,
            label=col,
            color=colors[i]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ctab.index, rotation=15)
    # ax.set_ylabel("Proportion")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    # ax.set_title("Intensity Distribution by Emotion (Likert 1–5)")
    ax.legend(title="Intensity",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.rcParams.update({
        "font.size": TEXT_HEIGHT,          # base
        "axes.titlesize": TEXT_HEIGHT,     # títulos
        "axes.labelsize": TEXT_HEIGHT,     # labels de ejes
        "xtick.labelsize": TEXT_HEIGHT,    # ticks eje X
        "ytick.labelsize": TEXT_HEIGHT,    # ticks eje Y
        "legend.fontsize": TEXT_HEIGHT,    # leyenda
        "legend.title_fontsize": TEXT_HEIGHT
    })
    plt.savefig(OUT_FIG4, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    long_df = build_long_df()

    plot_3d_counts(long_df)
    plot_polygon_categorical(long_df)
    plot_pie_overall_intensity(long_df)
    plot_grouped_likert_by_emotion(long_df)

    print("Saved figures:")
    print(f" - {OUT_FIG1}")
    print(f" - {OUT_FIG2}")
    print(f" - {OUT_FIG3}")
    print(f" - {OUT_FIG4}")


if __name__ == "__main__":
    main()