import os
import shutil
from pathlib import Path

AUDIO_PART_SEL = '60_20_20'
# IMAGE_PART_SEL = '40_30_30'

data_dir = Path(__file__).resolve().parent.parent / "data"

modalities = ['audio'] # modalities = ['audio', 'image']

def merge_train_val(base_path):
    base_path = Path(base_path)

    train_path = base_path / "train"
    val_path = base_path / "val"
    output_path = base_path / "train+val"

    # Crear carpeta destino
    os.makedirs(output_path, exist_ok=True)

    # Obtener emociones (asumiendo mismas en train y val)
    emotions = os.listdir(train_path)

    for emotion in emotions:
        train_emotion_path = train_path / emotion
        val_emotion_path = val_path / emotion
        output_emotion_path = output_path / emotion

        if not os.path.isdir(train_emotion_path):
            continue

        # Crear carpeta de emoción
        os.makedirs(output_emotion_path, exist_ok=True)

        # Copiar desde train
        for file in os.listdir(train_emotion_path):
            src = train_emotion_path / file
            dst = output_emotion_path / file

            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # Copiar desde val
        if os.path.exists(val_emotion_path):
            for file in os.listdir(val_emotion_path):
                src = val_emotion_path / file

                # Evitar sobreescribir si existe
                dst = output_emotion_path / f"val_{file}"

                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        print(f"{emotion}: OK")


if __name__ == "__main__":
    for modality in modalities:
        if modality == 'audio':
            part_sel = AUDIO_PART_SEL
        # else:
        #     part_sel = IMAGE_PART_SEL
        print(f"\n===== {modality.upper()} =====")
        base_dir = data_dir / f"UTeMo_{modality}_{part_sel}"
        merge_train_val(base_dir)