import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

AUDIO_PART_SEL = '60_20_20'
# IMAGE_PART_SEL = '40_30_30'
SEED = 42

data_dir = Path(__file__).resolve().parent.parent / "data"

audio_percentages = [float(s)/100.0 for s in AUDIO_PART_SEL.split("_")]
if len(audio_percentages) != 3:
    print("Audio partitioning selection length must be equal to three.")
    exit(1)

'''image_percentages = [float(s)/100.0 for s in IMAGE_PART_SEL.split("_")]
if len(image_percentages) != 3:
    print("Image partitioning selection length must be equal to three.")
    exit(1)'''

modalities = ['audio'] # modalities = ['audio', 'image']


def stratified_emotion_split(
    source_path,
    destination_path,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1
):
    """
    Divide una base de datos de manera estratificada por emoción.
    La semilla se fijó en 42 en el main
    """
    if (train_ratio + validation_ratio + test_ratio) != 1:
        raise ValueError("Los porcentajes deben sumar 1.")

        
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)


    emotions = defaultdict(list)

    # Agrupar por emoción
    for filename in os.listdir(source_path):
        emotion = filename.split("_")[0]

        emotions[emotion].append(filename)

    # Crear carpetas
    train_path = os.path.join(destination_path, "train")
    validation_path = os.path.join(destination_path, "val")
    test_path = os.path.join(destination_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    
# Separación estratificada
    for emotion, files in emotions.items():
        random.shuffle(files)
        n = len(files)

        train_end = int(n * train_ratio)
        validation_end = train_end + int(n * validation_ratio)

        train_files = files[:train_end]
        validation_files = files[train_end:validation_end]
        test_files = files[validation_end:]

        # Crear subcarpetas por emoción
        train_emotion_path = os.path.join(train_path, emotion)
        val_emotion_path = os.path.join(validation_path, emotion)
        test_emotion_path = os.path.join(test_path, emotion)

        os.makedirs(train_emotion_path, exist_ok=True)
        os.makedirs(val_emotion_path, exist_ok=True)
        os.makedirs(test_emotion_path, exist_ok=True)

        # Copiar train
        for file in train_files:
            shutil.copy2(
                os.path.join(source_path, file),
                os.path.join(train_emotion_path, file)
            )

        # Copiar validation
        for file in validation_files:
            shutil.copy2(
                os.path.join(source_path, file),
                os.path.join(val_emotion_path, file)
            )

        # Copiar test
        for file in test_files:
            shutil.copy2(
                os.path.join(source_path, file),
                os.path.join(test_emotion_path, file)
            )


if __name__ == "__main__":
    random.seed(SEED)
    for modality in modalities:
        if modality == 'audio':
            source_path = data_dir / "UTeMo_audio_spectrograms"
            part_sel = AUDIO_PART_SEL
            percentages = audio_percentages
        # else:
        #     source_path = data_dir / "UTeMo_image_viola_jones"
        #     part_sel = IMAGE_PART_SEL
        #     percentages = image_percentages
            
        destination_path = data_dir / f"UTeMo_{modality}_{part_sel}"
        if destination_path.exists():
            shutil.rmtree(destination_path)
        os.makedirs(destination_path, exist_ok=True)
        
        stratified_emotion_split(
            source_path, destination_path, train_ratio=percentages[0], validation_ratio=percentages[1], test_ratio=percentages[2])
