import numpy as np
import os
from keras_preprocessing.image import *
import random
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"

SEED = 42
AUDIO_AUG_FACTOR = 20
# IMAGE_AUG_FACTOR = 4
ALLOW_AUDIO_AUG = True
# ALLOW_IMAGE_AUG = True
AUDIO_PART_SEL = '60_20_20'
# IMAGE_PART_SEL = '40_30_30'
rotation_range = 5
zoom_range = 0.1


splits_to_augment = ['train']
datasets = ['samples', 'actor']
modalities = ['audio'] # modalities = ['audio', 'image']

def data_augmentation(source_file, destination_path, prefix, data_augmentator, aug_factor):
    """Esta función aumenta los datos en base a una imagen
    Args:
        source_file (String): Dirección del archivo de origen
        destination_file (String): Dirección de la carpeta destino
        prefix: Nombre del archivo que se esta aumentando
        data_augmentator: Generador de imagenes
    """
    image = load_img(source_file)
    image = img_to_array(image)
    image = np.expand_dims(image, 0)
    image_generator = data_augmentator.flow(
        image,
        batch_size=1,
        save_to_dir=destination_path,
        save_prefix=prefix,
        save_format="png"
    )
    for i in range(aug_factor):
        next(image_generator)


if __name__ == "__main__":
    random.seed(SEED)

    audio_data_augmentator = ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=False, # Avoid splits for spectrograms
        vertical_flip=False # Avoid splits for spectrograms
    )

    '''image_data_augmentator = ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=True,
        vertical_flip=False
    )'''

    for dataset in datasets:
        for modality in modalities:
            if modality == 'audio':
                part_sel = AUDIO_PART_SEL
                aug_factor = AUDIO_AUG_FACTOR
                allow = ALLOW_AUDIO_AUG
                data_augmentator = audio_data_augmentator
            # else:
            #     part_sel = IMAGE_PART_SEL
            #     aug_factor = IMAGE_AUG_FACTOR
            #     allow = ALLOW_IMAGE_AUG
            #     data_augmentator = image_data_augmentator

            if dataset == 'actor':
                lao_dir = data_dir / f"UTeMo_{dataset}_validation"
                if not lao_dir.exists():
                    continue
                dataset_dir = lao_dir / f"{modality}"
                unique_actors = [entry.name for entry in dataset_dir.iterdir() if entry.is_dir()]
                source_and_destination_paths = [dataset_dir / f"{actor}" for actor in unique_actors]
            else:
                source_and_destination_paths = [data_dir / f"UTeMo_{modality}_{part_sel}"]

            for path in source_and_destination_paths:
                print(f"Dataset: {dataset}, Modality: {modality}, Path: {path}")
                if allow:
                    for split in splits_to_augment:
                        split_path = path / split

                        # recorrer emociones
                        emotions = os.listdir(split_path)

                        for emotion in emotions:
                            emotion_path = split_path / emotion

                            # asegurar que sea carpeta
                            if not os.path.isdir(emotion_path):
                                continue

                            images = os.listdir(emotion_path)

                            for image in images:
                                if "aug" in image:
                                    continue

                                source_file = os.path.join(emotion_path, image)

                                # evitar procesar archivos generados previamente
                                if not image.lower().endswith((".png", ".jpg", ".jpeg")):
                                    continue

                                data_augmentation(
                                    source_file,
                                    emotion_path,
                                    f"{image[:-4]}_aug",
                                    data_augmentator,
                                    aug_factor
                                )
