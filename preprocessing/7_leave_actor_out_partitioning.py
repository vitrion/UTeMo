import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
ALLOW_ACTORS_WITH_ALL_EMOTIONS = False  # True (9 actors) or False (14 actors)

data_dir = Path(__file__).resolve().parent.parent / "data"


def extract_emotion(filename: str) -> str:
    return filename.split("_")[0]


def extract_actor(filename: str) -> str:
    return filename.split("_")[1]


def collect_actor_files(source_root: Path):
    actor_files = defaultdict(list)
    actor_emotions = defaultdict(set)
    all_emotions = set()

    path = source_root

    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta esperada: {path}")

    for file_path in path.iterdir():
        if not file_path.is_file():
            continue

        actor = extract_actor(file_path.name)
        emotion = extract_emotion(file_path.name)

        actor_files[actor].append(file_path)
        actor_emotions[actor].add(emotion)
        all_emotions.add(emotion)

    return actor_files, actor_emotions, all_emotions


def select_valid_actors(actor_files, actor_emotions):
    if ALLOW_ACTORS_WITH_ALL_EMOTIONS:
        valid_actors = [
            actor for actor in actor_files
            if len(actor_emotions[actor]) == 7
        ]
    else:
        valid_actors = list(actor_files.keys())

    return sorted(valid_actors)


def ensure_emotion_subfolders(base_path: Path, emotions):
    for emotion in sorted(emotions):
        os.makedirs(base_path / emotion, exist_ok=True)


def copy_files(files, split_root: Path):
    for src in files:
        emotion = extract_emotion(src.name)
        dst_dir = split_root / emotion
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def build_actor_partition_for_modality(source_root: Path, output_root: Path):
    actor_files, actor_emotions, all_emotions = collect_actor_files(source_root)
    valid_actors = select_valid_actors(actor_files, actor_emotions)

    if not valid_actors:
        raise RuntimeError("No hay actores válidos con la configuración actual")

    print(f"Actores válidos: {valid_actors}")
    print(f"Total: {len(valid_actors)}")

    for idx, test_actor in enumerate(valid_actors):
        actor_root = output_root / test_actor
        train_root = actor_root / "train"
        test_root = actor_root / "test"

        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)

        ensure_emotion_subfolders(train_root, all_emotions)
        ensure_emotion_subfolders(test_root, all_emotions)

        test_files = list(actor_files[test_actor])

        train_candidates = []
        for actor in valid_actors:
            if actor != test_actor:
                train_candidates.extend(actor_files[actor])

        rng = random.Random(SEED + idx)
        rng.shuffle(train_candidates)

        train_files = train_candidates

        copy_files(test_files, test_root)
        copy_files(train_files, train_root)

        assert all(
            extract_actor(f.name) == test_actor
            for f in test_files
        ), f"Error en test para {test_actor}"

        print(f"{test_actor} completado | train={len(train_files)} | test={len(test_files)}")


if __name__ == "__main__":
    modalities = {
        "audio": data_dir / "UTeMo_audio_spectrograms",
        # "image": data_dir / "UTeMo_image_viola_jones"
    }

    output_root = data_dir / "UTeMo_actor_validation"
    if output_root.exists():
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    for modality, source_root in modalities.items():
        print(f"\n===== {modality.upper()} =====")

        modality_output_root = output_root / modality
        os.makedirs(modality_output_root, exist_ok=True)

        build_actor_partition_for_modality(
            source_root,
            modality_output_root
        )

    print("\nProceso completado.")