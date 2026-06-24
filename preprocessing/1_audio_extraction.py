import os
from moviepy import VideoFileClip
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"


def extract_audio(source_file, destination_file):
    """Función para extraer audio de una muestra de video
    Args:
        pathV (String): Dirección del archivo de origen
        pathF (String): Dirección del archivo final
    """
    video = VideoFileClip(source_file)
    audio = video.audio
    audio.write_audiofile(destination_file + '.wav')


if __name__ == "__main__":
    source_path = data_dir / 'UTeMo_video'
    destination_path = data_dir / 'UTeMo_audio'
    os.makedirs(destination_path, exist_ok=True)
    videos = os.listdir(source_path)
    for video in videos:
        source_file = os.path.join(source_path, video)
        destination_file = os.path.join(destination_path, video[:-4])
        extract_audio(source_file, destination_file)
