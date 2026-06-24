import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from PIL import Image
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"


def mel_espectrogram_generator(source_file, destination_file):
    """Esta función genera el espectrograma de MEl de un archivo de audio, la imagen generada tiene una dimención 
    64 x 64 píxeles
    Args:
        source_file (String): Dirección del archivo de origen
        destination_file (String): Dirección del archivo final
    """
    Audio_File, sr = librosa.load(source_file, sr=None)
    S = librosa.feature.melspectrogram(y=Audio_File, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(frameon=False, dpi=128)
    plt.ioff()
    librosa.display.specshow(S_db, sr=sr)
    plt.savefig(destination_file + '.png',
                bbox_inches='tight', pad_inches=0, dpi=128)
    plt.close('all')
    img = Image.open(destination_file + ".png")
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    img.save(destination_file + ".png")


if __name__ == "__main__":
    source_path = data_dir / 'UTeMo_audio_nonsilent_detection'
    destination_path = data_dir / 'UTeMo_audio_spectrograms'
    os.makedirs(destination_path, exist_ok=True)
    audios = os.listdir(source_path)
    for audio in audios:
        source_file = os.path.join(source_path, audio)
        destination_file = os.path.join(destination_path, audio[:-4])
        mel_espectrogram_generator(source_file, destination_file)
