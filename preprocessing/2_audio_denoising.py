import librosa
import soundfile as sf
import noisereduce as nr
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = Path(__file__).resolve().parent.parent / "data"
noise_file = script_dir / "noise.wav"

def noise_reducer(source_file, destination_file, noise_file):
    """
    Reduce el ruido de un archivo de audio utilizando un perfil
    de ruido obtenido de otro archivo.
    Args:
        source_file (str): Ruta del archivo de audio a procesar.
        noise_file (str): Ruta del archivo que contiene únicamente ruido.
        destination_file (str): Ruta donde se guardará el audio filtrado.
    """
    # Lee el audio a procesar
    audio, sample_rate = librosa.load(
        source_file,
        sr=None
    )

    # Lee el perfil de ruido utilizando la misma frecuencia de muestreo
    noise, _ = librosa.load(
        noise_file,
        sr=sample_rate
    )

    # Aplica reducción de ruido
    reduced_audio = nr.reduce_noise(
        y=audio,
        y_noise=noise,
        sr=sample_rate
    )

    # Guarda el resultado
    sf.write(
        destination_file,
        reduced_audio,
        sample_rate
    )



if __name__ == "__main__":
    source_path = data_dir / 'UTeMo_audio'
    destination_path = data_dir / 'UTeMo_audio_denoising'
    os.makedirs(destination_path, exist_ok=True)
    audios = os.listdir(source_path)
    for audio in audios:
        source_file = os.path.join(source_path,audio)
        destination_file = os.path.join(destination_path,audio)
        noise_reducer(source_file, destination_file, noise_file)