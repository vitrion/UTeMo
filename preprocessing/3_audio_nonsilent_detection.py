from pydub.silence import detect_nonsilent
from pydub import AudioSegment
import os
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"


def preprocess_audio(source_file, destination_file):
    """Esta función aplica un preprocesamiento a una muestra de audio, elimina silencios inicial y final
    realiza una aplificación de 10dB y elimina sonidos menores a un umbral (ruido ambiental)
    NOTA: En esta función se cambió el método de muestreo de frecuencias por un umbreal para facilidad de ejecución
    Args:
        source_file (String): Dirección del archivo de origen
        destination_file (String): Dirección del archivo final
    """
    sound = AudioSegment.from_wav(source_file)
    # Amplifica el audio
    sound = sound.apply_gain(+10)

    # Detecta regiones con voz
    non_sil_times = detect_nonsilent(
        sound,
        min_silence_len=50,
        silence_thresh=sound.dBFS - 16
    )

    # No se detectó voz
    if len(non_sil_times) == 0:
        print(f"No speech detected: {source_file}")
        return False

    # Une segmentos cercanos
    non_sil_times_concat = [non_sil_times[0]]
    for start, end in non_sil_times[1:]:
        if start - non_sil_times_concat[-1][1] < 200:
            non_sil_times_concat[-1][1] = end
        else:
            non_sil_times_concat.append([start, end])

    # Elimina segmentos demasiado cortos
    non_sil_times = [
        segment
        for segment in non_sil_times_concat
        if segment[1] - segment[0] > 350
    ]

    # Después del filtrado ya no quedó voz
    if len(non_sil_times) == 0:
        print(f"No silence detected: {source_file}")
        sound.export(destination_file, format="wav")
        return True

    # Conserva desde el inicio del primer segmento
    # hasta el final del último
    processed_audio = sound[
        non_sil_times[0][0]:
        non_sil_times[-1][1]
    ]

    processed_audio.export(
        destination_file,
        format="wav"
    )

    return True


if __name__ == "__main__":
    source_path = data_dir / 'UTeMo_audio_denoising'
    destination_path = data_dir / 'UTeMo_audio_nonsilent_detection'
    os.makedirs(destination_path, exist_ok=True)
    audios = os.listdir(source_path)
    for audio in audios:
        source_file = os.path.join(source_path,audio)
        destination_file = os.path.join(destination_path,audio)
        preprocess_audio(source_file, destination_file)