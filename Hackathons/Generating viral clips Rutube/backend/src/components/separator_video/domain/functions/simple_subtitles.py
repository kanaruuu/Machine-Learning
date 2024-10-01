import whisper
from moviepy.editor import VideoFileClip
import tempfile
import os

model = whisper.load_model("medium")

# Функция для извлечения аудио из видео в байтах
def extract_audio_from_video_bytes(video_bytes):
    # Создаем временный файл для видео
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
        temp_video_file.write(video_bytes)  # Записываем байты видео во временный файл
        temp_video_file.flush()  # Сбрасываем буфер на диск

        # Открываем видеофайл через moviepy и извлекаем аудио
        video = VideoFileClip(temp_video_file.name)

        # Создаем временный файл для аудио
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio_path = temp_audio_file.name
            video.audio.write_audiofile(audio_path)  # Извлекаем аудиодорожку

    # Удаляем временный видеофайл, так как он больше не нужен
    os.remove(temp_video_file.name)

    return audio_path


# Функция для транскрипции аудио с помощью Whisper
def transcribe_audio_with_whisper(audio_path, model):
    result = model.transcribe(audio_path)

    return result['text']


# Основная функция для извлечения звука из видео в байтах и распознавания текста
def transcribe_video_bytes(video_bytes, model):
    # Извлекаем аудиодорожку во временный файл
    audio_path = extract_audio_from_video_bytes(video_bytes)

    try:
        # Выполняем транскрипцию аудио и получаем текст
        transcribed_text = transcribe_audio_with_whisper(audio_path, model)
        return transcribed_text
    finally:
        # Удаляем временный аудиофайл после завершения работы
        if os.path.exists(audio_path):
            os.remove(audio_path)