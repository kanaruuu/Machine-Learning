import os
import subprocess
import tempfile
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

MIN_DURATION = 10


def get_video_duration_from_bytes(video_bytes: bytes):
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_file_path = tmp_file.name

        result = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of',
                'default=noprint_wrappers=1:nokey=1', tmp_file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        duration = float(result.stdout)
    except Exception as e:
        logging.error(f"Ошибка при получении длительности видео: {e}")
        duration = 0
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    return duration


def filter_scenes(scenes: bytes):
    """
    Фильтрует список видео байтов, оставляя только те, которые соответствуют критериям.

    :param scenes: Список байтовых объектов видео
    :return: Отфильтрованный список байтовых объектов видео
    """
    filtered_scenes = []
    for index, video_bytes in enumerate(scenes, start=1):
        duration = get_video_duration_from_bytes(video_bytes)
        if duration >= MIN_DURATION:
            logging.info(f"Сцена {index} проходит проверку и сохраняется.")
            filtered_scenes.append(video_bytes)
        else:
            logging.info(f"Сцена {index} короче {MIN_DURATION} секунд и удаляется.")

    return filtered_scenes


