import re
import subprocess
from datetime import timedelta
import logging
import os
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def clean_time(time_str):

    match = re.match(r"(\d+:\d+:\d+\.\d{3})", time_str)
    if match:
        return match.group(1)
    else:

        parts = time_str.split(".")
        if len(parts) >= 2:

            return f"{parts[0]}.{parts[1][:3]}"
        else:
            raise ValueError(f"Не удалось разобрать время: {time_str}")


def get_duration(start, end):
    def parse_time(time_str):
        try:
            parts = time_str.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds, milliseconds = parts[2].split(".")
            seconds = int(seconds)
            milliseconds = int(milliseconds.ljust(3, "0")[:3])
            return timedelta(
                hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
            )
        except Exception as e:
            raise ValueError(f"Не удалось разобрать время: {time_str}") from e

    start_td = parse_time(start)
    end_td = parse_time(end)
    delta = end_td - start_td
    total_seconds = delta.total_seconds()
    if total_seconds < 0:
        raise ValueError(f"Конец сцены раньше начала: начало {start}, конец {end}")
    return total_seconds


def get_video_scenes_bytes(return_result: list[dict], temp_input_filepath: str) -> list:
    scenes = []
    for line in return_result:
        try:
            start_time = clean_time(line["start_time"])
            end_time = clean_time(line["end_time"])
            scenes.append({"number": line["idx"], "start": start_time, "end": end_time})
        except ValueError as ve:
            print(f"Ошибка обработки строки: {line.strip()}")
            print(ve)

    video_bytes_list = []
    for scene in scenes:
        number = scene["number"]
        start = scene["start"]
        end = scene["end"]
        try:
            duration = get_duration(start, end)
        except ValueError as ve:
            logging.error(f"Ошибка вычисления длительности для сцены {number}: {ve}")
            continue

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_output_file:
            temp_output_filepath = temp_output_file.name

        cmd = [
            "ffmpeg",
            "-ss",
            start,
            "-t",
            f"{duration:.3f}",
            "-i",
            temp_input_filepath,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-f",
            "mp4",
            "-y",
            temp_output_filepath,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Ошибка при обработке сцены {number}: {result.stderr}")
                continue
            with open(temp_output_filepath, "rb") as f:
                video_bytes = f.read()
            video_bytes_list.append((number, video_bytes))
            logging.info(f"Сцена {number} успешно преобразована в байты!")
        except subprocess.CalledProcessError as e:
            logging.error(f"Ошибка при обработке сцены {number}: {e}")
        finally:
            os.remove(temp_output_filepath)

    logging.info(f"Все сцены успешно обработаны!")

    return video_bytes_list