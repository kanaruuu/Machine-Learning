import cv2
import numpy as np
from tqdm import tqdm
import librosa
from datetime import timedelta
import os
from skimage.metrics import structural_similarity as compare_ssim
from ultralytics import YOLO
import torch
import tempfile
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def format_time(seconds) -> str:
    """
    Форматирует время из секунд в строку вида ЧЧ:ММ:СС.МММ
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int((seconds - total_seconds) * 1000)
    return f"{str(td)}.{milliseconds:03d}"


def analyze_audio(video_bytes_path: str, fps: int):
    """
    Анализирует моменты начала и тишины в аудио из видео байтов.

    Args:
        video_bytes (bytes): Видео данные в байтах.
        fps (float): Количество кадров в секунду видео.

    Returns:
        Tuple[set, set]: Наборы кадров с началом звука и тишиной.
    """

    audio_path = 'temp_audio.wav'
    command = f"ffmpeg -i \"{video_bytes_path}\" -q:a 0 -map a \"{audio_path}\" -y -loglevel quiet"
    os.system(command)

    y, sr = librosa.load(audio_path, sr=None)

    if y is None or sr is None:
        return set(), set()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=False, delta=0.5, wait=1
    )
    onset_times = librosa.frames_to_time(peaks, sr=sr)
    audio_onset_frames = (onset_times * fps).astype(int)

    frame_length = 2048
    hop_length = 512

    energy = np.array(
        [sum(abs(y[i: i + frame_length] ** 2)) for i in range(0, len(y), hop_length)]
    )

    energy = energy / np.max(energy)

    silence_threshold = 0.1
    silence_frames = np.where(energy < silence_threshold)[0]
    silence_times = librosa.frames_to_time(silence_frames, sr=sr, hop_length=hop_length)
    audio_silence_frames = (silence_times * fps).astype(int)

    return set(audio_onset_frames), set(audio_silence_frames)


def video_frames_generator(video_bytes_path: str, frame_step: int):
    """
    Генератор для последовательного чтения кадров из видео байтов.

    Args:
        video_bytes (bytes): Видео данные в байтах.
        frame_step (int): Шаг между обрабатываемыми кадрами.

    Yields:
        Tuple[int, np.ndarray]: Индекс кадра и сам кадр.
    """

    cap = cv2.VideoCapture(video_bytes_path)

    if not cap.isOpened():
        logging.error("Ошибка открытия видео из байтов")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            yield frame_idx, frame
        frame_idx += 1

    cap.release()


def detect_scenes(
        video_bytes: bytes,
        ssim_threshold: float = 0.8,
        min_scene_length_sec: float = 5.0,
        batch_size: int = 16,
        frame_step: int = 1,
) -> list:
    """
    Обнаруживает смены сцен в видео байтах на основе визуальных и аудио признаков.

    Args:
        video_bytes (bytes): Видео данные в байтах.
        ssim_threshold (float): Порог SSIM для определения значительных изменений.
        min_scene_length_sec (float): Минимальная длина сцены в секундах.
        batch_size (int): Размер батча для обработки кадров.
        frame_step (int): Шаг между обрабатываемыми кадрами.

    Returns:
        List[dict]: Список сцен с информацией о кадрах и времени начала и конца.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(video_bytes)
        temp_input_filepath = temp_input_file.name

    cap = cv2.VideoCapture(temp_input_filepath)

    if not cap.isOpened():
        logging.error("Ошибка открытия видео из байтов")
        return []


    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Используемое устройство: {device}")

    model = YOLO("yolov8s.pt").to(device)
    model.overrides["verbose"] = False

    logging.info("Анализ аудио...")
    audio_onset_frames, audio_silence_frames = analyze_audio(video_bytes_path=temp_input_filepath, fps=fps)

    scenes = []
    scene_changes = []
    prev_frame_gray = None
    last_scene_change_frame = 0
    prev_objects = []

    logging.info("Анализ видео...")
    batch_frames = []
    batch_frame_indices = []

    total_steps = (frame_count + frame_step - 1) // frame_step
    frame_generator = video_frames_generator(temp_input_filepath, frame_step)

    for _ in tqdm(range(total_steps), desc="Обработка видео", total=total_steps):
        try:
            frame_idx, frame = next(frame_generator)
        except StopIteration:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(frame_rgb)
        batch_frame_indices.append(frame_idx)

        if len(batch_frames) == batch_size or frame_idx >= frame_count - frame_step:
            results = model(batch_frames, imgsz=640)

            for i, result in enumerate(results):
                current_objects = []
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    if conf > 0.5:
                        class_name = model.names[cls]
                        current_objects.append(class_name)

                current_objects = list(set(current_objects))
                frame_idx_current = batch_frame_indices[i]
                frame_gray_current = cv2.cvtColor(batch_frames[i], cv2.COLOR_RGB2GRAY)

                significant_visual_change = False
                significant_audio_change = False
                significant_object_change = False

                if prev_frame_gray is not None:
                    ssim_value = compare_ssim(prev_frame_gray, frame_gray_current)

                    if ssim_value < ssim_threshold:
                        significant_visual_change = True

                    if set(current_objects) != set(prev_objects):
                        significant_object_change = True

                    time_in_seconds = frame_idx_current / fps
                    window_frames = int(fps * 1.0)
                    audio_window = range(
                        max(0, frame_idx_current - window_frames),
                        min(frame_count, frame_idx_current + window_frames),
                    )

                    if any(frame in audio_onset_frames for frame in audio_window) or \
                            any(frame in audio_silence_frames for frame in audio_window):
                        significant_audio_change = True

                    if (significant_visual_change and significant_audio_change) or significant_object_change:
                        if (frame_idx_current - last_scene_change_frame) / fps >= min_scene_length_sec:
                            scene_changes.append(frame_idx_current)
                            last_scene_change_frame = frame_idx_current
                            time_str = format_time(time_in_seconds)
                            logging.info(
                                f"\nОбнаружена смена шота на кадре {frame_idx_current} (время {time_str})"
                            )
                        else:
                            if scene_changes:
                                scene_changes[-1] = frame_idx_current
                else:
                    scene_changes.append(0)
                    last_scene_change_frame = 0

                prev_frame_gray = frame_gray_current
                prev_objects = current_objects

            batch_frames = []
            batch_frame_indices = []

    if frame_count - 1 not in scene_changes:
        scene_changes.append(frame_count - 1)

    scene_changes = sorted(set(scene_changes))

    logging.info("\nФормирование окончательного списка шотов:")
    for i in range(len(scene_changes) - 1):
        start_frame = scene_changes[i]
        end_frame = scene_changes[i + 1]
        scene_length = (end_frame - start_frame) / fps
        if scene_length >= min_scene_length_sec:
            scenes.append((start_frame, end_frame))
            idx = len(scenes)
            start_time = format_time(start_frame / fps)
            end_time = format_time(end_frame / fps)
            logging.info(f"Шот {idx}: начало {start_time}, конец {end_time}")
        else:
            if scenes:
                previous_scene = scenes[-1]
                scenes[-1] = (previous_scene[0], end_frame)
                logging.info(
                    f"Объединение короткой сцены с предыдущей: сцена {len(scenes)} до {format_time(end_frame / fps)}"
                )
            else:
                logging.warning(
                    f"Шот от {format_time(start_frame / fps)} до {format_time(end_frame / fps)} слишком короткий и будет пропущен."
                )

    return_list = []
    for idx, (start_frame, end_frame) in enumerate(scenes, 1):
        return_list.append(
            {
                "idx": idx,
                "start_time": format_time(start_frame / fps),
                "end_time": format_time(end_frame / fps),
            }
        )

    logging.info(f"\nШоты сохранены")

    return return_list, temp_input_filepath


def process_video(
        video_bytes: bytes,
        ssim_threshold: float = 0.9,
        min_scene_length_sec: float = 0.4,
        batch_size: int = 32,
        frame_step: int = 8,
) -> list:
    """
    Основная функция для обработки видео байтов и обнаружения сцен.

    Args:
        video_bytes (bytes): Видео данные в байтах.
        ssim_threshold (float): Порог SSIM для определения значительных изменений.
        min_scene_length_sec (float): Минимальная длина сцены в секундах.
        batch_size (int): Размер батча для обработки кадров.
        frame_step (int): Шаг между обрабатываемыми кадрами.

    Returns:
        List[dict]: Список обнаруженных сцен с информацией о кадрах и времени начала и конца.
    """
    return detect_scenes(
        video_bytes=video_bytes,
        ssim_threshold=ssim_threshold,
        min_scene_length_sec=min_scene_length_sec,
        batch_size=batch_size,
        frame_step=frame_step,
    )
