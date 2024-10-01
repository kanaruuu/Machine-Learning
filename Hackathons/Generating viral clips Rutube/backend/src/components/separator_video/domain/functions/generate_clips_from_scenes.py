import re
import tempfile
from moviepy.editor import VideoFileClip
import logging

PAUSE_THRESHOLD = 1.5
MAX_CLIP_DURATION = 60
STANDARD_BEFORE = 2
STANDARD_AFTER = 2
EXTENDED_BEFORE_AFTER = 4


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def find_subtitle_index(dialogue, subtitles):
    cleaned_dialogue = re.sub(r"[^а-яА-Я\s]", "", dialogue.lower()).strip()
    for idx, subtitle in enumerate(subtitles):
        cleaned_subtitle = re.sub(r"[^а-яА-Я\s]", "", subtitle.lower()).strip()
        if cleaned_dialogue == cleaned_subtitle:
            return idx
    return None


def clips_info_generate(top_dialogues_list: list, subtitles: str) -> list:
    clips_info = []

    dialogue_indices = []

    for dialogue in top_dialogues_list:
        index = find_subtitle_index(dialogue, subtitles)
        if index is not None:
            dialogue_indices.append(index)
        else:
            print(f"Диалог '{dialogue}' не найден в субтитрах.")

        for idx, subtitle_idx in enumerate(dialogue_indices, 1):
            is_first = subtitle_idx == 0
            is_last = subtitle_idx == len(subtitles) - 1

            if is_first:
                additional_before = 0
                additional_after = EXTENDED_BEFORE_AFTER
            elif is_last:
                additional_before = EXTENDED_BEFORE_AFTER
                additional_after = 0
            else:
                additional_before = STANDARD_BEFORE
                additional_after = STANDARD_AFTER

            start_idx = max(subtitle_idx - additional_before, 0)
            end_idx = min(subtitle_idx + additional_after, len(subtitles) - 1)

            context_subtitles = subtitles[start_idx : end_idx + 1]
            clip_start_time = context_subtitles[0].start.to_time()
            clip_end_time = context_subtitles[-1].end.to_time()

            start_seconds = time_to_seconds(clip_start_time)
            end_seconds = time_to_seconds(clip_end_time)

            if (end_seconds - start_seconds) > MAX_CLIP_DURATION:
                end_seconds = start_seconds + MAX_CLIP_DURATION

            clips_info.append(
                {
                    "index": idx,
                    "dialogue": subtitles[subtitle_idx].text,
                    "start": start_seconds,
                    "end": end_seconds,
                }
            )

    return clips_info


def save_clips_without_subs(clip_info, video_bytes: bytes):
    index = clip_info["index"]
    start = clip_info["start"]
    end = clip_info["end"]

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_video_file:
            temp_video_file.write(video_bytes)
            temp_video_file.close()
            video = VideoFileClip(temp_video_file.name).subclip(start, end)

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp_clip_file:
                video.write_videofile(
                    temp_clip_file.name,
                    codec="libx264",
                    audio_codec="aac",
                    fps=24,
                    verbose=False,
                    logger=None,
                )
                temp_clip_file.seek(0)

                with open(temp_clip_file.name, "rb") as f:
                    clip_bytes = f.read()

                logging.info(f"Сохранён клип {index} без субтитров")
                return clip_bytes, clip_info["dialogue"]
    except Exception as e:
        logging.error(f"Ошибка при сохранении клипа {index}: {e}")
        return None


def run_process(video_bytes: bytes, top_dialogues_list: list, subtitles: str) -> list:
    clips_info = clips_info_generate(top_dialogues_list, subtitles)
    clips = []
    for clip_info in clips_info:
        clip_bytes, clip_dialogue = save_clips_without_subs(clip_info, video_bytes)
        if clip_bytes:
            clips.append((clip_bytes, clip_dialogue, ))

    return clips

