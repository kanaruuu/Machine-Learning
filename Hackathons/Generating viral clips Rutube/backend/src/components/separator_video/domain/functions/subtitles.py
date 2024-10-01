import os
import tempfile
import whisper
import ffmpeg
import json
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip, ColorClip


def make_subtitled_video(video_bytes, model):
    # Создаем временный файл для входного видео
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(video_bytes)
        temp_input_filepath = temp_input_file.name

    temp_output_filepath = None  # Инициализируем переменную для временного файла вывода

    try:
        # Получаем путь к временной аудиодорожке
        audiofilename = get_audio(temp_input_filepath)

        # Транскрибируем аудио с использованием Whisper
        result = model.transcribe(audiofilename, word_timestamps=True)

        wordlevel_info = []
        for each in result['segments']:
            words = each['words']
            for word in words:
                wordlevel_info.append({'word': word['word'].strip(), 'start': word['start'], 'end': word['end']})

        # Сохраняем промежуточные данные (при необходимости)
        with open('data.json', 'w') as f:
            json.dump(wordlevel_info, f, indent=4)

        # Генерируем субтитры
        linelevel_subtitles = split_text_into_lines(wordlevel_info)
        frame_size = (1080, 1920)

        all_linelevel_splits = []
        for line in linelevel_subtitles:
            out = create_caption(line, frame_size)
            all_linelevel_splits.extend(out)

        # Открываем исходное видео для обработки
        input_video = VideoFileClip(temp_input_filepath)
        input_video_duration = input_video.duration

        background_clip = ColorClip(size=frame_size, color=(0, 0, 0)).set_duration(input_video_duration)
        final_video = input_video.resize(width=1080)
        final_video = CompositeVideoClip([background_clip, final_video.set_position(("center", "center"))])
        final_video = CompositeVideoClip([final_video] + all_linelevel_splits)
        final_video = final_video.set_audio(input_video.audio)

        # Создаем временный файл для вывода
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
            temp_output_filepath = temp_output_file.name

        # Записываем результат в временный файл
        final_video.write_videofile(temp_output_filepath, fps=24, codec="libx264", audio_codec="aac")

        # Читаем результат в байтах
        with open(temp_output_filepath, 'rb') as f:
            video_output_bytes = f.read()

        # Возвращаем итоговое видео в байтах
        return video_output_bytes

    finally:
        # Удаляем временные файлы
        if os.path.exists(temp_input_filepath):
            os.remove(temp_input_filepath)
        if os.path.exists(audiofilename):
            os.remove(audiofilename)
        if temp_output_filepath and os.path.exists(temp_output_filepath):
            os.remove(temp_output_filepath)

def video_to_bytes(filepath: str) -> bytes:
    with open(filepath, 'rb') as video_file:
        return video_file.read()

def get_audio(videofilename):
  audiofilename = videofilename.replace(".mp4",'.mp3')
  input_stream = ffmpeg.input(videofilename)
  audio = input_stream.audio
  output_stream = ffmpeg.output(audio, audiofilename)
  output_stream = ffmpeg.overwrite_output(output_stream)
  ffmpeg.run(output_stream)
  return(audiofilename)

def split_text_into_lines(data):

    MaxChars = 16
    #maxduration in seconds
    MaxDuration = 3.0
    #Split if nothing is spoken (gap) for these many seconds
    MaxGap = 1.5

    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0

    for idx,word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        line.append(word_data)
        line_duration += end - start

        temp = " ".join(item["word"] for item in line)


        # Check if adding a new word exceeds the maximum character count or duration
        new_line_chars = len(temp)

        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars
        if idx>0:
          gap = word_data['start'] - data[idx-1]['end']
          # print (word,start,end,gap)
          maxgap_exceeded = gap > MaxGap
        else:
          maxgap_exceeded = False


        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line
                }
                subtitles.append(subtitle_line)
                line = []
                line_duration = 0
                line_chars = 0


    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line
        }
        subtitles.append(subtitle_line)

    return subtitles


def create_caption(textJSON, framesize, font="Roboto-Medium.ttf", fontsize=44, color='white', bgcolor='#354D73'):
    wordcount = len(textJSON['textcontents'])
    full_duration = textJSON['end'] - textJSON['start']

    word_clips = []
    xy_textclips_positions = []

    frame_width, frame_height = framesize
    x_buffer = frame_width * 1/10  # Можно оставить или настроить по необходимости
    bottom_margin = 600  # Отступ от нижнего края в пикселях
    y_pos = frame_height - bottom_margin  # Начальная позиция по вертикали

    space_width = ""
    space_height = ""

    # Вычисляем ширину всей строки для центрирования
    total_line_width = sum([TextClip(word['word'], font=font, fontsize=fontsize, color=color).size[0] for word in textJSON['textcontents']])
    x_pos = (frame_width - total_line_width) / 2  # Центрирование по горизонтали

    for index, wordJSON in enumerate(textJSON['textcontents']):
        duration = wordJSON['end'] - wordJSON['start']
        word_clip = TextClip(wordJSON['word'], font=font, fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
        word_clip_space = TextClip(" ", font=font, fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
        word_width, word_height = word_clip.size
        space_width, space_height = word_clip_space.size

        if x_pos + word_width + space_width > frame_width - 2 * x_buffer:
            # Переход на следующую строку
            remaining_words = textJSON['textcontents'][index:]
            total_line_width = sum([TextClip(word['word'], font=font, fontsize=fontsize, color=color).size[0] for word in remaining_words])
            x_pos = (frame_width - total_line_width - word_width) / 2 # Центрирование новой строки
            y_pos += word_height + 40  # Смещение вниз для новой строки

        # Сохранение позиции каждого слова
        xy_textclips_positions.append({
            "x_pos": x_pos + x_buffer,
            "y_pos": y_pos,
            "width": word_width,
            "height": word_height,
            "word": wordJSON['word'],
            "start": wordJSON['start'],
            "end": wordJSON['end'],
            "duration": duration
        })

        # Установка позиции слова и пробела
        word_clip = word_clip.set_position((x_pos + x_buffer, y_pos))
        word_clip_space = word_clip_space.set_position((x_pos + word_width + x_buffer, y_pos))

        # Обновление позиции по горизонтали
        x_pos += word_width + space_width

        word_clips.append(word_clip)
        word_clips.append(word_clip_space)

    # Добавление выделенных слов с фоном
    for highlight_word in xy_textclips_positions:
        word_clip_highlight = TextClip(
            highlight_word['word'],
            font=font,
            fontsize=fontsize,
            color=color,
            bg_color=bgcolor
        ).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
        word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
        word_clips.append(word_clip_highlight)

    return word_clips
