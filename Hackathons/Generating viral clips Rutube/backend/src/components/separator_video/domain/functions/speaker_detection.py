import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import subprocess
import tempfile
import os


def speaker_detect(input_video_bytes: bytes) -> bytes:
    '''
    Возвращает видео с выделенным спикером.

    :param input_video_bytes: Байты видео.
    :return: Байты видео с выделенным спикером.
    '''
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
        temp_input_file.write(input_video_bytes)
        temp_input_filepath = temp_input_file.name

    input_video = cv2.VideoCapture(temp_input_filepath)

    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_height = frame_height
    new_width = int(frame_height * 9 / 16)

    temp_output_filepath = tempfile.mktemp(suffix='.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(temp_output_filepath, fourcc, fps, (new_width, new_height))

    mp_face_detection = mp.solutions.face_detection
    num_frames_to_smooth = 5
    face_centers = deque(maxlen=num_frames_to_smooth)

    prev_x_center = frame_width // 2

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
        while input_video.isOpened():
            success, frame = input_video.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            x_center, _ = prev_x_center, frame_height // 2

            if results.detections:
                largest_face = None
                max_area = 0
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    area = bbox.width * bbox.height
                    if area > max_area:
                        max_area = area
                        largest_face = bbox

                x_center = int((largest_face.xmin + largest_face.width / 2) * frame_width)

            face_centers.append(x_center)
            x_center_smoothed = int(np.mean(face_centers))

            x_start = max(0, x_center_smoothed - new_width // 2)
            x_end = x_start + new_width

            if x_end > frame_width:
                x_end = frame_width
                x_start = x_end - new_width

            cropped_frame = frame[0:new_height, x_start:x_end]
            output_video.write(cropped_frame)

    input_video.release()
    output_video.release()

    temp_final_output_filepath = tempfile.mktemp(suffix='.mp4')

    ffmpeg_command = [
        'ffmpeg',
        '-i', temp_output_filepath,
        '-i', temp_input_filepath,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y',
        temp_final_output_filepath
    ]

    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with open(temp_final_output_filepath, 'rb') as final_output_file:
        final_output_bytes = final_output_file.read()

    os.remove(temp_input_filepath)
    os.remove(temp_output_filepath)
    os.remove(temp_final_output_filepath)

    return final_output_bytes
