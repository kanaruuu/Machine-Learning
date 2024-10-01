import os
import subprocess
import cv2
import numpy as np
import librosa
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import tempfile
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import logging


# Параметры
VISUAL_THRESHOLD = 0.3  # Порог для визуального сходства. 0 - полная схожесть, 1 - максимальное различие. Если значение меньше порога, то шоты визуально похожи.
AUDIO_THRESHOLD = 0.2  # Порог для аудио сходства. 0 - полная схожесть, 1 - максимальное различие. Если меньше порога, то, соответственно, шоты похожи.
OBJECT_THRESHOLD = 0.49  # Порог для объектного сходства. ВАЖНО! 0 - ОТСУТСТВИЕ общих объектов, 1 - полное совпадение объектов. т.е. если БОЛЬШЕ порога, то они похожи.
SUBJECT_THRESHOLD = 0.49  # Порог для субъектного сходства. Аналогично OBJECT_THRESHOLD, но для субъектов.
MIN_SCENE_LENGTH = 2.5  # Минимальная длина сцены в секундах

# Веса для комбинирования признаков. Сумма весов ДОЛЖНА быть равна 1.
WEIGHT_VISUAL = 0.2  # Вес визуальных признаков
WEIGHT_AUDIO = 0.1  # Вес аудио признаков
WEIGHT_OBJECT = 0.15  # Вес объектов в кадре
WEIGHT_SUBJECT = 0.55  # Вес субъектов в кадре
COMBINED_THRESHOLD = 0.25  # Порог для комбинированного показателя сходства

VERBOSE = True


def compare_features(feat1, feat2):

    if np.linalg.norm(feat1) == 0 or np.linalg.norm(feat2) == 0:
        return 1.0
    return cosine(feat1, feat2)


def compare_objects(obj_set1, obj_set2):

    intersection = obj_set1.intersection(obj_set2)
    union = obj_set1.union(obj_set2)
    if not union:
        return 0
    return len(intersection) / len(union)


def compare_subjects(subj_set1, subj_set2):

    intersection = subj_set1.intersection(subj_set2)
    union = subj_set1.union(subj_set2)
    if not union:
        return 0
    return len(intersection) / len(union)


def extract_visual_features_from_shot(cap, visual_model, device):

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(min(5, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if not frames:
        return np.zeros(2048)

    features = []
    with torch.no_grad():
        for frame in frames:
            input_tensor = preprocess(frame).unsqueeze(0).to(device)
            output = visual_model(input_tensor)
            features.append(output.cpu().numpy().flatten())
    mean_feature = np.mean(features, axis=0)
    return mean_feature


def extract_audio_features_from_shot(shot_path):
    temp_audio_file = "temp_audio.wav"
    command = (
        f'ffmpeg -i "{shot_path}" -q:a 0 -map a "{temp_audio_file}" -y -loglevel quiet'
    )
    os.system(command)

    if not os.path.exists(temp_audio_file):
        return np.zeros(128)

    y, sr = librosa.load(temp_audio_file, sr=None)
    os.remove(temp_audio_file)

    if len(y) == 0:
        return np.zeros(128)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB_mean = np.mean(S_DB, axis=1)
    return S_DB_mean


def extract_object_features_from_shot(cap, model):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(min(3, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not frames:
        return set(), set()

    results = model(frames, imgsz=640)

    objects = set()
    subjects = set()
    subject_classes = {
        "person",
        "man",
        "woman",
        "child",
        "baby",
        "teenager",
        "elderly person",
        "boy",
        "girl",
        "infant",
        "toddler",
        "adolescent",
        "adult",
        "senior",
        "worker",
        "student",
        "teacher",
        "doctor",
        "nurse",
        "police officer",
        "firefighter",
        "soldier",
        "athlete",
        "musician",
        "artist",
        "dancer",
        "chef",
        "driver",
        "pilot",
        "captain",
        "scientist",
        "engineer",
        "programmer",
        "actor",
        "singer",
        "model",
        "reporter",
        "photographer",
        "businessperson",
        "lawyer",
        "judge",
        "politician",
        "priest",
        "monk",
        "nun",
        "coach",
        "trainer",
        "guide",
        "tourist",
        "referee",
        "student",
        "professor",
        "researcher",
        "athlete",
        "player",
        "swimmer",
        "runner",
        "climber",
        "cyclist",
        "skater",
        "golfer",
        "tennis player",
        "boxer",
        "martial artist",
        "yoga instructor",
        "lifeguard",
        "gardener",
        "farmer",
        "hunter",
        "fisherman",
        "zookeeper",
        "vet",
        "nanny",
        "baby sitter",
        "caregiver",
        "assistant",
        "secretary",
        "manager",
        "executive",
        "director",
        "supervisor",
        "customer",
        "client",
        "patient",
        "visitor",
        "guest",
        "resident",
        "citizen",
        "driver",
        "rider",
        "commuter",
        "traveler",
        "hiker",
        "backpacker",
        "camper",
        "explorer",
        "adventurer",
        "scout",
        "animal",
        "cat",
        "dog",
        "bird",
        "fish",
        "horse",
        "cow",
        "sheep",
        "pig",
        "chicken",
        "duck",
        "goose",
        "rabbit",
        "deer",
        "lion",
        "tiger",
        "elephant",
        "bear",
        "giraffe",
        "zebra",
        "monkey",
        "kangaroo",
        "panda",
        "koala",
        "alligator",
        "snake",
        "frog",
        "turtle",
        "shark",
        "whale",
        "dolphin",
        "octopus",
        "crab",
        "lobster",
        "spider",
        "bee",
        "butterfly",
        "ant",
        "mosquito",
        "squirrel",
        "fox",
        "wolf",
        "bat",
        "rat",
        "mouse",
        "hamster",
        "guinea pig",
        "parrot",
        "owl",
        "eagle",
        "hawk",
        "falcon",
        "pigeon",
        "seagull",
        "crow",
        "swan",
        "peacock",
        "flamingo",
        "penguin",
        "seal",
        "walrus",
        "otter",
        "chimpanzee",
        "gorilla",
        "baboon",
        "lemur",
        "sloth",
        "armadillo",
        "hedgehog",
        "meerkat",
        "mole",
        "platypus",
        "porcupine",
        "wombat",
    }

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            if conf > 0.3:
                class_name = model.names[cls]
                if class_name in subject_classes:
                    subjects.add(class_name)
                else:
                    objects.add(class_name)
    return objects, subjects


def merge_shots_videos(scenes) -> list:
    merged_videos_bytes = []  # Список для хранения конечных видео в байтах

    for idx, scene in enumerate(scenes, 1):
        scene_shot_videos_bytes = scene["shots"]
        temp_files = []

        # Сохраняем каждый шот как временный файл
        for shot_bytes in scene_shot_videos_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(shot_bytes)
                temp_files.append(temp_file.name)

        list_file = f"scene_{idx}_list.txt"
        with open(list_file, "w") as f:
            for temp_file_path in temp_files:
                f.write(f"file '{temp_file_path}'\n")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_output_file:
            temp_output_filepath = temp_output_file.name

        ffmpeg_command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            temp_output_filepath,
            "-y",
            "-loglevel",
            "error",
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
            with open(temp_output_filepath, "rb") as f:
                video_bytes = f.read()

            merged_videos_bytes.append(video_bytes)
            logging.info(f"Сцена {idx} успешно сохранена в байты!")

        except subprocess.CalledProcessError as e:
            print(f"Ошибка при объединении сцены {idx}: {e}")

        finally:
            os.remove(list_file)
            os.remove(temp_output_filepath)
            for temp_file_path in temp_files:
                os.remove(temp_file_path)

    return merged_videos_bytes


def merge_shots_into_scenes(video_bytes_list: list) -> list:
    shot_files = [video_bytes for _, video_bytes in video_bytes_list]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s.pt").to(device)
    model.overrides["verbose"] = False

    visual_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    visual_model.eval()

    visual_features = []
    audio_features = []
    object_features_list = []
    subject_features_list = []
    durations = []

    logging.info("Извлечение признаков из шотов...")

    for video_bytes in shot_files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_input_file:
            temp_input_file.write(video_bytes)
            temp_input_filepath = temp_input_file.name

        cap = cv2.VideoCapture(temp_input_filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        durations.append(duration)

        vis_feat = extract_visual_features_from_shot(cap, visual_model, device)
        visual_features.append(vis_feat)

        aud_feat = extract_audio_features_from_shot(temp_input_filepath)
        audio_features.append(aud_feat)

        obj_feat, subj_feat = extract_object_features_from_shot(cap, model)
        object_features_list.append(obj_feat)
        subject_features_list.append(subj_feat)

    visual_features = np.array(visual_features)
    audio_features = np.array(audio_features)

    visual_scaler = StandardScaler()
    audio_scaler = StandardScaler()

    if len(visual_features) > 1:
        visual_features_scaled = visual_scaler.fit_transform(visual_features)
    else:
        visual_features_scaled = visual_features

    if len(audio_features) > 1:
        audio_features_scaled = audio_scaler.fit_transform(audio_features)
    else:
        audio_features_scaled = audio_features

    scenes = []
    current_scene = {
        "shots": [shot_files[0]],  # Сохраняем первый шот (байты)
        "durations": [durations[0]],
    }

    for i in range(1, len(shot_files)):  # Идем по байтам шотов
        vis_dists = []
        aud_dists = []
        obj_sims = []
        subj_sims = []

        for j in range(len(current_scene["shots"])):
            index = i - j - 1
            if index < 0:
                break

            vis_dist = compare_features(
                visual_features_scaled[i], visual_features_scaled[index]
            )
            aud_dist = compare_features(
                audio_features_scaled[i], audio_features_scaled[index]
            )
            obj_sim = compare_objects(
                object_features_list[i], object_features_list[index]
            )
            subj_sim = compare_subjects(
                subject_features_list[i], subject_features_list[index]
            )

            vis_dists.append(vis_dist)
            aud_dists.append(aud_dist)
            obj_sims.append(obj_sim)
            subj_sims.append(subj_sim)

        vis_dist_avg = np.mean(vis_dists)
        aud_dist_avg = np.mean(aud_dists)
        obj_sim_avg = np.mean(obj_sims)
        subj_sim_avg = np.mean(subj_sims)

        combined_score = (
            (WEIGHT_VISUAL * vis_dist_avg)
            + (WEIGHT_AUDIO * aud_dist_avg)
            - (WEIGHT_OBJECT * obj_sim_avg)
            - (WEIGHT_SUBJECT * subj_sim_avg)
        )

        is_similar = combined_score < COMBINED_THRESHOLD

        if is_similar:
            current_scene["shots"].append(shot_files[i])
            current_scene["durations"].append(durations[i])
        else:
            scenes.append(current_scene)
            current_scene = {"shots": [shot_files[i]], "durations": [durations[i]]}

    scenes.append(current_scene)

    merged_scenes = []
    for scene in scenes:
        total_duration = sum(scene["durations"])
        if total_duration < MIN_SCENE_LENGTH:
            if merged_scenes:
                merged_scenes[-1]["shots"].extend(scene["shots"])
                merged_scenes[-1]["durations"].extend(scene["durations"])
                if VERBOSE:
                    logging.info(
                        f"Сцена слишком короткая ({total_duration:.2f} сек). Объединяем с предыдущей."
                    )
            else:
                merged_scenes.append(scene)
        else:
            merged_scenes.append(scene)

    merged_scenes = merge_shots_videos(merged_scenes)

    return merged_scenes
