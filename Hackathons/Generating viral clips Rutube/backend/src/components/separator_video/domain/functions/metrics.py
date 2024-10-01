import re
import math
import nltk
import pyphen
import spacy
from nltk.corpus import stopwords
import torch
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
from torch import argmax
import torch.nn.functional as F
from typing import Tuple

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("russian"))

TOPIC_PRIORITIES = {
    "мемы": 0.9,
    "юмор": 0.9,
    "музыка": 0.5,
    "лайфхак": 0.9,
    "советы": 0.9,
    "животные": 0.9,
    "спорт": 0.6,
    "культура": 0.3,
    "технологии": 0.4,
    "политика": 0.9,
    "видеоигры": 0.8,
    "мода": 0.7,
    "новости ": 0.9,
    "розыгрыш": 0.6,
    "рецепты": 0.2,
    "история": 0.3,
    "медиа": 0.7,
    "наука": 0.4,
    "автомобили": 0.7,
    "кино": 0.4,
    "космос": 0.3,
    "искусство": 0.3,
    "путешествие": 0.4,
    "здоровье": 0.8,
    "фотография": 0.3,
    "образование": 0.4,
    "философия": 0.2,
    "роботы": 0.4,
    "программирование": 0.6,
    "игры": 0.7,
    "литература": 0.3,
    "финансы": 0.3

}
EMOTION_PRIORITIES = {
    "joy": 1.0,
    "anger": 0.9,
    "enthusiasm": 0.8,
    "surprise": 0.8,
    "neutral": 0.5,
    "sadness": 0.3,
    "disgust": 0.2,
    "fear": 0.2,
    "guilt": 0.2,
    "shame": 0.2,
}
WEIGHTS_COMPLEXITY = {
    "avg_sentence_length": 0.25,
    "avg_word_length": 0.20,
    "avg_syllables_per_word": 0.20,
    "polysyllabic_percentage": 0.20,
    "lexical_diversity": 0.15,
}

try:
    nlp = spacy.load("ru_core_news_md")
except OSError:
    from spacy.cli import download

    download("ru_core_news_md")
    nlp = spacy.load("ru_core_news_md")


def count_syllables(word: str) -> int:
    """
    Подсчитывает количество слогов в слове.

    :param word: Слово для проверки.
    :return: Количество слогов в слове.
    """
    dic = pyphen.Pyphen(lang="ru")
    hyphens = dic.inserted(word)
    return max(1, hyphens.count("-") + 1)


def is_polysyllabic(word: str) -> bool:
    """
    Определяет, является ли слово полисиллабическим (более трех слогов).

    :param word: Слово для проверки.
    :return: 1, если слово полисиллабическое, иначе 0.
    """
    return count_syllables(word) > 3


def preprocess_text(text: str) -> str:
    """
    Предобрабатывает текст: удаляет символы и приводит слова к начальной форме.

    :param text: Исходный текст.
    :return: Предобработанный текст.
    """
    cleaned = re.sub(r"[^а-яА-Я\s]", "", text).lower()
    doc = nlp(cleaned)
    lemmatized = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOP_WORDS and token.is_alpha
    ]

    return " ".join(lemmatized)


def get_text_metrics(text: str) -> dict:
    """
    Извлекает метрики текста, включая количество предложений и слов.

    :param text: Обработанный текст.
    :return: Словарь с метриками текста.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    words = [token.text for token in doc if token.is_alpha]

    num_sentences = len(sentences)
    num_words = len(words)
    num_letters = sum(len(word) for word in words)
    num_syllables = sum(count_syllables(word) for word in words)
    num_polysyllables = sum(is_polysyllabic(word) for word in words)
    unique_words = set(word.lower() for word in words)
    num_unique_words = len(unique_words)

    return {
        "num_sentences": num_sentences,
        "num_words": num_words,
        "num_letters": num_letters,
        "num_syllables": num_syllables,
        "num_polysyllables": num_polysyllables,
        "num_unique_words": num_unique_words,
    }


def normalize(value, min_val, max_val) -> float:
    """
    Нормализует значение в диапазоне [0, 1].

    :param value: Значение для нормализации.
    :param min_val: Минимальное значение диапазона.
    :param max_val: Максимальное значение диапазона.
    :return: Нормализованное значение.
    """
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def analyze_text_complexity_custom(text: str) -> float:
    """
    Анализирует сложность текста и возвращает оценку его простоты.

    :param text: Исходный текст для анализа.
    :return: Оценка простоты текста (0-100).
    """
    preprocessed_text = preprocess_text(text)
    metrics = get_text_metrics(preprocessed_text)

    if metrics["num_sentences"] == 0 or metrics["num_words"] == 0:
        return 0

    avg_sentence_length = metrics["num_words"] / metrics["num_sentences"]
    avg_word_length = metrics["num_letters"] / metrics["num_words"]
    avg_syllables_per_word = metrics["num_syllables"] / metrics["num_words"]
    polysyllabic_percentage = (
                                      metrics["num_polysyllables"] / metrics["num_words"]
                              ) * 100
    lexical_diversity = metrics["num_unique_words"] / metrics["num_words"]

    try:
        norm_avg_sentence_length = normalize(
            math.log(avg_sentence_length), math.log(5), math.log(30)
        )
    except:
        norm_avg_sentence_length = 0.5
    norm_avg_word_length = normalize(avg_word_length, 4, 10)
    norm_avg_syllables_per_word = normalize(avg_syllables_per_word, 1, 5)
    norm_polysyllabic_percentage = normalize(polysyllabic_percentage, 0, 50)
    norm_lexical_diversity = normalize(lexical_diversity, 0.3, 0.8)

    norm_avg_sentence_length = min(max(norm_avg_sentence_length, 0), 1)
    norm_avg_word_length = min(max(norm_avg_word_length, 0), 1)
    norm_avg_syllables_per_word = min(max(norm_avg_syllables_per_word, 0), 1)
    norm_polysyllabic_percentage = min(max(norm_polysyllabic_percentage, 0), 1)
    norm_lexical_diversity = min(max(norm_lexical_diversity, 0), 1)

    complexity_score = (
            WEIGHTS_COMPLEXITY["avg_sentence_length"] * norm_avg_sentence_length
            + WEIGHTS_COMPLEXITY["avg_word_length"] * norm_avg_word_length
            + WEIGHTS_COMPLEXITY["avg_syllables_per_word"] * norm_avg_syllables_per_word
            + WEIGHTS_COMPLEXITY["polysyllabic_percentage"] * norm_polysyllabic_percentage
            + WEIGHTS_COMPLEXITY["lexical_diversity"] * (1 - norm_lexical_diversity)
    )

    complexity_score = complexity_score * 100
    complexity_score = max(1, min(round(complexity_score, 2), 100))
    simplicity_score = 100 - complexity_score

    return simplicity_score


def get_simplicity_score(text: str) -> float:
    """
    Получает оценку простоты текста (нормализированное значение).

    :param text: Исходный текст для анализа.
    :return: Нормализованная оценка простоты (0-1).
    """
    simplicity_score = analyze_text_complexity_custom(text)
    normalized_score = simplicity_score / 100

    return normalized_score


def calculate_score_from_length(text):
    max_words = 60
    words = text.split()
    word_count = len(words)
    score = word_count / max_words
    return min(score, 1.0)


@torch.no_grad()
def predict_sentiment(text):
    """
    Предсказывает тональность текста.

    :param text: Текст для анализа.
    :return: Массив предсказанных тональностей.
    """
    tokenizer_sentiment = BertTokenizerFast.from_pretrained(
        "blanchefort/rubert-base-cased-sentiment"
    )
    model_sentiment = AutoModelForSequenceClassification.from_pretrained(
        "blanchefort/rubert-base-cased-sentiment", return_dict=True
    )
    inputs = tokenizer_sentiment(
        text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model_sentiment(**inputs)
    predicted = F.softmax(outputs.logits, dim=1)
    predicted = argmax(predicted, dim=1).numpy()

    return predicted


def get_sentiment_score(text: str) -> Tuple[float, str]:
    """
    Получает оценку тональности текста и его метку.

    :param text: Исходный текст для анализа.
    :return: Кортеж с оценкой тональности и меткой.
    """
    sentiment_label = predict_sentiment(text)[0]
    if sentiment_label == 0:
        return 0.3, "NEUTRAL"
    elif sentiment_label == 1:
        return 1.0, "POSITIVE"
    elif sentiment_label == 2:
        return 1.0, "NEGATIVE"
    else:
        return 0.5, "NEUTRAL"


def get_emotion_score(text: str) -> Tuple[float, str]:
    """
    Определяет эмоциональную окраску текста.

    :param text: Исходный текст для анализа.
    :return: Кортеж с оценкой эмоциональности и меткой.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_emotion = pipeline(
        "sentiment-analysis", model="Djacon/rubert-tiny2-russian-emotion-detection", device=device
    )
    result_emotion = classifier_emotion(text)[0]
    emotion = result_emotion["label"]

    priority = EMOTION_PRIORITIES.get(emotion)

    score_num = priority

    return score_num, emotion


def get_topic_score(text: str) -> Tuple[float, list]:
    """
    Определяет тематику текста и возвращает оценку и используемые темы.

    :param text: Исходный текст для анализа.
    :return: Кортеж с оценкой тематики и списком использованных тем.
    """
    candidate_labels = [
        "мемы",
        "юмор",
        "музыка",
        "лайфхак",
        "советы",
        "животные",
        "спорт",
        "культура",
        "технологии",
        "политика",
        "видеоигры",
        "технологии ",
        "мода",
        "новости ",
        "розыгрыш",
        "рецепты",
        "история",
        "медиа",
        "наука",
        "автомобили",
        "кино",
        "космос",
        "искусство",
        "путешествие",
        "здоровье",
        "фотография",
        "образование",
        "философия",
        "роботы",
        "программирование",
        "игры",
        "литература",
        "финансы"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_topic = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device)
    result = classifier_topic(text, candidate_labels)
    adjusted_score = 0.0
    topics_used = []
    for i in range(3):
        top_score = result["scores"][i]
        top_label = result["labels"][i]

        priority = TOPIC_PRIORITIES.get(top_label)
        priority = TOPIC_PRIORITIES.get(top_label)
        adjusted_score += top_score * priority
        topics_used.append((top_label, top_score, priority))

    return adjusted_score, topics_used


def evaluate_text(text: str, weights: dict = None) -> float:
    """
    Оценивает текст на основе тематической, тональной, эмоциональной и простоты.

    :param text: Исходный текст для анализа.
    :param weights: Словарь весов для оценки (по умолчанию равномерные).
    :return: Итоговая оценка текста.
    """
    if weights is None:
        weights = {
            "topic_weight": 0.25,
            "sentiment_weight": 0.25,
            "emotion_weight": 0.25,
            "simplicity_weight": 0.25,
        }
    else:
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

    topic_score, _ = get_topic_score(text)
    sentiment_score, _ = get_sentiment_score(text)
    emotion_score, _ = get_emotion_score(text)
    simplicity_score = get_simplicity_score(text)
    score_from_length = calculate_score_from_length(text)

    final_score = (
            weights["topic_weight"] * topic_score
            + weights["sentiment_weight"] * sentiment_score
            + weights["emotion_weight"] * emotion_score
            + weights["simplicity_weight"] * simplicity_score
            + weights["length_weight"] * score_from_length
    )

    final_score_scaled = final_score * 9 + 1
    final_score_scaled = round(final_score_scaled, 2)

    return final_score_scaled


def start_proccess(text: str) -> str:
    """
    Запускает процесс оценки текста с заданными весами.

    :param text: Текст для анализа.
    :return: Результат оценки текста.
    """
    weights = {
        "topic_weight": 0.60,
        "sentiment_weight": 0.15,
        "emotion_weight": 0.15,
        "simplicity_weight": 0.05,
        "length_weight": 0.05
    }

    score = evaluate_text(text, weights)
    return score