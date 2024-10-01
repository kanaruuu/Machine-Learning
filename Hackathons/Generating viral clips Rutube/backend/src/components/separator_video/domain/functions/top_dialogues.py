import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
import tempfile
from nltk.corpus import stopwords
from math import ceil
import nltk
import os

nltk.download('stopwords')
nlp = spacy.load('ru_core_news_md')

STOP_WORDS = set(stopwords.words("russian"))


def preprocess_dialogue(dialogue):
    cleaned = re.sub(r'[^а-яА-Я\s]', '', dialogue).lower()
    tokens = cleaned.split()
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS and token.is_alpha]

    return ' '.join(lemmatized)


def get_video_duration_from_bytes(video_bytes):
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
        print(f"Ошибка при получении длительности видео: {e}")
        duration = 0
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    return duration


def get_top_dialogues_tfidf(subtitles: str, top_n=10):
    doc = nlp(subtitles)
    dialogues = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    processed_dialogues = [preprocess_dialogue(dialogue) for dialogue in dialogues]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_dialogues)
    avg_tfidf = tfidf_matrix.mean(axis=1).A1
    dialogue_scores = list(zip(dialogues, avg_tfidf))
    dialogue_scores.sort(key=lambda x: x[1], reverse=True)
    top_dialogues = dialogue_scores[:top_n]
    result = [dialogue.strip() for dialogue, _ in top_dialogues]

    return result


