from transformers import MBartTokenizer, MBartForConditionalGeneration



def generate_fields(text):
    # Загрузка модели и токенизатора
    tokenizer = MBartTokenizer.from_pretrained('./fine_tuned_mbart')
    model_describer = MBartForConditionalGeneration.from_pretrained('./fine_tuned_mbart')

    # Установка языка модели на русский
    tokenizer.src_lang = "ru_RU"
    tokenizer.tgt_lang = "ru_RU"

    # Подготовка входного текста
    inputs = tokenizer(f"Текст: {text}", return_tensors="pt", max_length=512, truncation=True)

    # Генерация выходных данных
    outputs = model_describer.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)

    # Декодирование выходных данных
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text