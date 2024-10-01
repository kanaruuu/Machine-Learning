from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.schema import SystemMessage
from langchain_community.chat_models.gigachat import GigaChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
import psycopg2
import traceback
import pgsanity
import asyncio
import uvicorn
import os
import json
from langchain.docstore.document import Document

with open("rag.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

documents = []
for item in json_data:
    doc = Document(
        page_content=item["text"],  
        metadata={
            "id": item["id"],
            "title": item["title"],
            "page": item["page"],
            "keywords": item["keywords"]
        }
    )
    documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)

model_name = "DeepPavlov/rubert-base-cased"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs,
                                  encode_kwargs=encode_kwargs)

vector_store = FAISS.from_documents(documents, embedding=embedding)

embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 5})


prompt_context="""
Контекст: {context}
Вопрос:
Ниже представлен код запроса для СУБД Oracle. Преобразуй этот код для его использования в PostgreSQL, строго соблюдая следующие требования:

Перевод синтаксиса и функций: Если в Oracle используются специфические функции или операторы, замени их на точные эквиваленты в PostgreSQL. Например, для SEQUENCE, используй SERIAL или GENERATED. Обязательно комментируй каждую замену и поясняй, почему она сделана.
Адаптация типов данных: Внимательно следи за типами данных, которые могут отличаться между Oracle и PostgreSQL (например, VARCHAR2 в Oracle и VARCHAR в PostgreSQL). Перечисли все измененные типы данных и объясни, почему они были изменены.
Поддержка полной функциональности: Сохрани логику запроса без изменений. Не удаляй и не модифицируй части кода, которые не требуют преобразования.
Проверка совместимости: Убедись, что финальный код полностью соответствует стандартам PostgreSQL и корректно выполняется в этой СУБД. Если существуют ограничения, которые могут повлиять на выполнение, обязательно укажи их.
Пояснение работы запроса: Помимо преобразования, опиши в деталях, что делает этот запрос. Объясни шаги логики, чтобы было ясно, как работает запрос на уровне действий (выборка, фильтрация, агрегация и т.д.).
Присылание кода в sql Присылай только полный код, не нужно присылать его частями.
Использование EXCEPT Не используй MINUS, мне нужно через EXCEPT.
Конструкция SQL: Используй конструкцию SQL только для того, чтобы отправить весь код целиком. Для отдельных пояснений частей кода конструкцию SQL применять категорически запрещено, иначе система перестанет работать и я буду за это наказан.
Напиши обязательно описание переданного запроса. Обязательно!!!
Если в запросе есть CREATE TABLE или INSERT обязательно оставь их, у меня нет пальцев.
{input}
"""

prompt = ChatPromptTemplate.from_template(prompt_context)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Функция для создания экземпляра GigaChat в зависимости от выбранной модели
def create_chat_model(selected_model: str) -> GigaChat:
    return GigaChat(
        credentials="Yjg4MTQzMmUtNDAwMS00NDk0LThjOGUtNmU5ZWQ2YzQ4NDQ2OjcxODg0YjhmLTYyMDItNDQ5ZC04NjI2LTY1NjM0ZjU1MGJlYg==",
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        streaming=True,
        model=selected_model
    )

# Хранилище сообщений, начальное сообщение системы
messages = [
    SystemMessage(
        content=prompt_context)
]

class Message(BaseModel):
    text: str
    model: str = "GigaChat-Pro"  # Устанавливаем значение по умолчанию


def send_message(user_input: str, selected_model: str) -> str:
    # Используем модель выбранную пользователем
    document_chain = create_stuff_documents_chain(
    llm=create_chat_model(selected_model),
    prompt=prompt
    )
    retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
    response = retrieval_chain.invoke({"input": user_input})
    return response["answer"]

@app.post("/message/")
async def handle_message(message: Message):
    try:
        response_text = send_message(prompt_context + message.text, message.model)  # Передаем выбранную модель
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/check_query")
async def check_query(query):
  result = ''
  file_path = "temp/pstgrs.sql"
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, "w", encoding="utf-8") as file:
      file.write(query)
  try:
      process = await asyncio.create_subprocess_exec(
          'pgsanity', file_path
      )
  except Exception as e:
      print(e)
  
  stdout, stderr = await process.communicate()

  os.remove(file_path)

  if process.returncode == 0:
    result += "Синтаксис корректен"

  else:
    result += "Ошибка: " + stderr.decode()
    #return f'Ошибка: {stderr.decode()}'

  if "CREATE TABLE" in query:
    connection_string = 'postgresql://Moexdb_owner:Z6x5RkhAGJQm@ep-old-sound-a2zto35h.eu-central-1.aws.neon.tech/Moexdb?sslmode=require'
    try:
      psycopg2.connect(connection_string).autocommit = True
      con = psycopg2.connect(connection_string)
    except:
      result +="\nВ данный момент проверка недоступна"
      #return("В данный момент проверка недоступна")
    cursor = con.cursor()
    cursor.execute("""DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO pg_database_owner;""")
    try:
      cursor.execute(query)
      con.commit()
      con.close()
      result+="\nЗапрос не содержит ошибок"
      #return("Запрос не содержит ошибок")

    except Exception as e:
      con.close()
      formatted_lines = traceback.format_exc().splitlines()
      result += "\nЗапрос содержит ошибки:"+'\n'+ formatted_lines[-4][16:]+'\n'+ formatted_lines[-3]+'\n'+ formatted_lines[-2]+'\n'+ formatted_lines[-1]
  return(result)


@app.post("/check_sql/")
async def check_sql(message: Message):
    try:
        result = await check_query(message.text) 
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
