<h1 align="center">Generating Viral Clips</h1>

## Как запустить бэкенд часть?
* Загрузите необходимые библиотеки:
```commandline
cd backend
pip install -r requirements.txt
```
* Далее, скопируйте .env: 
```commandline
cp .env.example .env
```
* С помощью следующей команды, запустите Postgresql и Adminer:
```commandline
docker compose up db adminer -d
```
* С помощью Adminer-а, либо консольной команды выполните SQL код, находящийся в файле migrations/initial_structure.sql
* Далее запустите все остальные сервисы, укажите в environments необходимые переменные: 
```commandline
docker compose up -d
```

## Как запустить фронтенд часть?
* Перейдите в папку frontend
```commandline
cd frontend
```
* Установите зависимости
```commandline
npm install
```
* Запустите веб приложение
```commandline
npm run dev
```
