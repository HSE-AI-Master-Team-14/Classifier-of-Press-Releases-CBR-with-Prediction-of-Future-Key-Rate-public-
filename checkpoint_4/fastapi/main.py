from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks, Query, UploadFile, File, Request
from pydantic import BaseModel, Field
from typing import List, Annotated, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
import multiprocessing
import os
import uvicorn
from fastapi.responses import StreamingResponse, JSONResponse
import io
import asyncio
import signal
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from nltk.corpus import stopwords
from pymystem3 import Mystem
import logging

# Расширяем список русских стоп-слов
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['это', 'совет', 'директор', 'банк', 'россия', 'годовой', 'год'])

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="API для предсказания решений об изменении ключевой ставки ЦБ")

# Глобальные переменные для модели
active_model = None
active_preprocessor = None
active_model_id = None

class Hyperparameters(BaseModel):
    C: float = 1.0

class InputData(BaseModel):
    text: Annotated[str, Field(description="Текст пресс-релиза")]

class PredictionResult(BaseModel):
    target: Annotated[int, Field(
        description="Предсказанное изменение ставки (-1: понижение, 0: без изменений, 1: повышение)"
    )]

# При старте приложения загружаем необходимые ресурсы
@app.on_event("startup")
async def startup_event():
    try:
        global mystem_analyzer
        mystem_analyzer = Mystem()
        print("Mystem успешно установлен.")
    except Exception as e:
        print(f"Ошибка при установке Mystem: {str(e)}")

    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        print("Ресурсы NLTK загружены.")
    except Exception as e:
        print(f"Ошибка при загрузке NLTK: {str(e)}")

async def load_and_preprocess(model_id: str):
    try:
        if model_id == "best_model":
            model_path = "best_model.joblib"
            preprocessor_path = "best_model_preprocessor.joblib"
        else:
            model_path = "latest_best_model.joblib"
            preprocessor_path = "latest_best_model_preprocessor.joblib"

        logger.info(f"Загрузка модели из файла: {model_path}")
        logger.info(f"Загрузка предобработчика из файла: {preprocessor_path}")

        # Загружаем модель и предобработчик
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        # Проверяем соответствие признаков
        if len(preprocessor.get_feature_names_out()) != model.coef_.shape[1]:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Несоответствие признаков: предобработчик ({len(preprocessor.get_feature_names_out())}) "
                    f"!= модель ({model.coef_.shape[1]})"
                )
            )

        logger.info(f"Модель '{model_id}' и предобработчик успешно загружены.")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели или предобработчика '{model_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели или предобработчика: {str(e)}")

def train_model(hyperparameters: Hyperparameters, queue):
    try:
        logger.info(f"Начало обучения модели. Параметры: C={hyperparameters.C}")
        
        # Загрузка данных
        df = pd.read_excel("dataset.xlsx")
        logger.info(f"Данные загружены: {df.shape[0]} строк.")

        X = df['text']
        y = df['target']

        # Предобработка текста
        logger.info("Начинается предобработка текста.")
        X = [" ".join([w for w in word_tokenize(t) if w.isalpha()]) for t in X]
        X = [mystem_analyzer.lemmatize(text) for text in X]
        X = [[j for j in i if j != ' ' and j != '\n'] for i in X]
        X = [" ".join([w for w in t]) for t in X]
        logger.info("Предобработка завершена.")

        # Создаем TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=russian_stopwords)
        X_transformed = vectorizer.fit_transform(X)

        # Обучение модели
        logger.info("Начинается обучение модели LogisticRegression.")
        model = LogisticRegression(C=hyperparameters.C)
        model.fit(X_transformed, y)
        logger.info("Модель успешно обучена.")

        # Сохранение модели и предобработчика
        joblib.dump(model, "latest_best_model.joblib")
        joblib.dump(vectorizer, "latest_best_model_preprocessor.joblib")
        logger.info("Модель сохранена как latest_best_model.joblib")
        logger.info("Предобработчик сохранен как latest_best_model_preprocessor.joblib")

        queue.put({"status": "success", "message": "Модель успешно обучена"})
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        queue.put({"status": "error", "message": str(e)})

async def get_model_info(model_path):
    try:
        model = joblib.load(model_path)
        return {
            "model_type": type(model).__name__,
            "model_params": model.get_params()
        }
    except FileNotFoundError:
        return None

training_status = {"status": "idle", "message": "Ожидание начала обучения"}

@app.post("/fit", summary="Обучение модели")
async def fit_model(background_tasks: BackgroundTasks, request: Request):
    try:
        body = await request.json()
        logger.info(f"Тело запроса: {body}")

        hyperparameters = Hyperparameters(**body)
        logger.info(f"Полученные параметры: {hyperparameters.dict()}")

        logger.info("Получен запрос на обучение модели.")
        training_status.update({"status": "training", "message": "Модель обучается..."})

        def training_task(params: Hyperparameters):
            try:
                queue = multiprocessing.Queue()
                logger.info("Создан новый процесс для обучения модели.")
                p = multiprocessing.Process(target=train_model, args=(params, queue))
                p.start()
                p.join(10)

                if p.is_alive():
                    logger.warning("Процесс обучения превысил лимит времени. Завершается.")
                    os.kill(p.pid, signal.SIGKILL)
                    training_status.update({"status": "error", "message": "Обучение превысило лимит времени"})
                else:
                    result = queue.get()
                    logger.info(f"Обучение завершено с результатом: {result}")
                    training_status.update(result)
            except Exception as e:
                logger.error(f"Ошибка в процессе обучения: {str(e)}")
                training_status.update({"status": "error", "message": str(e)})

        background_tasks.add_task(training_task, hyperparameters)
        logger.info("Фоновая задача для обучения модели добавлена.")
        return JSONResponse(
            content={"status": "training_started", "message": "Обучение модели запущено"}, 
            status_code=202
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=400)

@app.get("/fit_status")
async def get_fit_status():
    logger.info(f"Запрос текущего статуса обучения: {training_status}")
    return JSONResponse(content=training_status)

@app.post("/one_predict", response_model=PredictionResult)
async def predict_one_rate(data: InputData) -> PredictionResult:
    logger.info(f"Запрос на предсказание: {data.text}")

    if active_model is None or active_preprocessor is None:
        logger.error("Активная модель не установлена.")
        raise HTTPException(status_code=400, detail="Активная модель не установлена.")

    try:
        logger.info(f"Используемая модель: {active_model_id}")
        logger.info(f"Текст перед преобразованием: {data.text}")
        
        # Преобразование текста
        processed_data = active_preprocessor.transform([data.text])
        logger.info(f"Размер преобразованных данных: {processed_data.shape}")
        
        # Предсказание
        prediction = int(active_model.predict(processed_data)[0])
        logger.info(f"Предсказание выполнено: {prediction}")
        return PredictionResult(target=prediction)
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

@app.post("/multiple_predict", summary="Предсказания для нескольких объектов", tags=["Prediction"])
async def predict_multiple_rates(data: List[InputData]) -> StreamingResponse:
    try:
        if active_model is None or active_preprocessor is None:
            raise HTTPException(status_code=400, detail="Активная модель не установлена.")

        results = []
        for item in data:
            try:
                processed_item = active_preprocessor.transform([item.text])
                prediction = int(active_model.predict(processed_item)[0])
                results.append({"text": item.text, "target": prediction})
            except Exception as e:
                results.append({"text": item.text, "target": None, "error": str(e)})

        results_df = pd.DataFrame(results)

        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()), 
            media_type='text/csv', 
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказаниях: {str(e)}")

@app.get("/models", summary="Список моделей", response_model=List[Dict[str, Any]])
async def list_models():
    models_info = []
    for filename in os.listdir():
        if filename.endswith("_model.joblib"):
            model_path = filename
            model_id = filename[:-7]
            info = await get_model_info(model_path)
            if info:
                models_info.append({"model_id": model_id, **info})

    return models_info

@app.post("/set")
async def set_active_model(model_id: str):
    global active_model, active_preprocessor, active_model_id

    logger.info(f"Запрос на установку активной модели: {model_id}")

    # Сбрасываем глобальные переменные
    active_model = None
    active_preprocessor = None
    active_model_id = None

    try:
        # Загружаем новую модель и предобработчик
        active_model, active_preprocessor = await load_and_preprocess(model_id)
        active_model_id = model_id

        logger.info(f"Активная модель успешно установлена: {model_id}")
        return {"message": f"Модель '{model_id}' и её предобработчик успешно установлены."}
    except Exception as e:
        logger.error(f"Ошибка при установке модели '{model_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при установке модели: {str(e)}")

@app.post("/retrain", summary="Дообучение модели")
def retrain_model(file: UploadFile = File(...), hyperparameters: Hyperparameters = Depends()):
    global active_model, active_preprocessor

    try:
        if active_model is None or active_preprocessor is None:
            active_model, active_preprocessor = asyncio.run(load_and_preprocess(active_model_id))

        contents = file.file.read()
        df_new = pd.read_excel(io.BytesIO(contents))

        X_new = active_preprocessor.transform(df_new['text']).toarray()

        df = pd.read_excel("dataset.xlsx")
        X = df['text']
        y = df['target']
        X_retrain = active_preprocessor.transform(X).toarray()
        y_retrain = y.values
        
        X_combined = np.concatenate((X_retrain, X_new))
        y_combined = np.concatenate((y_retrain, df_new['target'].values))

        active_model.fit(X_combined, y_combined)
        joblib.dump(active_model, "latest_best_model.joblib")
        return {"message": "Модель успешно дообучена"}

    except Exception as e:
        raise HTTPException(500, f"Ошибка при дообучении: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
