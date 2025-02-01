import streamlit as st
import requests
import json
import os
import pandas as pd
import io
import time

st.title("Предсказание ставки ЦБ на основе пресс-релизов")

api_url = os.environ.get("API_URL", "http://fastapi:8000")

# Функция для выполнения запросов к API
def make_api_request(url, method="post", data=None):
    with st.spinner("Выполнение запроса..."):
        try:
            headers = {'Content-Type': 'application/json'} if data and method == "post" else None
            if method == "post":
                response = requests.post(
                    url,
                    data=json.dumps(data, ensure_ascii=False).encode('utf-8') if data else None,
                    headers=headers
                )
            elif method == "get":
                response = requests.get(url, headers=headers)
            response.raise_for_status()
            if response.content:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return response.content
            return None
        except requests.exceptions.RequestException as e:
            try:
                error_message = (
                    response.json().get("detail", str(e))
                    if 'response' in locals() and response.content
                    else str(e)
                )
            except (json.JSONDecodeError, AttributeError):
                error_message = str(e)
            st.error(f"Ошибка при запросе к API: {error_message}")
            return None
        except Exception as e:
            st.error(f"Непредвиденная ошибка: {e}")
            return None

# Вкладка для обучения модели
st.subheader("Обучение модели")

C_value = st.slider("Выберите значение C", min_value=0.01, max_value=100.0, value=1.0, step=0.01)

# Кнопка для запуска обучения
if st.button("Запустить обучение"):
    st.write(f"Отправляемое значение C: {C_value}")
    with st.spinner("Запуск обучения..."):
        response = make_api_request(f"{api_url}/fit", method="post", data={"C": C_value})
        
        if response:
            status = response.get("status")
            message = response.get("message")
            
            if status == "training_started":
                st.info(message)
                
                while True:
                    time.sleep(2)
                    result = make_api_request(f"{api_url}/fit_status", method="get")
                    if result and result.get("status") in ["success", "error"]:
                        st.success(f"Обучение завершено: {result.get('message')}")
                        break
                    elif result:
                        st.info(f"Статус обучения: {result.get('status')}")
                    else:
                        st.error("Не удалось получить статус обучения.")
                        break
            else:
                st.error(f"Ошибка: {message}")
        else:
            st.error("Не удалось запустить обучение модели.")

# Вкладка для списка моделей
st.subheader("Список моделей")
if st.button("Получить список моделей"):
    models = make_api_request(f"{api_url}/models", method="get")
    if models:
        for model in models:
            st.write(f"Модель ID: {model['model_id']}")
            st.write(f"Тип модели: {model['model_type']}")
            st.write(f"Параметры модели: {model['model_params']}")
            st.write("---")
    else:
        st.error("Не удалось получить список моделей.")

# Вкладка для установки активной модели
st.subheader("Установить активную модель")
model_id = st.text_input("Введите ID модели")

if st.button("Установить модель"):
    response = make_api_request(f"{api_url}/set?model_id={model_id}", method="post")
    if response:
        st.success(f"Модель '{model_id}' установлена как активная.")
    else:
        st.error("Ошибка при установке модели.")

# Вкладка для дообучения модели
st.subheader("Дообучение модели")
uploaded_file = st.file_uploader("Загрузите новый датасет для дообучения", type="xlsx")
if uploaded_file and st.button("Дообучить модель"):
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file, "xlsx")
        }
        with st.spinner("Дообучение модели..."):
            response = requests.post(f"{api_url}/retrain", files=files)
            response.raise_for_status()
            result = response.json()
            st.success("Модель успешно дообучена!")
    except Exception as e:
        st.error(f"Ошибка при дообучении модели: {e}")

# Вкладка для предсказания по одному пресс-релизу
st.subheader("Предсказание по одному пресс-релизу")
release_text = st.text_area("Вставьте текст пресс-релиза", height=200)
if st.button("Сделать прогноз"):
    if release_text:
        prediction = make_api_request(f"{api_url}/one_predict", data={"text": release_text})
        if prediction:
            target = prediction.get("target")
            if target == 0:
                st.success("Ставка не изменится")
            elif target == 1:
                st.success("Ставка вырастет")
            elif target == -1:
                st.success("Ставка снизится")
            else:
                st.warning("Неожиданный результат прогноза. Пожалуйста, проверьте API.")
        else:
            st.error("Ошибка при получении прогноза.")
    else:
        st.warning("Пожалуйста, вставьте текст пресс-релиза.")

# Вкладка для предсказания для нескольких пресс-релизов
st.subheader("Предсказание для множества пресс-релизов")
uploaded_file_multiple = st.file_uploader("Загрузите CSV файл с текстами пресс-релизов (колонка 'text')", type="csv")
if uploaded_file_multiple and st.button("Получить предсказания"):
    if uploaded_file_multiple:
        try:
            df = pd.read_csv(uploaded_file_multiple, encoding='utf-8')
            if 'text' not in df.columns:
                st.error("CSV файл должен содержать колонку 'text'")
            else:
                data = [{"text": row['text']} for _, row in df.iterrows()]
                response_content = make_api_request(f"{api_url}/multiple_predict", data=data)

                if response_content:
                    try:
                        results = json.loads(response_content.decode('utf-8'))
                        results_df = pd.DataFrame(results)
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                        csv_buffer.seek(0)
                        st.download_button(
                            label="Скачать в CSV",
                            data=csv_buffer.getvalue().encode('utf-8'),
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    except json.JSONDecodeError:
                        try:
                            st.download_button(
                                label="Скачать результаты в CSV",
                                data=response_content,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Ошибка при обработке ответа сервера (не JSON): {e}")
        except pd.errors.ParserError as e:
            st.error(f"Ошибка при чтении CSV файла: {e}. Убедитесь, что файл имеет правильный формат и кодировку.")
        except Exception as e:
            st.error(f"Произошла непредвиденная ошибка: {e}")
    else:
        st.warning("Пожалуйста, загрузите CSV файл.")
