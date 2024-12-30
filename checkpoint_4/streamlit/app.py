import streamlit as st
import requests
import json
import os
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

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

# Вкладка для одного предсказания
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

# Вкладка для множества предсказаний / Исследование датасета
st.subheader("Обработка CSV файла")
uploaded_file = st.file_uploader("Загрузите CSV файл с текстами пресс-релизов (колонка 'text')", type="csv")
operation_type = st.radio("Выберите, что хотите сделать с датасетом:", ("Исследовать", "Предсказать ставку"))

if st.button("Выполнить"):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            if df.empty:
                st.error("DataFrame пустой. Проверьте содержимое CSV файла.")
            elif 'text' not in df.columns:
                st.error("CSV файл должен содержать колонку 'text'")
            else:
                df = df.dropna(subset=['text'])
                df = df[df['text'] != '']
                if df.empty:
                    st.error("После очистки DataFrame оказался пустым. Проверьте данные.")
                else:
                    if operation_type == "Предсказать ставку":
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
                    elif operation_type == "Исследовать":
                        st.write(f"**Количество строк:** {df.shape[0]}")
                        st.write(f"**Количество колонок:** {df.shape[1]}")
                        st.write("**Первые 5 строк:**")
                        st.write(df.head())

                        df['text_length'] = df['text'].str.len()

                        st.write("**Статистика по длине пресс-релизов:**")
                        st.write(f"Средняя длина: {df['text_length'].mean():.2f} символов")
                        st.write(f"Минимальная длина: {df['text_length'].min()} символов")
                        st.write(f"Максимальная длина: {df['text_length'].max()} символов")

                        fig, ax = plt.subplots()
                        sns.histplot(df['text_length'], ax=ax, bins=50)
                        ax.set_title('Распределение длины пресс-релизов')
                        ax.set_xlabel('Длина (количество символов)')
                        ax.set_ylabel('Количество пресс-релизов')
                        st.pyplot(fig)

                        fig_box, ax_box = plt.subplots()
                        sns.boxplot(x=df['text_length'], ax=ax_box)
                        ax_box.set_title('Boxplot длины пресс-релизов')
                        ax_box.set_xlabel('Длина (количество символов)')
                        st.pyplot(fig_box)

        except pd.errors.ParserError as e:
            st.error(f"Ошибка при чтении CSV файла: {e}. Убедитесь, что файл имеет правильный формат и кодировку.")
        except Exception as e:
            st.error(f"Произошла непредвиденная ошибка: {e}")
    else:
        st.warning("Пожалуйста, загрузите CSV файл.")

# Вкладка для информации о модели
st.subheader("Информация о модели")
if st.button("Получить информацию о модели"):
    model_info = make_api_request(f"{api_url}/model_info", method="get")
    if model_info:
        st.write(f"**Название модели:** {model_info.get('model_name', 'Неизвестно')}")
        st.write(f"**Версия модели:** {model_info.get('model_version', 'Неизвестно')}")
