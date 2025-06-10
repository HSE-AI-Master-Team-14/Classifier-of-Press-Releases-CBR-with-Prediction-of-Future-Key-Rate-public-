import streamlit as st
import requests, json, os, io, time
import pandas as pd

st.set_page_config(page_title="Предсказание ставки ЦБ на основе пресс-релизов", layout="centered")
st.title("Предсказание ставки ЦБ на основе пресс-релизов")

API = os.environ.get("API_URL", "http://fastapi:8000")

def api(path, *, method="post", json_body=None, files=None, params=None):
    url = f"{API}{path}"
    try:
        if method == "post":
            r = requests.post(url, json=json_body, files=files, params=params)
        else:
            r = requests.get(url, params=params)
        r.raise_for_status()
        try:
            return r.json()
        except json.JSONDecodeError:
            return r.content
    except requests.RequestException as e:
        st.error(f"Ошибка API: {e}")
        return None

st.subheader("Обучение модели")
lr  = st.selectbox("Learning rate", [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2],
                   format_func=lambda x:f"{x:.0e}", index=3)
bs  = st.selectbox("Batch size", [16,32,64,128], index=1)
ep  = st.slider("Epochs", 1, 50, 5)
pat = st.slider("Early-stopping patience", 1, 10, 2)

if st.button("Запустить обучение"):
    payload = {"learning_rate": lr, "batch_size": bs,
               "epochs": ep, "early_stopping_patience": pat}
    if api("/fit", json_body=payload):
        with st.spinner("Обучение запущено…"):
            while True:
                time.sleep(3)
                status = api("/fit_status", method="get")
                if status and status["status"] != "training":
                    break
        if status["status"] == "success":
            st.success(status["message"])
            if "metrics" in status: st.json(status["metrics"])
        else:
            st.error(status["message"])

st.subheader("Список моделей")
if st.button("Получить список моделей"):
    models = api("/models", method="get")
    if not models:
        st.info("Модели не найдены.")
    else:
        for m in models:
            st.write(f"**ID:** {m['model_id']}")
            if all(k in m for k in
                   ("learning_rate","batch_size","epochs_trained","patience")):
                st.write(f"lr={m['learning_rate']}, batch={m['batch_size']}, "
                         f"epochs={m['epochs_trained']}, "
                         f"patience={m['patience']}")
            if "metrics" in m:
                with st.expander("Метрики"): st.json(m["metrics"])
            st.write("---")

st.subheader("Установить активную модель")
mid = st.text_input("ID модели")
if st.button("Установить модель") and mid.strip():
    resp = api(f"/set?model_id={mid.strip()}", method="post")
    if resp: st.success(resp["message"])

st.subheader("Информация об активной модели")
if st.button("Получить информацию о модели"):
    info = api("/model_info", method="get")
    if info:
        st.write(f"**ID модели:** {info['model_id']}")
        st.json(info["metrics"])

st.subheader("Дообучение модели")
xlsx = st.file_uploader("XLSX (text, target)", type="xlsx")
if xlsx and st.button("Дообучить модель"):
    params = {"learning_rate": lr, "batch_size": bs,
              "epochs": ep, "early_stopping_patience": pat}
    files = {"file": (xlsx.name, xlsx,
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    with st.spinner("Дообучение модели…"):
        resp = api("/retrain", method="post", files=files, params=params)
    if resp:
        st.success("Модель дообучена")
        if "metrics" in resp: st.json(resp["metrics"])

st.subheader("Предсказание по одному пресс-релизу")
text = st.text_area("Текст релиза", height=180)
if st.button("Сделать прогноз") and text.strip():
    r = api("/one_predict", json_body={"text": text})
    if r:
        mapping = {-1:"Ставка снизится",0:"Ставка без изменений",1:"Ставка вырастет"}
        st.success(mapping.get(r["target"], r["target"]))

st.subheader("Предсказание по csv-файлу с текстами пресс-релизов")
csv = st.file_uploader("CSV с колонкой text", type="csv")
if csv and st.button("Получить предсказания"):
    df = pd.read_csv(csv)
    if "text" not in df.columns:
        st.error("Колонка text не найдена.")
    else:
        data = [{"text": t} for t in df["text"].astype(str)]
        csv_bytes = api("/multiple_predict", json_body=data)
        if csv_bytes:
            st.download_button("Скачать predictions.csv", csv_bytes,
                               file_name="predictions.csv", mime="text/csv")
