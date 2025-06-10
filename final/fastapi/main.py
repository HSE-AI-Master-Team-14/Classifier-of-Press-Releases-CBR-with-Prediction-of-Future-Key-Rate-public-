import io
import os
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field, confloat, conint
import dl_model_manager as dlm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="API-DL: прогноз решения ЦБ")

active_model: Optional[tf.keras.Model] = None
active_tokenizer = None
active_model_id: Optional[str] = None

training_status: Dict[str, Any] = {
    "status": "idle",
    "message": "Ожидание начала обучения",
    "plot_available": False,
}

class DLHyper(BaseModel):
    learning_rate: confloat(ge=1e-5, le=1e-2) = 5e-4
    batch_size:    conint(ge=16,  le=128)      = 32
    epochs:        conint(ge=1,   le=50)      = 10
    early_stopping_patience: conint(ge=1, le=10) = 3

class InputData(BaseModel):
    text: str = Field(...)

class PredictionResult(BaseModel):
    target: int = Field(..., description="-1 ↓  0 =  1 ↑")

@app.post("/fit", summary="Обучить новую DL-модель")
async def fit_model(bg: BackgroundTasks, req: Request):
    params = DLHyper(**await req.json())
    logger.info(f"/fit  {params.dict()!r}")

    training_status.update(status="training",
                           message="Запуск обучения…",
                           plot_available=False)

    def _run(p: DLHyper):
        try:
            df = pd.read_csv("dataset.csv")
            res = dlm.train_new(df,
                                p.learning_rate,
                                p.batch_size,
                                p.epochs,
                                p.early_stopping_patience)
            training_status.update(
                status="success",
                message=f"Модель {res['model_id']} обучена",
                plot_available=True,
                metrics=res["metrics"],
                model_id=res["model_id"]
            )
        except Exception:
            tb = traceback.format_exc()
            training_status.update(status="error", message=tb)
            logger.error(tb)

    bg.add_task(_run, params)
    return JSONResponse({"status": "training_started"}, 202)

@app.get("/fit_status")
async def fit_status():
    return JSONResponse(training_status)

@app.get("/models")
async def models() -> List[Dict[str, Any]]:
    info = []
    if Path("dl_model.keras").exists():
        info.append({"model_id": "preloaded"})
    for meta_path in Path(dlm.MODELS_DIR).glob("*_meta.json"):
        try:
            meta_json = json.load(open(meta_path, encoding="utf-8"))
            if "model_id" not in meta_json:
                # meta_path.stem == "<model_id>_meta"
                model_id = meta_path.stem.rsplit("_meta", 1)[0]
                meta_json["model_id"] = model_id
            info.append(meta_json)
        except Exception as e:
            logger.warning(f"Не удалось прочитать мета {meta_path}: {e}")
    return info

@app.post("/set")
async def set_model(model_id: str = Query(..., description="ID модели")):
    global active_model, active_tokenizer, active_model_id
    if model_id == active_model_id:
        return {"message": f"Модель {model_id} уже активна"}
    if model_id == "preloaded":
        if not Path("dl_model.keras").exists():
            raise HTTPException(404, "Файл dl_model.keras отсутствует")
        active_model = tf.keras.models.load_model("dl_model.keras", compile=False)
        active_tokenizer = dlm._load_tokenizer(Path("tokenizer.json"))
        active_model_id = "preloaded"
        return {"message": "Предзагруженная модель активирована"}
    try:
        active_model, active_tokenizer = dlm.load_by_id(model_id)
        active_model_id = model_id
        return {"message": f"Модель {model_id} активирована"}
    except FileNotFoundError:
        raise HTTPException(404, "Модель не найдена")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/one_predict", response_model=PredictionResult,
          summary="Прогноз на один релиз")
async def one_predict(data: InputData):
    if active_model is None:
        raise HTTPException(400, "Активная модель не установлена")
    target = dlm.predict(active_model, active_tokenizer, [data.text])[0]
    return PredictionResult(target=target)

@app.post("/multiple_predict", summary="Пакетный прогноз (CSV)")
async def multiple_predict(data: List[InputData]):
    if active_model is None:
        raise HTTPException(400, "Активная модель не установлена")
    texts = [item.text for item in data]
    preds = dlm.predict(active_model, active_tokenizer, texts)
    df = pd.DataFrame({"text": texts, "target": preds})
    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.post("/retrain", summary="Дообучить активную модель")
async def retrain(
    file: UploadFile = File(...),
    learning_rate: float = Query(5e-4),
    batch_size: int = Query(32),
    epochs: int = Query(5),
    early_stopping_patience: int = Query(2),
):
    if active_model is None:
        raise HTTPException(400, "Активная модель не установлена")
    df_new = pd.read_excel(io.BytesIO(await file.read()))
    res = dlm.finetune(active_model, active_tokenizer, df_new,
                       learning_rate, batch_size, epochs, early_stopping_patience)
    return {"message": "Дообучение завершено", "metrics": res["metrics"]}

@app.get("/model_info", summary="Информация об активной модели")
async def model_info():
    global active_model_id

    if active_model_id is None:
        raise HTTPException(400, "Активная модель не установлена")
    if active_model_id == "preloaded":
        metrics = {
            "accuracy": 0.850,
            "f1_weighted": 0.848,
            "auc_macro": 0.924,
            "auc_per_class": {
                "0": 0.929,
                "1": 0.869,
                "2": 0.973
            }
        }
    else:
        meta_path = Path(dlm.MODELS_DIR) / f"{active_model_id}_meta.json"
        if not meta_path.exists():
            raise HTTPException(404, f"Мета для модели {active_model_id} не найдена")
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = raw.get("metrics", raw)

    return {"model_id": active_model_id, "metrics": metrics}

@app.on_event("startup")
async def preload():
    global active_model, active_tokenizer, active_model_id
    if Path("dl_model.keras").exists():
        active_model = tf.keras.models.load_model("dl_model.keras", compile=False)
        active_tokenizer = dlm._load_tokenizer(Path("tokenizer.json"))
        active_model_id = "preloaded"
        logger.info("Предобученная модель загружена.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
