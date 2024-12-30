from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Annotated
import pandas as pd
import joblib
import uvicorn
from fastapi.responses import StreamingResponse
import io

app = FastAPI(title="API для предсказания решений об изменении ключевой ставки ЦБ")

class InputData(BaseModel):
    text: Annotated[str, Field(description="Текст пресс-релиза")]

class PredictionResult(BaseModel):
    target: Annotated[int, Field(description="Предсказанное изменение ставки (-1: понижение, 0: без изменений, 1: повышение)")]

async def load_and_preprocess():
    try:
        model = joblib.load("best_model.joblib")
        preprocessor = joblib.load("N_GRAM_BOW_TRANSFORMER.joblib")
        return model, preprocessor
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка при загрузке: {str(e)}")

@app.post("/one_predict", response_model=PredictionResult, summary="Предсказание для одного объекта", tags=["Prediction"],
         responses={status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": List[dict]}})
async def predict_one_rate(
        data: InputData,
        model_preprocessor: Annotated[tuple, Depends(load_and_preprocess)]
) -> PredictionResult:
    model, preprocessor = model_preprocessor
    try:
        row_data = pd.DataFrame([{"text": data.text}])
        processed_data = preprocessor.transform(row_data)
        prediction = int(model.predict(processed_data)[0])
        return PredictionResult(target=prediction)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/multiple_predict", summary="Предсказания для нескольких объектов", tags=["Prediction"])
async def predict_multiple_rates(
        data: List[InputData],
        model_preprocessor: Annotated[tuple, Depends(load_and_preprocess)]
) -> StreamingResponse:
    model, preprocessor = model_preprocessor
    results = []
    for item in data:
        df = pd.DataFrame([{"text": item.text}])
        processed_item = preprocessor.transform(df)
        prediction = int(model.predict(processed_item)[0])
        results.append({"text": item.text, "target": prediction})

    results_df = pd.DataFrame(results)

    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(io.BytesIO(csv_buffer.getvalue().encode()), media_type='text/csv', headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.get("/model_info", summary="Информация о модели", tags=["Information"])
async def model_info(
        model_preprocessor: Annotated[tuple, Depends(load_and_preprocess)]
) -> dict:
    model, _ = model_preprocessor
    return {"model_name": model.__class__.__name__, "model_version": "1.0"}

if __name__ == "__main__":
    uvicorn.run('main:app', port=8000, reload=True)
