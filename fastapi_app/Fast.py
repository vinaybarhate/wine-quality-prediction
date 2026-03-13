from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
import uvicorn

# =========================
# INITIALIZE APP
# =========================
app = FastAPI()

templates = Jinja2Templates(directory="templates")

# =========================
# LOAD MODEL & SCALER
# =========================
model = joblib.load("model/wine_quality_model.joblib")
scaler = joblib.load("model/wine_quality_scaler.joblib")

# =========================
# HOME PAGE
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# =========================
# SINGLE PREDICTION
# =========================
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    ph: float = Form(...),              # HTML uses "ph"
    sulphates: float = Form(...),
    alcohol: float = Form(...)
):
    # NumPy array → no feature names → safe
    input_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        ph,
        sulphates,
        alcohol
    ]])

    scaled_input = scaler.transform(input_data)
    prediction = float(model.predict(scaled_input)[0])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction, 2)
        }
    )

# =========================
# BATCH PREDICTION (CSV)
# =========================
@app.post("/batch-predict", response_class=HTMLResponse)
def batch_predict(
    request: Request,
    file: UploadFile = File(...)
):
    # Read CSV
    df = pd.read_csv(file.file)

    # required column same as use at training time 
    required_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "pH",              # 🔥 KEEP pH (NO rename)
        "sulphates",
        "alcohol"
    ]

    # Select columns in correct order
    X_batch = df[required_columns]

    # Scale and predict
    X_scaled = scaler.transform(X_batch)
    predictions = model.predict(X_scaled)

    # Add prediction column
    df["Predicted Quality"] = predictions.round(2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "batch_result": df.to_dict(orient="records")
        }
    )


# when we use below lines then we also run the file using 
# python app.py then local page comes http://0.0.0.0:8000 then edit it 127.0.0.1.8000 then it runs or works

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port=8000)

#  the run command for these file is 
# or python Fast.py
# uvicorn Fast:app --reload
