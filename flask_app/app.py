# # # from flask import Flask, request, render_template
# # # import pandas as pd 
# # # import joblib 
# # # import numpy 

# # # app = Flask(__name__)

# # # # then we load our modl 
# # # model = joblib.load("model/wine_quality_model.joblib")
# # # scaler = joblib.load("model/wine_quality_scaler.joblib")

# # # feature_names = [
# # #         "fixed acidity",
# # #         "volatile acidity",
# # #         "citric acid",
# # #         "residual sugar",
# # #         "chlorides",
# # #         "free sulfur dioxide",
# # #         "total sulfur dioxide",
# # #         "pH",              # 🔥 KEEP pH (NO rename)
# # #         "sulphates",
# # #         "alcohol"
# # # ]


# # # @app.route("/", methods=["GET", "POST"])
# # # def index():
# # #     prediction = None

# # #     if request.method == "POST":
# # #         values = [float(request.form[f]) for f in feature_names]
# # #         input_df = pd.DataFrame([values], columns=feature_names)
# # #         scaled_input = scaler.transform(input_df)
# # #         prediction = round(model.predict(scaled_input)[0], 2)

# # #     return render_template("index.html", prediction=prediction)

# # # if __name__ == "__main__":
# # #     app.run(host="0.0.0.0", port=5000)

# # from flask import Flask, render_template, request
# # import joblib
# # import pandas as pd

# # app = Flask(__name__)

# # # Load model & scaler
# # model = joblib.load("model/wine_quality_model.joblib")
# # scaler = joblib.load("model/wine_quality_scaler.joblib")

# # # Feature order (must match training)
# # FEATURES = [
# #     "fixed acidity",
# #     "volatile acidity",
# #     "citric acid",
# #     "residual sugar",
# #     "chlorides",
# #     "free sulfur dioxide",
# #     "total sulfur dioxide",
# #     "pH",
# #     "sulphates",
# #     "alcohol"
# # ]

# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     prediction = None
# #     batch_result = None

# #     # ---------- SINGLE PREDICTION ----------
# #     if request.method == "POST" and "single_predict" in request.form:
# #         values = [float(request.form[f]) for f in FEATURES]
# #         input_df = pd.DataFrame([values], columns=FEATURES)
# #         scaled_input = scaler.transform(input_df)
# #         prediction = round(model.predict(scaled_input)[0], 2)

# #     # ---------- BATCH PREDICTION ----------
# #     if request.method == "POST" and "batch_predict" in request.form:
# #         file = request.files["file"]
# #         if file:
# #             df = pd.read_csv(file)

# #             # Ensure correct columns
# #             df = df[FEATURES]

# #             scaled_data = scaler.transform(df)
# #             preds = model.predict(scaled_data)

# #             df["Predicted Quality"] = preds.round(2)
# #             batch_result = df.to_dict(orient="records")

# #     return render_template(
# #         "index.html",
# #         prediction=prediction,
# #         batch_result=batch_result,
# #         features=FEATURES
# #     )

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=5000)

# from flask import Flask, render_template, request
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load model and scaler
# model = joblib.load("model/wine_quality_model.joblib")
# scaler = joblib.load("model/wine_quality_scaler.joblib")

# FEATURES = [
#     "fixed acidity",
#     "volatile acidity",
#     "citric acid",
#     "residual sugar",
#     "chlorides",
#     "free sulfur dioxide",
#     "total sulfur dioxide",
#     "pH",
#     "sulphates",
#     "alcohol"
# ]

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     batch_result = None

#     # -------- GET (like @app.get("/")) --------
#     if request.method == "GET":
#         return render_template(
#             "index.html",
#             prediction=None,
#             batch_result=None,
#             features=FEATURES
#         )

#     # -------- POST (like @app.post("/predict")) --------
#     if request.method == "POST":

#         # SINGLE PREDICTION (FastAPI /predict equivalent)
#         if "single_predict" in request.form:
#             values = [float(request.form[f]) for f in FEATURES]
#             input_df = pd.DataFrame([values], columns=FEATURES)
#             scaled = scaler.transform(input_df)
#             prediction = round(model.predict(scaled)[0], 2)

#         # BATCH PREDICTION (FastAPI /batch equivalent)
#         elif "batch_predict" in request.form:
#             file = request.files.get("file")
#             if file:
#                 df = pd.read_csv(file)
#                 df = df[FEATURES]
#                 scaled = scaler.transform(df)
#                 df["Predicted Quality"] = model.predict(scaled).round(2)
#                 batch_result = df.to_dict(orient="records")

#     return render_template(
#         "index.html",
#         prediction=prediction,
#         batch_result=batch_result,
#         features=FEATURES
#     )

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


# important

# from flask import Flask, request, render_template
# import joblib
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # =========================
# # LOAD MODEL & SCALER
# # =========================
# model = joblib.load("model/wine_quality_model.joblib")
# scaler = joblib.load("model/wine_quality_scaler.joblib")

# # =========================
# # HOME PAGE  (FastAPI: GET /)
# # =========================
# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html")

# # =========================
# # SINGLE PREDICTION (FastAPI: POST /predict)
# # =========================
# @app.route("/predict", methods=["POST"])
# def predict():

#     input_data = np.array([[  
#         float(request.form["fixed_acidity"]),
#         float(request.form["volatile_acidity"]),
#         float(request.form["citric_acid"]),
#         float(request.form["residual_sugar"]),
#         float(request.form["chlorides"]),
#         float(request.form["free_sulfur_dioxide"]),
#         float(request.form["total_sulfur_dioxide"]),
#         float(request.form["ph"]),          # HTML uses "ph"
#         float(request.form["sulphates"]),
#         float(request.form["alcohol"])
#     ]])

#     scaled_input = scaler.transform(input_data)
#     prediction = model.predict(scaled_input)[0]

#     return render_template(
#         "index.html",
#         prediction=round(float(prediction), 2)
#     )

# # =========================
# # BATCH PREDICTION (FastAPI: POST /batch-predict)
# # =========================
# @app.route("/batch-predict", methods=["POST"])
# def batch_predict():

#     file = request.files["file"]
#     df = pd.read_csv(file)

#     required_columns = [
#         "fixed acidity",
#         "volatile acidity",
#         "citric acid",
#         "residual sugar",
#         "chlorides",
#         "free sulfur dioxide",
#         "total sulfur dioxide",
#         "pH",              # 🔥 KEEP pH
#         "sulphates",
#         "alcohol"
#     ]

#     X_batch = df[required_columns]
#     X_scaled = scaler.transform(X_batch)

#     df["Predicted Quality"] = model.predict(X_scaled).round(2)

#     return render_template(
#         "index.html",
#         batch_result=df.to_dict(orient="records")
#     )

# # =========================
# # RUN FLASK APP
# # =========================
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=True)
    

from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD MODEL & SCALER
# =========================
model = joblib.load("model/wine_quality_model.joblib")
scaler = joblib.load("model/wine_quality_scaler.joblib")


# =========================
# HOME PAGE
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# =========================
# SINGLE PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    input_data = np.array([[  
        float(request.form["fixed_acidity"]),
        float(request.form["volatile_acidity"]),
        float(request.form["citric_acid"]),
        float(request.form["residual_sugar"]),
        float(request.form["chlorides"]),
        float(request.form["free_sulfur_dioxide"]),
        float(request.form["total_sulfur_dioxide"]),
        float(request.form["ph"]),
        float(request.form["sulphates"]),
        float(request.form["alcohol"])
    ]])

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    prediction = round(float(prediction), 2)

    # QUALITY LABEL
    if prediction >= 7:
        label = "High Quality 🍷"
    elif prediction >= 5:
        label = "Medium Quality"
    else:
        label = "Low Quality"

    return render_template(
        "index.html",
        prediction=prediction,
        label=label
    )


# =========================
# BATCH PREDICTION
# =========================
@app.route("/batch-predict", methods=["POST"])
def batch_predict():

    file = request.files["file"]

    df = pd.read_csv(file)

    required_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "pH",
        "sulphates",
        "alcohol"
    ]

    X_batch = df[required_columns]

    X_scaled = scaler.transform(X_batch)

    df["Predicted Quality"] = model.predict(X_scaled).round(2)

    return render_template(
        "index.html",
        batch_result=df.to_dict(orient="records")
    )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)