import os
import sys
import certifi
import traceback
import threading

ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response, FileResponse
from starlette.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Training status tracker ──────────────────────────────────────────────────
training_status = {
    "status": "idle",
    "current_step": 0,
    "steps": {1: "idle", 2: "idle", 3: "idle", 4: "idle"},
    "best_model": None,
    "error": None
}

def run_training():
    global training_status
    try:
        training_status = {
            "status": "running",
            "current_step": 1,
            "steps": {1: "running", 2: "idle", 3: "idle", 4: "idle"},
            "best_model": None,
            "error": None
        }

        pipeline = TrainingPipeline()

        # Step 1: Data Ingestion
        data_ingestion_artifact = pipeline.start_data_ingestion()
        training_status["steps"][1] = "done"
        training_status["steps"][2] = "running"
        training_status["current_step"] = 2

        # Step 2: Data Validation
        data_validation_artifact = pipeline.start_data_validation(
            data_ingestion_artifact=data_ingestion_artifact
        )
        training_status["steps"][2] = "done"
        training_status["steps"][3] = "running"
        training_status["current_step"] = 3

        # Step 3: Data Transformation
        data_transformation_artifact = pipeline.start_data_transformation(
            data_validation_artifact=data_validation_artifact
        )
        training_status["steps"][3] = "done"
        training_status["steps"][4] = "running"
        training_status["current_step"] = 4

        # Step 4: Model Trainer
        pipeline.start_model_trainer(
            data_transformation_artifact=data_transformation_artifact
        )
        training_status["steps"][4] = "done"

        model = load_object("final_model/model.pkl")
        model_name = type(model).__name__
        training_status["status"] = "done"
        training_status["best_model"] = model_name

    except Exception as e:
        training_status["status"] = "error"
        training_status["error"] = str(e)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/app.html")
async def app_page():
    return FileResponse("static/app.html")


@app.get("/train")
async def train_route():
    if training_status["status"] == "running":
        return {"message": "Already running"}
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    return {"message": "Training started"}


@app.get("/train/status")
async def train_status():
    return training_status


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv')

        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception:
        return Response(traceback.format_exc(), status_code=500)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=5000)   # Render
    # app_run(app, host="127.0.0.1", port=5000)    # Localhost
