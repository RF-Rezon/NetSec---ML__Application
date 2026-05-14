import sys
import certifi
import traceback

ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

# @app.get("/index.html")
# async def index():
#     return FileResponse("static/index.html")

@app.get("/app.html")
async def app_page():
    return FileResponse("static/app.html")

    
@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        # best model name বের করো
        model = load_object("final_model/model.pkl")
        model_name = type(model).__name__
        
        return {"message": "Training successful", "best_model": model_name}
    except Exception as e:
        raise NetworkSecurityException(e, sys)

    
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)

        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)

        y_pred = network_model.predict(df)

        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        
        df.to_csv('prediction_output/output.csv')

        table_html = df.to_html(classes='table table-striped')
        
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

        
    except Exception:
        return Response(traceback.format_exc(), status_code=500)
    



@app.get("/docs", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


    
if __name__=="__main__":
    app_run(app,host="0.0.0.0", port=5000)
    
