# NetSec — AI Network Threat Detection

An end-to-end machine learning system that classifies network traffic as **normal (0)** or **threat (1)** using a full automated pipeline — from raw data ingestion to real-time prediction.

## Live demo https://mlproject-gi0k.onrender.com 

is deployed on Render's free tier, has cold start delays and training limitations. For the **best experience**, please run it locally.

---

## Features

- Full ML pipeline: Data Ingestion → Validation → Transformation → Model Training
- Trains and evaluates 9 classifiers, selects the best automatically
- Full hyperparameter tuning via `RandomizedSearchCV` (local only)
- Real-time pipeline progress tracking in the UI
- Clean, minimal web interface — no frameworks
- MLflow experiment tracking
- MongoDB Atlas as data source

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Uvicorn, Gunicorn |
| ML | scikit-learn, XGBoost, CatBoost, LightGBM |
| Tracking | MLflow, DagsHub |
| Database | MongoDB Atlas |
| Frontend | Vanilla HTML/CSS/JS |
| Deploy | Render |

---

## Local Setup (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/RF-Rezon/NetSec---ML__Application.git
cd NetSec---ML__Application
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
MONGO_DB_URL=your_mongodb_atlas_connection_string
```

### 5. Run the app

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## Usage

### Step 1 — Train the model
- Go to the **App** page
- Click **Start Training**
- Watch the pipeline steps complete in real time
- The best model name will appear when training is done

### Step 2 — Predict
- Upload a CSV file with network traffic data
- Click **Run Prediction**
- See threat rate, normal vs threat breakdown, and a verdict
- Download the results as CSV

---

## Local vs Render

| | Local | Render (Free Tier) |
|---|---|---|
| Hyperparameter tuning | Full `RandomizedSearchCV` | Disabled (fast fit) |
| Training time | 2–3 minutes | Immediate |
| Model accuracy | Higher | Slightly lower |
| Cold start | None | ~30 seconds |

> Render's free tier spins down after inactivity. The first request may take 30–60 seconds to respond.

## License

MIT
