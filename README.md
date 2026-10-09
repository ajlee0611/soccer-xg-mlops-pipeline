# soccer-xg-mlops-pipeline
MLOps pipeline for real-time Expected Goals (xG) calculation using ZenML, MLflow, FastAPI, and Docker

# ⚽ Soccer Expected Goals (xG) MLOps Platform

[![CI/CD Quality Gate](https://github.com/ajlee0611/soccer-xg-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ajlee0611/soccer-xg-mlops-pipeline/actions)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)
![ZenML](https://img.shields.io/badge/ZenML-Pipeline%20Orchestration-431D64)
![MLflow](https://img.shields.io/badge/MLflow-Registry%20%26%20Tracking-0194E2?logo=mlflow&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC?logo=tailwindcss&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

An end-to-end production Machine Learning pipeline and real-time inference service that calculates **Expected Goals (xG)** from StatsBomb event telemetry.

The platform orchestrates continuous data extraction, feature engineering, and model validation in **ZenML**, tracks and gates deployments using the **MLflow Model Registry**, serves predictions through a containerized **FastAPI** microservice, and provides an interactive **React + Tailwind CSS** telemetry playground for real-time spatial simulation.

---

## 🚀 Key Highlights

* **Automated MLOps Lifecycle:** End-to-end orchestration with ZenML, transitioning raw event ingestion to artifact logging in MLflow.
* **Registry Promotion Gates:** Automatic candidate promotion to the `champion` alias only after clearing strict calibration and discrimination thresholds.
* **Low-Latency Spatial Inference:** Sub-50ms REST API built on FastAPI, utilizing vectorized NumPy operations for trigonometry and distance transformations.
* **Interactive Frontend Canvas:** React client mapping user click coordinates to the official StatsBomb pitch grid ($120 \times 80$ yards).

---

## 📊 Model Quality & Validation Benchmarks

Evaluating probabilistic models requires balancing discrimination (ranking dangerous opportunities above speculative efforts) with calibration (ensuring predicted probabilities reflect observed conversion rates).

| Metric | Champion Value | Validation Threshold | Operational Meaning |
|---|---|---|---|
| **ROC-AUC** | **0.8261** | $> 0.7000$ | Discriminative rank-ordering between goals and non-goals |
| **Brier Score** | **0.0849** | $< 0.1200$ | Mean squared probabilistic error (calibration quality) |
| **Log Loss** | **0.2924** | $< 0.3500$ | Penalizes overconfident incorrect classifications |

---

## 📐 Spatial Feature Engineering

Shot telemetry coordinates are mapped onto the standard StatsBomb spatial grid: Length $X \in [0.0, 120.0]$, Width $Y \in [0.0, 80.0]$, with the attacking goal line centered at $(120.0, 40.0)$ and goalposts positioned at $(120.0, 36.0)$ and $(120.0, 44.0)$.

### 1. Euclidean Goal Distance
$$\text{Distance (yards)} = \sqrt{(120.0 - X)^2 + (40.0 - Y)^2}$$

### 2. Subtended Goal-Post Angle
Calculated by computing the angle $\theta$ between vectors $\vec{v}_1$ (shot origin to left post) and $\vec{v}_2$ (shot origin to right post):
$$\vec{v}_1 = (120.0 - X, 36.0 - Y), \quad \vec{v}_2 = (120.0 - X, 44.0 - Y)$$
$$\theta = \arccos\left( \frac{\vec{v}_1 \cdot \vec{v}_2}{\Vert{}\vec{v}_1\Vert{} \Vert{}\vec{v}_2\Vert{}} \right)$$

---

## 🛠️ Quickstart & Reproduction

### Prerequisites
* Python 3.11+
* Node.js 18+ & npm
* Docker (optional)

### 1. Environment Setup & Pipeline Execution

```bash
# Clone repository and enter directory
git clone [https://github.com/ajlee0611/soccer-xg-mlops-pipeline.git](https://github.com/ajlee0611/soccer-xg-mlops-pipeline.git)
cd soccer-xg-mlops-pipeline

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize ZenML and train the production champion model
zenml init
python -m src.pipelines.training_pipeline
```

### 2. Start the Inference Microservice

Run the microservice using either a local Python environment or a Docker container.

### Option A: Local Uvicorn (Development)

Run directly from the repository root with hot reloading enabled:

```bash
source .venv/bin/activate
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Docker Container
```bash
docker build -t soccer-xg-api:v1 .
docker run -d -p 8000:8000 --name xg-service soccer-xg-api:v1
```

### Verification
```bash
curl http://localhost:8000/health
# Returns: {"status":"healthy","model_loaded":true}
```

### 3. Launch the Interactive Frontend

In a separate terminal tab:
```bash
cd frontend
npm install --legacy-peer-deps
npm run dev -- --open
```

Navigate to http://localhost:5173 to test real-time pitch telemetry inference.
