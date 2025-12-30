# ✈️ Travel ML - MLOps Capstone Project

A comprehensive MLOps project for travel data analytics, featuring machine learning models for flight price prediction, gender classification, and hotel recommendations, with full CI/CD pipeline, containerization, and orchestration.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [MLflow Tracking](#mlflow-tracking)
- [Airflow Workflows](#airflow-workflows)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)

## 🎯 Project Overview

| Component | Description |
|-----------|-------------|
| **Regression Model** | Predict flight prices based on route, class, and timing |
| **Classification Model** | Classify user gender from travel patterns |
| **Recommendation Model** | Suggest hotels based on user preferences |
| **REST API** | Flask-based API serving all models |
| **Containerization** | Docker images for all services |
| **Orchestration** | Kubernetes deployment for scalability |
| **Workflow Automation** | Apache Airflow DAGs for data pipelines |
| **CI/CD** | Jenkins pipeline for automated deployment |
| **Experiment Tracking** | MLflow for model versioning |
| **Dashboard** | Streamlit app for visualization |

## 📁 Project Structure

```
mlops_travel_project/
├── data/                       # Datasets
│   ├── users.csv
│   ├── flights.csv
│   ├── hotels.csv
│   └── generate_data.py
├── models/                     # ML Models
│   ├── flight_price_model.py
│   ├── gender_classifier.py
│   └── hotel_recommender.py
├── api/                        # Flask REST API
│   └── app.py
├── docker/                     # Docker Configuration
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
├── kubernetes/                 # K8s Manifests
│   └── deployment.yaml
├── airflow/                    # Airflow DAGs
│   └── dags/
│       └── travel_pipeline_dag.py
├── jenkins/                    # Jenkins Pipeline
│   └── Jenkinsfile
├── mlflow/                     # MLflow Tracking
│   └── experiment_tracker.py
├── streamlit/                  # Dashboard
│   └── app.py
├── tests/                      # Unit Tests
│   └── test_models.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Extract the zip file and navigate to project
cd mlops_travel_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models
cd models
python flight_price_model.py
python gender_classifier.py
python hotel_recommender.py
cd ..

# 5. Start API
cd api && python app.py
# API runs at http://localhost:5001

# 6. Start Dashboard (new terminal)
cd streamlit && streamlit run app.py
# Dashboard at http://localhost:8501
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- Docker Desktop (for containerization)
- Kubernetes - minikube/kind (for K8s deployment)
- Jenkins (for CI/CD)

### VS Code Extensions (Recommended)
- Python
- Jupyter
- Docker
- Kubernetes
- YAML
- GitLens

## 📡 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/api/v1/predict/flight-price` | POST | Predict flight price |
| `/api/v1/predict/gender` | POST | Classify gender |
| `/api/v1/recommend/hotels` | POST | Get hotel recommendations |

### Example: Flight Price Prediction
```bash
curl -X POST http://localhost:5001/api/v1/predict/flight-price \
  -H "Content-Type: application/json" \
  -d '{
    "from": "New York",
    "to": "Los Angeles",
    "flightType": "business",
    "distance": 2800,
    "time": 5.5
  }'
```

### Example: Hotel Recommendations
```bash
curl -X POST http://localhost:5001/api/v1/recommend/hotels \
  -H "Content-Type: application/json" \
  -d '{
    "user_code": 100,
    "method": "hybrid",
    "n_recommendations": 5
  }'
```

## 🐳 Docker Deployment

```bash
cd docker

# Build and start all services
docker-compose up -d

# Services:
# - ML API: http://localhost:5001
# - MLflow: http://localhost:5001
# - Streamlit: http://localhost:8501
# - Airflow: http://localhost:8080

# Stop services
docker-compose down
```

## ☸️ Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f kubernetes/deployment.yaml

# Check deployment
kubectl get pods -n travel-ml
kubectl get services -n travel-ml

# Port forward
kubectl port-forward svc/travel-ml-api-service 5000:80 -n travel-ml
```

## 📊 MLflow Tracking

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5001

# Access UI at http://localhost:5001
```

## 🔄 Airflow Workflows

```bash
# Initialize Airflow
airflow db init

# Create admin user
airflow users create --username admin --password admin \
    --firstname Admin --lastname User --role Admin --email admin@example.com

# Start webserver (terminal 1)
airflow webserver --port 8080

# Start scheduler (terminal 2)
airflow scheduler

# Access at http://localhost:8080
```

## 🔧 CI/CD Pipeline (Jenkins)

The Jenkinsfile includes stages for:
1. Checkout code
2. Setup environment
3. Lint & code quality
4. Unit tests
5. Train models
6. Build Docker image
7. Security scan
8. Integration tests
9. Deploy to staging
10. Deploy to production (manual approval)

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=models --cov=api --cov-report=html
```

## 📈 Model Performance

### Flight Price Prediction
| Model | RMSE | R² |
|-------|------|-----|
| Random Forest | ~$120 | 0.85 |
| Gradient Boosting | ~$125 | 0.83 |

### Gender Classification
| Model | Accuracy | F1 |
|-------|----------|-----|
| Random Forest | 0.55 | 0.55 |

## 🗂️ Datasets

- **users.csv**: 500 users with demographics
- **flights.csv**: 2000 flight bookings
- **hotels.csv**: 1500 hotel reservations

---

**Built with ❤️ for MLOps learning**
