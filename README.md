# CNC Machine Predictive Maintenance MLOps Project

A complete end-to-end MLOps pipeline for predicting CNC machine maintenance needs.
It uses a synthetic industrial dataset of sensor readings and applies a Multi-Layer Perceptron (MLP) classifier.
The project demonstrates best practices in data handling, model tracking, deployment, and CI/CD.

---

## 🔍 Project Overview

- **Objective**: Classify CNC machines into three maintenance risk levels (Normal, Maintenance Soon, Immediate).
- **Dataset**: 5,000 samples of CNC sensor features + a multiclass `Maintenance_Needed` label.
- **Model**: Scikit-learn `MLPClassifier` with layers (128, 64, 32).
- **Tracking**: MLflow logs hyperparameters, metrics, artifacts (confusion matrix, classification report), and model versions.
- **Deployment**: Docker container that runs both MLflow UI (port 5001) and Streamlit web app (port 8501).
- **CI/CD**: GitHub Actions workflow automates training, linting, Docker build/push, and Streamlit Cloud deployment.

---

## 📂 Repository Structure

```
cnc-maintenance-mlops/
├── .github/
│   └── workflows/ci_cd.yml           # GitHub Actions pipelines
├── app/
│   ├── streamlit_app.py             # Streamlit UI code
│   └── mlruns/                      # (optional) MLflow logs
├── data/
│   └── cnc_machine_data_multiclass.csv  # Input dataset
├── model/
│   └── mlp_cnc_model.pkl            # Pickled model + scaler
├── src/
│   └── train.py                     # Training script with MLflow integration
├── Dockerfile                       # Containerizes MLflow UI + Streamlit
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

1. **Clone repository**
   ```bash
   git clone https://github.com/<YourUser>/cnc-maintenance-mlops.git
   cd cnc-maintenance-mlops
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model locally**
   ```bash
   python src/train.py
   ```
   - This logs runs to `mlruns/` and saves `model/mlp_cnc_model.pkl`.

4. **Run MLflow UI**
   ```bash
   mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5001
   ```
   Open `http://localhost:5001` to inspect experiments.

5. **Launch Streamlit app**
   ```bash
   streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
   ```
   Visit `http://localhost:8501` for the maintenance prediction UI.

6. **Docker (all-in-one)**
   ```bash
   docker build -t cnc-maintenance-app .
   docker run -d -v $(pwd)/mlruns:/app/mlruns -p 7000:5001 -p 8501:8501 cnc-maintenance-app
   ```
   - MLflow UI: `http://localhost:7000`  
   - Streamlit: `http://localhost:8501`

7. **CI/CD**
   - On push to `main`, GitHub Actions will run training, linting, build & push Docker image, then deploy to Streamlit Cloud.

---

## 🎯 Key Benefits

- **Reduced Downtime**: Proactive maintenance classification prevents unexpected failures.
- **Reproducibility**: MLflow tracking and Docker guarantee consistent runs across environments.
- **Automated Deployment**: CI/CD pipeline ensures every code change is tested and deployed.

---

## 🔗 Resources

- [MLflow Documentation](https://mlflow.org/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [GitHub Actions](https://docs.github.com/actions)

---

_Made with ❤️ by [Your Name]_

