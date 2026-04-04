# Workflow-CI – Water Quality Model

Repository CI/CD untuk pelatihan otomatis model Machine Learning Water Quality menggunakan **MLflow Project** dan **GitHub Actions**.

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                              
├── MLProject/
│   ├── MLProject                               
│   ├── conda.yaml                              
│   ├── modelling.py                            
│   ├── DockerHub.txt                           
│   └── water_potability_preprocessing/         
│       ├── water_potability_train.csv
│       └── water_potability_test.csv
└── README.md
```

## 🔄 Alur Workflow CI (Advanced)

```
Push ke main
     │
     ▼
Checkout → Setup Python 3.12.7 → Check Env
     │
     ▼
Install Dependencies
     │
     ▼
Set MLflow Tracking URI (DagsHub)
     │
     ▼
Run mlflow project (train RandomForest)
     │
     ▼
Get latest MLflow run_id
     │
     ▼
Upload Artifact → Google Drive
     │
     ▼
Build Docker Model (mlflow models build-docker)
     │
     ▼
Login Docker Hub → Tag → Push Image
```

## ⚙️ Setup Secrets GitHub

Masuk ke **Settings → Secrets and variables → Actions** lalu tambahkan:

| Secret | Keterangan |
|--------|-----------|
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_REPO_NAME` | Nama repo DagsHub |
| `DAGSHUB_TOKEN` | Token akses DagsHub |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Access Token Docker Hub |
| `GDRIVE_CREDENTIALS` | Service Account JSON Google Drive |
| `GDRIVE_FOLDER_ID` | ID folder Google Drive |

## 🚀 Cara Menjalankan

### Otomatis (trigger CI)
Push perubahan ke branch `main` pada folder `MLProject/`:
```bash
git add MLProject/
git commit -m "update: retrain model"
git push origin main
```

### Manual (workflow_dispatch)
GitHub → Actions → **CI – Water Quality Model Training** → **Run workflow**
- Input `n_estimators` dan `max_depth` sesuai kebutuhan

### Lokal (tanpa CI)
```bash
cd MLProject
pip install mlflow==2.19.0 scikit-learn==1.6.0 pandas numpy matplotlib seaborn
mlflow run . --env-manager=local -P n_estimators=200 -P max_depth=10
```