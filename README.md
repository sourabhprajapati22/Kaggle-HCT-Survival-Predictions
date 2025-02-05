# Kaggle-HCT-Survival-Predictions
CIBMTR - Equity in post-HCT Survival Predictions


#### steps

1. Download [VsCode](https://code.visualstudio.com/download)
2. Download [GitHub](https://git-scm.com/downloads)
3. Open [github](https://github.com/)
4. Create Repository [Kaggle-HCT-Survival-Predictions](https://github.com/sourabhprajapati22/Kaggle-HCT-Survival-Predictions)
5. Clone Repository
    ```
    git clone https://github.com/sourabhprajapati22/Kaggle-HCT-Survival-Predictions.git
    ```
6. Change directory
    ```
    cd Kaggle-HCT-Survival-Predictions
    ```
7. Create virtual environment (Differences Between --name and --prefix)
    **Method** | **Stores Env In** | **Activation Command** | **Use Case**  
    --- | --- | --- | ---  
    `--name` | `~/anaconda3/envs/env_name` | `conda activate env_name` | Shared environments  
    `--prefix` | `./env` (or any specified path) | `conda activate ./env` | Project-specific environments  

    ```
    conda create --prefix ./venv python==3.8 -y
    conda activate ./venv
    ```

8. Create template.py
    ```
        📂 Kaggle-HCT-Survival-Predictions/
    │── 📂 data/                  # Store dataset (if allowed)
    │   ├── 📂 raw/               # Unprocessed original data
    │   ├── 📂 processed/         # Cleaned/engineered data
    │── 📂 notebooks/             # Jupyter notebooks for EDA & modeling
    │   ├── 01_eda.ipynb          # Exploratory Data Analysis
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_model_training.ipynb
    │   ├── 04_evaluation.ipynb
    │── 📂 src/                   # Source code for ML pipeline
    │   ├── data_preprocessing.py # Data cleaning & feature engineering
    │   ├── train.py              # Training script
    │   ├── inference.py          # Inference script
    │── 📂 models/                # Saved models
    │   ├── best_model.pth        # Best trained model
    │   ├── experiment_1.pkl
    │── 📂 scripts/               # Bash scripts for automation
    │   ├── run_train.sh
    │── 📂 logs/                  # Training logs & experiment tracking
    │── 📂 reports/               # Results & analysis
    │── 📂 configs/               # Config files (YAML/JSON)
    │── 📂 submission/            # Kaggle submission files
    │   ├── submission.csv
    │── .gitignore                # Ignore unnecessary files
    │── README.md                 # Project documentation
    │── requirements.txt          # Python dependencies
    │── LICENSE                   # Open-source license (e.g., MIT)
    ```

9.