# 🚀 Spaceship Survival Prediction

This project uses machine learning models (**LightGBM** and **CatBoost**) to predict whether passengers on the Spaceship Titanic were transported to an alternate dimension. It includes data preprocessing, feature engineering, model training, and an ensemble prediction pipeline.

---

## 🧠 Models Used

- **LightGBM** – Gradient Boosting Decision Trees, optimized for class imbalance and performance.
- **CatBoost** – Efficient boosting model that handles categorical features natively.
- **Ensemble** – Soft voting strategy to combine predictions from both models for better accuracy.

---

## 🛠️ Features Engineered

- **Cabin Splitting**: Extracted `Cabin_Deck`, `Cabin_Num`, and `Cabin_Side` from the `Cabin` column.
- **Spending Features**: Total spending and luxury usage indicators (`TotalSpending`, `HasLuxuryService`, `LuxuryCount`).
- **Group Features**: `GroupSize` based on shared cabin number and `IsAlone` flag.
- **Categorical Encoding**: Used `LabelEncoder` for all categorical features.

---
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/spaceship-survival-prediction.git
   cd spaceship-survival-prediction
2. Install requirements:

   ```bash
    pip install -r requirements.txt
3. Add your train.csv and test.csv to the data/ folder.

4. Run the pipeline:
    ```bash 
   python main.py
   
5. Check output/ensemble_submission.csv for the final predictions.

# 📌 Notes
    Missing values are handled with sensible defaults or imputation.
    
    The ensemble generally performs better than either model individually.
    
    RobustScaler is used to scale numeric features for LightGBM (CatBoost handles raw data better).

# 📄 License
    This project is released under the MIT License.