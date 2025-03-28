import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==================== Load Training Data ====================
train_df = pd.read_csv('data/train.csv')
train_df.dropna(subset=['Transported'], inplace=True)

# ==================== Fill Missing Values ====================
fill_values = {
    'HomePlanet': 'Unknown',
    'CryoSleep': False,
    'Cabin': 'Unknown/0/Unknown',
    'Destination': 'Unknown',
    'Age': train_df['Age'].median(),
    'VIP': False,
    'RoomService': 0.0,
    'FoodCourt': 0.0,
    'ShoppingMall': 0.0,
    'Spa': 0.0,
    'VRDeck': 0.0,
    'Name': 'Unknown'
}
train_df.fillna(fill_values, inplace=True)

# ==================== Feature Engineering ====================
# Split 'Cabin' into separate features
cabin_split = train_df['Cabin'].str.split('/', expand=True)
train_df['Cabin_Deck'] = cabin_split[0]
train_df['Cabin_Num'] = pd.to_numeric(cabin_split[1], errors='coerce').fillna(0)
train_df['Cabin_Side'] = cabin_split[2]
train_df.drop(columns='Cabin', inplace=True)

# Spending & social features
train_df['TotalSpending'] = train_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
train_df['HasLuxuryService'] = (train_df[['Spa', 'VRDeck', 'RoomService']] > 0).any(axis=1).astype(int)
train_df['LuxuryCount'] = (train_df[['Spa', 'VRDeck', 'RoomService']] > 0).sum(axis=1)
train_df['GroupSize'] = train_df.groupby('Cabin_Num')['Cabin_Num'].transform('count')
train_df['IsAlone'] = (train_df['GroupSize'] == 1).astype(int)

# ==================== Encode Categorical Columns ====================
label_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Name']
encoders = {}
encoded_df = train_df.copy()

for col in label_cols:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
    encoders[col] = le

# ==================== Prepare Feature Matrix ====================
features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Cabin_Deck', 'Cabin_Num', 'Cabin_Side', 'Name',
    'TotalSpending', 'HasLuxuryService', 'LuxuryCount', 'GroupSize', 'IsAlone'
]

X_lgb = encoded_df[features]
X_cat = train_df[features]
y = train_df['Transported'].astype(int)

# Scale numeric features for LightGBM
numeric_cols = [
    'Age', 'Cabin_Num', 'RoomService', 'FoodCourt', 'ShoppingMall',
    'Spa', 'VRDeck', 'TotalSpending', 'LuxuryCount', 'GroupSize'
]
scaler = RobustScaler()
X_lgb[numeric_cols] = scaler.fit_transform(X_lgb[numeric_cols])

# ==================== Train/Test Split ====================
X_lgb_train, X_lgb_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_lgb, X_cat, y, test_size=0.2, stratify=y, random_state=42
)

# ==================== Train LightGBM ====================
lgbm_model = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.015,
    num_leaves=64,
    max_depth=10,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42
)
lgbm_model.fit(X_lgb_train, y_train)
lgb_preds = lgbm_model.predict_proba(X_lgb_test)[:, 1]

# ==================== Train CatBoost ====================
cat_features_idx = [features.index(col) for col in label_cols]

catboost_model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=0
)
catboost_model.fit(X_cat_train, y_train, cat_features=cat_features_idx)
cat_preds = catboost_model.predict_proba(X_cat_test)[:, 1]

# ==================== Ensemble Predictions ====================
ensemble_probs = (lgb_preds + cat_preds) / 2
ensemble_preds = (ensemble_probs > 0.5).astype(int)

# ==================== Accuracy ====================
print(f"LightGBM Accuracy:  {accuracy_score(y_test, (lgb_preds > 0.5)):.4f}")
print(f"CatBoost Accuracy:  {accuracy_score(y_test, (cat_preds > 0.5)):.4f}")
print(f"Ensemble Accuracy:  {accuracy_score(y_test, ensemble_preds):.4f}")

# ==================== Process Test Set ====================
test_df = pd.read_csv('data/test.csv')
test_df.fillna(fill_values, inplace=True)

# Cabin split
cabin_split = test_df['Cabin'].str.split('/', expand=True)
test_df['Cabin_Deck'] = cabin_split[0].fillna('Unknown')
test_df['Cabin_Num'] = pd.to_numeric(cabin_split[1], errors='coerce').fillna(0)
test_df['Cabin_Side'] = cabin_split[2].fillna('Unknown')
test_df.drop(columns='Cabin', inplace=True)

# Test feature engineering
test_df['TotalSpending'] = test_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
test_df['HasLuxuryService'] = (test_df[['Spa', 'VRDeck', 'RoomService']] > 0).any(axis=1).astype(int)
test_df['LuxuryCount'] = (test_df[['Spa', 'VRDeck', 'RoomService']] > 0).sum(axis=1)
test_df['GroupSize'] = test_df.groupby('Cabin_Num')['Cabin_Num'].transform('count')
test_df['IsAlone'] = (test_df['GroupSize'] == 1).astype(int)

# Encode test set
test_encoded = test_df.copy()
for col in label_cols:
    le = encoders[col]
    test_encoded[col] = test_encoded[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

X_lgb_final = test_encoded[features]
X_lgb_final[numeric_cols] = scaler.transform(X_lgb_final[numeric_cols])
X_cat_final = test_df[features]

# Predict on test set
lgb_test_preds = lgbm_model.predict_proba(X_lgb_final)[:, 1]
cat_test_preds = catboost_model.predict_proba(X_cat_final)[:, 1]
ensemble_test_preds = (lgb_test_preds + cat_test_preds) / 2 > 0.5

# ==================== Save Submission ====================
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': ensemble_test_preds.astype(bool)
})
submission.to_csv('output/ensemble_submission.csv', index=False)
