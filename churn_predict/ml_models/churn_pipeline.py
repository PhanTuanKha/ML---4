"""
churn_pipeline.py

Pipeline for Telco Customer Churn classification:
- Load and preprocess data
- Train: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Evaluate: Accuracy, AUC-ROC, Precision, Recall, F1-score, Confusion Matrix
- Predictions: 3 sample predictions per model (first 2 from test set, 1 manual)
- Explain models using SHAP and LIME
- Save trained models with pickle

Put your dataset CSV at 'dataset/Telco-Customer-Churn.csv' or change DATA_PATH.

Run:
    python churn_pipeline.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import metrics
from collections import Counter
from sklearn.utils import resample

# Optional explainability libs
try:
    import shap
except Exception as e:
    shap = None
    print("SHAP not available:", e)
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception as e:
    LimeTabularExplainer = None
    print("LIME not available:", e)

# ---------- Config ----------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "..", "dataset", "Telco-Customer-Churn.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "../output_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------

def load_and_clean(path):
    df = pd.read_csv(path)
    # Clean column names (lower-case)
    df.columns = [c.strip() for c in df.columns]

    # Convert TotalCharges to numeric (some rows may be blank)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing TotalCharges for customers with tenure 0 -> set to 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Map binary 'Yes'/'No' to 1/0 when appropriate (we'll let ColumnTransformer handle categorical)
    # Ensure target exists:
    if 'Churn' in df.columns:
        df.rename(columns={'Churn': 'churn'}, inplace=True)
        # Map target to 0/1
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    else:
        raise ValueError("Target column 'Churn' not found in dataset (case-insensitive).")

    return df

def build_preprocessor(df):
    # Separate numeric and categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove target
    if 'churn' in numeric_features:
        numeric_features.remove('churn')

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # ColumnTransformer: numeric -> impute+scale, categorical -> impute + onehot
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features

def handle_imbalance(X_train, y_train):
    """Under-sampling majority class"""
    train_df = X_train.copy()
    train_df['churn'] = y_train
    majority = train_df[train_df.churn == 0]
    minority = train_df[train_df.churn == 1]

    print(f"Before resampling: {Counter(y_train)}")

    majority_under = resample(majority,
                              replace=False,
                              n_samples=len(minority),
                              random_state=RANDOM_STATE)
    train_balanced = pd.concat([majority_under, minority])
    train_balanced = train_balanced.sample(frac=1, random_state=RANDOM_STATE)

    y_train_bal = train_balanced['churn']
    X_train_bal = train_balanced.drop(columns=['churn'])
    print(f"After under-sampling: {Counter(y_train_bal)}")
    print("Resampled training set shape:", X_train_bal.shape)
    return X_train_bal, y_train_bal

def fit_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        # Có trọng số tự cân bằng
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'),
        # Không có class_weight → dùng under-sampling
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
    }

    results = {}

    for name, clf in models.items():
        print("\n" + "="*60)
        print("Training model:", name)

        # Nếu là GradientBoosting → dùng tập cân bằng
        if name == 'GradientBoosting':
            X_train_bal, y_train_bal = handle_imbalance(X_train, y_train)
            X_used, y_used = X_train_bal, y_train_bal
        else:
            X_used, y_used = X_train, y_train

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', clf)])
        pipeline.fit(X_used, y_used)

        # Save model
        model_path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Saved model to {model_path}")

        # Predict
        y_pred = pipeline.predict(X_test)
        # predict_proba for AUC (some classifiers may not support; handle)
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            # fallback: use decision_function then map to sigmoid
            try:
                df_dec = pipeline.decision_function(X_test)
                y_proba = 1 / (1 + np.exp(-df_dec))
            except Exception:
                y_proba = None

        # Metrics
        acc = metrics.accuracy_score(y_test, y_pred)
        prec = metrics.precision_score(y_test, y_pred, zero_division=0)
        rec = metrics.recall_score(y_test, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
        if y_proba is not None:
            auc = metrics.roc_auc_score(y_test, y_proba)
        else:
            auc = None

        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        if auc is not None:
            print(f"AUC-ROC: {auc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("Classification Report:")
        print(metrics.classification_report(y_test, y_pred, zero_division=0))

        results[name] = {
            'pipeline': pipeline,
            'accuracy': acc,
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

    return results

def sample_predictions_and_print(pipeline, X_test, sample_manual=None, nsamples=3):
    """
    Show 3 sample predictions:
      - first two are rows from X_test,
      - third is manual input if provided else a random test row
    """
    print("--- Sample Predictions ---")
    X_test_reset = X_test.reset_index(drop=True)
    samples = []

    if len(X_test_reset) >= 2:
        samples.append(X_test_reset.iloc[0])
        samples.append(X_test_reset.iloc[1])
    else:
        samples.extend([X_test_reset.iloc[i] for i in range(len(X_test_reset))])

    if sample_manual is not None:
        samples.append(pd.Series(sample_manual))
    else:
        if len(X_test_reset) >= 3:
            samples.append(X_test_reset.iloc[2])
        else:
            samples.append(X_test_reset.iloc[0])

    for i, s in enumerate(samples[:nsamples], start=1):
        # s may be Series; convert to DataFrame row
        s_df = pd.DataFrame([s])
        try:
            pred = pipeline.predict(s_df)[0]
            proba = pipeline.predict_proba(s_df)[0][1] if hasattr(pipeline, "predict_proba") else None
        except Exception as e:
            # If pipeline expects raw column order, ensure same columns present
            pred = f"Error predicting: {e}"
            proba = None
        print(f"Input {i}:")
        print(s_df.to_dict(orient='records')[0])
        print("Prediction (churn=1):", pred)
        if proba is not None:
            print("Probability churn:", round(proba, 4))
        print("-"*30)

def explain_with_shap(pipeline, X_train, X_test, model_name):
    if shap is None:
        print("SHAP not installed; skipping SHAP explanations for", model_name)
        return
    print(f"Running SHAP for {model_name} ...")
    # Extract preprocessor and classifier
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # Produce transformed train data for explainer background
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    # If classifier is tree-based, use TreeExplainer; if linear use LinearExplainer
    try:
        if hasattr(shap, "TreeExplainer") and ("RandomForest" in model_name or 'Forest' in str(type(classifier)).lower() or 'GradientBoosting' in model_name):
            expl = shap.TreeExplainer(classifier)
        else:
            expl = shap.LinearExplainer(classifier, X_train_trans, feature_perturbation="interventional")
    except Exception:
        expl = shap.Explainer(classifier, X_train_trans)

    shap_values = expl.shap_values(X_test_trans) if hasattr(expl, "shap_values") else expl(X_test_trans)

    # Lấy danh sách tên feature thật sau khi transform:
    feature_names = preprocessor.get_feature_names_out()

    # summary plot (save to file)
    save_path = os.path.join(OUTPUT_DIR, f"shap_summary_{model_name}.png")
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        # Truyền tên biến thật vào biểu đồ
        shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print("Saved SHAP summary plot to:", save_path)

    except Exception as e:
        print("Failed to save SHAP plot:", e)

def explain_with_lime(pipeline, X_train, X_test, numeric_features, categorical_features, model_name):
    if LimeTabularExplainer is None:
        print("LIME not installed; skipping LIME explanations for", model_name)
        return
    print(f"Running LIME for {model_name} ...")

    # LIME dùng dữ liệu đã qua transform để tránh lỗi so sánh chuỗi
    preprocessor = pipeline.named_steps['preprocessor']
    X_train_enc = preprocessor.transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # Lấy danh sách feature sau khi encode
    feature_names = []
    if hasattr(preprocessor.named_transformers_['num'], 'get_feature_names_out'):
        feature_names.extend(preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features))
    else:
        feature_names.extend(numeric_features)
    if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
        feature_names.extend(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    else:
        feature_names.extend(categorical_features)

    class_names = ['NoChurn', 'Churn']

    explainer = LimeTabularExplainer(
        training_data=X_train_enc,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    instance = X_test_enc[0]

    # Dự đoán trực tiếp từ pipeline
    def predict_fn(x_np):
        return pipeline.named_steps['classifier'].predict_proba(x_np)

    exp = explainer.explain_instance(instance, predict_fn, num_features=10)
    print("LIME explanation (feature weights):")
    print(exp.as_list())

    html_path = os.path.join(OUTPUT_DIR, f"lime_explanation_{model_name}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(exp.as_html())
    print("Saved LIME explanation (HTML) to:", html_path)

def main():
    print("Loading dataset...")
    df = load_and_clean(DATA_PATH)
    print("Data shape:", df.shape)
    print("Sample rows:")
    print(df.head())

    # target
    y = df['churn']
    X = df.drop(columns=['churn'])

    preprocessor, numeric_features, categorical_features = build_preprocessor(df)

    # Split with stratify to keep class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE,
                                                        stratify=y)
    print("Class distribution before training:", Counter(y_train))
    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # Fit models and evaluate
    results = fit_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    # For each trained model: sample predictions, save predictions, explain with SHAP/LIME
    # Prepare a manual example (dictionary) - you may want to adjust values according to your feature list.
    # We'll build a manual sample using median/nice values for numeric and most common for categorical.
    manual_sample = {}
    # choose median numeric values
    for col in X.select_dtypes(include=[np.number]).columns:
        manual_sample[col] = float(X[col].median())
    # choose mode for categorical
    for col in X.select_dtypes(include=['object']).columns:
        manual_sample[col] = X[col].mode()[0]

    for model_name, info in results.items():
        pipeline = info['pipeline']
        print("\n" + "#"*40)
        print("Model:", model_name)
        # sample predictions
        sample_predictions_and_print(pipeline, X_test, sample_manual=manual_sample, nsamples=3)

        # explain with SHAP
        try:
            explain_with_shap(pipeline, X_train, X_test, model_name)
        except Exception as e:
            print("Error in SHAP explanation:", e)

        # explain with LIME (use raw feature lists)
        try:
            explain_with_lime(pipeline, X_train, X_test, numeric_features, categorical_features, model_name)
        except Exception as e:
            print("Error in LIME explanation:", e)

    print("\nAll models trained and explanations attempted.")
    print("Saved models and explanation files to folder:", OUTPUT_DIR)
    print("If any explainability lib (SHAP/LIME) is missing, install it via pip and re-run.")

if __name__ == "__main__":
    main()
