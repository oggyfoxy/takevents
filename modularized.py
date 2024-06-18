import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import phospho
from dotenv import load_dotenv
import joblib
import xgboost as xgb
from sklearn.pipeline import make_pipeline
import numpy as np


def load_data(limit=10000):
    load_dotenv()
    phospho.init()
    tasks_df = phospho.tasks_df(limit=limit)
    tasks_df.set_index("task_id", inplace=True)
    new_df = tasks_df.dropna(subset=["task_input", "task_output"]).copy()
    new_df["text"] = new_df[["task_input", "task_output"]].agg(" ".join, axis=1)

    event_df = pd.get_dummies(tasks_df["event_name"]).astype(int)
    new_df = new_df[["text"]].merge(event_df, left_index=True, right_index=True)
    new_df = new_df.drop_duplicates()
    new_df = new_df.groupby(by=["task_id", "text"]).sum().reset_index()
    new_df = new_df.set_index("task_id")
    return new_df


def preprocess_data(df):
    X = df["text"]
    y = df.drop(columns="text")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, params):
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.8, ngram_range=(1, 2))
    xgb_model = xgb.XGBClassifier(
        **params, use_label_encoder=False, eval_metric="logloss"
    )

    text_classifier = make_pipeline(vectorizer, xgb_model)
    text_classifier.fit(X_train, y_train)

    return text_classifier


def evaluate_model(model, X_test, y_test):
    import time
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        multilabel_confusion_matrix,
    )

    start_ts = time.time()
    y_pred = model.predict(X_test)
    end_ts = time.time()

    print(f"Prediction Time [s]: {(end_ts-start_ts):.3f}")

    f1 = f1_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    confusion = multilabel_confusion_matrix(y_test, y_pred)

    print("F1 Score : {:.3f}".format(f1))
    print("Recall Score : {:.3f}".format(recall))
    print("Precision Score : {:.3f}".format(precision))
    print("Confusion Matrix : ", confusion)


def save_model(model, filepath):
    joblib.dump(model, filepath)


def load_model(filepath):
    return joblib.load(filepath)


if __name__ == "__main__":
    df = load_data(limit=10000)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    params = {
        "max_depth": 2,
        "learning_rate": 0.09982190399764716,
        "n_estimators": 444,
    }

    model = train_model(X_train, y_train, params)
    evaluate_model(model, X_test, y_test)

    save_model(model, "text_classifier.joblib")

    # To load and use the model later
    # model = load_model("text_classifier.joblib")
    # new_predictions = model.predict(new_X)
