import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from helper_functions import log_info, log_error

ARTIFACTS_PATH = "C:/Users/DHARSHAN KUMAR B J/Music/heart-disease-prediction/artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "heart_model.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "heart_label_encoder.pkl")

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    log_info("Model trained and saved.")
    return model

def predict(X_val):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    preds = model.predict(X_val)
    return le.inverse_transform(preds)

def evaluate(X_val, y_val):
    preds = predict(X_val)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    y_val_decoded = le.inverse_transform(y_val)
    acc = accuracy_score(y_val_decoded, preds)
    cm = confusion_matrix(y_val_decoded, preds)
    report = classification_report(y_val_decoded, preds)
    log_info("Evaluation complete.")
    return cm, acc, report
