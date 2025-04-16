# analysis_module.py

import os
import joblib
import pandas as pd
import numpy as np
from typing import List

# 현재 파일(analysis_module.py)이 위치한 디렉터리
FILE_DIR = os.path.dirname(__file__)

def load_model(model_name: str):
    """
    model_name = 'rf_reg_accumulated_sales', 'rf_reg_roi_bep' 등
    모델 폴더(models)에서 해당 pkl을 로드
    """
    model_path = os.path.join(FILE_DIR, "models", f"{model_name}.pkl")
    model = joblib.load(model_path)
    return model


# ---------------------------
# 1) 회귀: 관객 수 예측
# ---------------------------
def predict_accumulated_sales(input_data: List[dict]) -> np.ndarray:
    model = load_model("rf_reg_accumulated_sales")
    df = pd.DataFrame(input_data)
    preds = model.predict(df)
    return preds


# ---------------------------
# 2) 회귀: 손익 예측 (ROI, BEP)
# ---------------------------
def predict_roi_bep(input_data: List[dict]) -> np.ndarray:
    model = load_model("rf_reg_roi_bep")
    df = pd.DataFrame(input_data)
    preds = model.predict(df)  # 다중 출력인 경우 (n_samples, 2) 형태
    return preds


# ---------------------------
# 3) 분류: 티켓 판매 위험 예측
# ---------------------------
def predict_ticket_risk(input_data: List[dict]) -> np.ndarray:
    model = load_model("rf_cls_ticket_risk")
    df = pd.DataFrame(input_data)
    preds = model.predict(df)  # 예: [0, 1, 2] => 위험도
    return preds


# ---------------------------
# 4) 군집: 관객 세분화
# ---------------------------
def predict_audience_cluster(input_data: List[dict]) -> np.ndarray:
    model = load_model("kmeans_audience_seg")
    df = pd.DataFrame(input_data)
    clusters = model.predict(df)
    return clusters
