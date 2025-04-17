import os
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                            confusion_matrix, classification_report, accuracy_score)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import xgboost as xgb

# 한글 폰트 설정
import matplotlib.font_manager as fm
import platform

system_name = platform.system()

if system_name == 'Windows':
    # Windows 환경
    plt.rc('font', family='Malgun Gothic')  # 윈도우의 '맑은 고딕' 폰트 사용
elif system_name == 'Darwin':
    # macOS 환경
    plt.rc('font', family='AppleGothic')  # macOS의 'AppleGothic' 폰트 사용
else:
    # Linux 등 기타 환경
    # Linux에 한글 폰트가 설치되어 있어야 함 (예: Nanum Gothic)
    try:
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 경로는 시스템마다 다를 수 있음
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        print("한글 폰트를 찾을 수 없습니다. 그래프의 한글이 정상적으로 표시되지 않을 수 있습니다.")

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# 프로젝트 루트 경로 - 상대 경로를 사용하도록 변경
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, 'visualizations')  # 시각화 저장 디렉토리

# 디렉토리 생성
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# 데이터 파일 경로
PERFORMANCE_DATA_PATH = os.path.join(DATA_DIR, 'performance_tb.csv')
SALES_DATA_PATH = os.path.join(DATA_DIR, 'sales_tb.csv')

# 결과 저장 경로
RESULTS_PATH = os.path.join(VISUALIZATION_DIR, 'model_results.json')

# 모델 시각화 결과 저장용 딕셔너리
model_results = {}

# 공통으로 사용되는 데이터 전처리 함수
def load_and_preprocess_data():
    """
    공통 데이터 로드 및 기본 전처리 수행
    """
    # 데이터 불러오기
    performance_df = pd.read_csv(PERFORMANCE_DATA_PATH)
    sales_df = pd.read_csv(SALES_DATA_PATH)
    
    # 'performance_id' 컬럼을 기준으로 두 데이터프레임 병합
    df = pd.merge(performance_df, sales_df, on='performance_id', how='inner')
    
    # start_date 컬럼 전처리: datetime으로 변환 후 기준 날짜와의 차이를 일수로 계산
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    reference_date = pd.Timestamp("2020-01-01")
    df['start_date_numeric'] = (df['start_date'] - reference_date).dt.days
    df.drop(columns=['start_date'], inplace=True)
    
    # duration 변수가 정수형인지 확인하고 변환
    if 'duration' in df.columns:
        df['duration'] = df['duration'].astype(int)
    else:
        print("경고: 'duration' 컬럼이 데이터셋에 없습니다.")
    
    # variable_cost_rate가 없는 경우 기본값 추가
    if 'variable_cost_rate' not in df.columns:
        print("경고: 'variable_cost_rate' 컬럼이 데이터셋에 없습니다. 기본값 0.1을 사용합니다.")
        # 티켓 가격의 10%를 변동비율로 가정
        df['variable_cost_rate'] = 0.1
    
    # production_cost가 없는 경우 대체 로직
    if 'production_cost' not in df.columns:
        print("경고: 'production_cost' 컬럼이 데이터셋에 없습니다. 마케팅 예산의 5배를 제작비로 가정합니다.")
        df['production_cost'] = df['marketing_budget'] * 5
    
    return df

# 모델 저장 함수
def save_model(model, model_name):
    """
    훈련된 모델을 통일된 경로에 저장하는 함수
    """
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f">>> 모델이 '{model_path}' 경로에 저장되었습니다.")
    return model_path

# 결과 저장 함수
def save_results_to_json():
    """
    모든 모델 결과를 JSON 파일로 저장
    """
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_results, f, indent=4, ensure_ascii=False)
    print(f">>> 모델 결과가 '{RESULTS_PATH}' 경로에 저장되었습니다.")

#-----------------------------------------------------------------------------
# 시각화 함수
#-----------------------------------------------------------------------------

def plot_feature_importance(model, feature_names, model_name, save=True):
    """
    Feature importance 시각화
    """
    # 모델 유형에 따라 feature importance 추출
    if isinstance(model, Pipeline):
        if 'regressor' in model.named_steps and isinstance(model.named_steps['regressor'], XGBRegressor):
            importances = model.named_steps['regressor'].feature_importances_
        elif 'regressor' in model.named_steps and isinstance(model.named_steps['regressor'], MultiOutputRegressor):
            # 다중 출력 모델의 경우 첫 번째 모델 사용
            importances = model.named_steps['regressor'].estimators_[0].feature_importances_
        elif 'classifier' in model.named_steps and isinstance(model.named_steps['classifier'], RandomForestClassifier):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            print(f"지원하지 않는 모델 유형: {type(model)}")
            return None
    else:
        print(f"지원하지 않는 모델 유형: {type(model)}")
        return None
    
    # 전처리 단계에서 원핫 인코딩된 특성명 추출 시도
    try:
        if 'preprocessing' in model.named_steps:
            preprocessor = model.named_steps['preprocessing']
            if 'cat' in preprocessor.transformers_:
                cat_feature_names = []
                cat_indices = []
                
                for name, transformer, columns in preprocessor.transformers_:
                    if name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                        cat_feature_names.extend(transformer.get_feature_names_out(columns))
                        cat_indices.append(len(cat_feature_names))
                
                if cat_feature_names:
                    # 범주형 + 수치형 특성 합치기
                    numerical_features = [f for f in feature_names if f not in preprocessor.transformers_[0][2]]
                    feature_names = list(cat_feature_names) + numerical_features
    except Exception as e:
        print(f"특성명 추출 중 오류: {str(e)}")
    
    # 특성 중요도 데이터 수집
    if len(feature_names) != len(importances):
        print(f"특성 수 불일치: {len(feature_names)} vs {len(importances)}")
        # 일치하지 않는 경우 인덱스 사용
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # 특성 중요도 시각화
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.title(f'Feature Importance - {model_name}')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    # 결과 저장
    if save:
        plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_feature_importance.png')
        plt.savefig(plt_path)
        print(f">>> 특성 중요도 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # JSON에 저장할 데이터 생성
    importance_data = {
        'features': [feature_names[i] for i in indices],
        'importance_values': importances[indices].tolist()
    }
    
    return importance_data

def plot_regression_results(y_test, y_pred, model_name, save=True):
    """
    실제값과 예측값 비교 산점도 시각화
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - {model_name}')
    
    # 결과 저장
    if save:
        plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_regression_results.png')
        plt.savefig(plt_path)
        print(f">>> 회귀 결과 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # 잔차 분포 시각화
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution - {model_name}')
    
    if save:
        plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_residuals.png')
        plt.savefig(plt_path)
    
    # JSON에 저장할 데이터 생성
    scatter_data = {
        'actual': y_test.tolist() if isinstance(y_test, (pd.Series, np.ndarray)) else y_test,
        'predicted': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        'residuals': residuals.tolist() if isinstance(residuals, np.ndarray) else residuals
    }
    
    return scatter_data

def plot_multi_output_regression(y_test, y_pred, target_names, model_name, save=True):
    """
    다중 출력 회귀 모델 결과 시각화
    """
    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 5))
    scatter_data = {}
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        ax.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
        ax.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 
                [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'k--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{target}')
        
        # JSON에 저장할 데이터 수집
        scatter_data[target] = {
            'actual': y_test.iloc[:, i].tolist(),
            'predicted': y_pred[:, i].tolist(),
            'residuals': (y_test.iloc[:, i] - y_pred[:, i]).tolist()
        }
    
    plt.suptitle(f'Actual vs Predicted - {model_name}')
    plt.tight_layout()
    
    # 결과 저장
    if save:
        plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_multi_regression.png')
        plt.savefig(plt_path)
        print(f">>> 다중 회귀 결과 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    return scatter_data

def plot_confusion_matrix(y_test, y_pred, model_name, save=True):
    """
    혼동 행렬 시각화
    """
    # 모든 가능한 클래스 레이블 가져오기 (0, 1, 2 예상)
    labels = sorted(list(set(list(y_test) + list(y_pred))))
    if len(labels) <= 1:
        print(f"경고: 단일 클래스만 있습니다. 레이블: {labels}")
        # 더미 혼동 행렬 생성
        cm = np.array([[len(y_test)]])
    else:
        cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # 결과 저장
    if save:
        plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_confusion_matrix.png')
        plt.savefig(plt_path)
        print(f">>> 혼동 행렬 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # 분류 보고서 생성
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # JSON에 저장할 데이터 생성
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return cm_data

def plot_learning_curve(estimator, X, y, model_name, save=True):
    """
    학습 곡선 시각화
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend(loc='best')
    
    # 결과 저장
    if save:
        plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_learning_curve.png')
        plt.savefig(plt_path)
        print(f">>> 학습 곡선 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # JSON에 저장할 데이터 생성
    curve_data = {
        'train_sizes': train_sizes.tolist(),
        'train_scores': {
            'mean': train_mean.tolist(),
            'std': train_std.tolist()
        },
        'test_scores': {
            'mean': test_mean.tolist(),
            'std': test_std.tolist()
        }
    }
    
    return curve_data

#-----------------------------------------------------------------------------
# 모델 1: 관객 수 예측 (기획 단계) XGBoost
#-----------------------------------------------------------------------------
def train_xgb_accumulated_sales_planning():
    """기획 단계의 누적 판매량 예측 모델 훈련"""
    model_name = 'xgb_reg_accumulated_sales_planning'
    df = load_and_preprocess_data()
    
    # 회귀 예측을 위한 피처(X)와 타깃(y) 정의
    feature_cols = [
        'genre', 
        'region', 
        'start_date_numeric',
        'capacity', 
        'star_power', 
        'ticket_price',
        'marketing_budget', 
        'sns_mention_count',
        'duration'  # 공연 기간 변수 추가
    ]
    
    # 입력 변수가 모두 존재하는지 확인
    for col in feature_cols:
        if col not in df.columns:
            print(f"경고: '{col}' 컬럼이 데이터셋에 없습니다.")
    
    # 실제 존재하는 컬럼만 사용
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    y = df['accumulated_sales']
    
    # 범주형/수치형 변수 지정
    categorical_features = [col for col in ['genre', 'region'] if col in available_features]
    numerical_features = [col for col in available_features if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    # XGBoost 모델 구성
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            random_state=42,
            eval_metric='rmse'
        ))
    ])
    
    # 학습 데이터 분할 및 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # eval_set 설정
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # 학습 과정 추적
    evals_result = {}
    model.fit(X_train, y_train, 
              regressor__eval_set=eval_set,
              regressor__verbose=False,
              regressor__early_stopping_rounds=10,
              regressor__callbacks=[xgb.callback.EvaluationMonitor(evals_result)])
    
    # 테스트 데이터 예측
    y_pred = model.predict(X_test)
    
    # 평가 지표 계산
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"[관객 수 예측 (기획 단계): XGBoost]")
    print(f" - R² Score: {r2:.4f}")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" - Mean Absolute Error (MAE): {mae:.4f}")
    
    # 모델 저장
    save_model(model, model_name)
    
    # 시각화
    importance_data = plot_feature_importance(model, available_features, model_name)
    scatter_data = plot_regression_results(y_test, y_pred, model_name)
    learning_data = plot_learning_curve(model, X, y, model_name)
    
    # 학습 과정 시각화
    plt.figure(figsize=(10, 6))
    
    if 'validation_0' in evals_result:
        # XGBoost 1.7.x 버전 이상
        plt.plot(evals_result['validation_0']['rmse'], label='Training')
        plt.plot(evals_result['validation_1']['rmse'], label='Validation')
    else:
        # evals_result가 비어있는 경우
        print("경고: 학습 과정 데이터를 가져올 수 없습니다.")
    
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.title(f'XGBoost Training Progress - {model_name}')
    plt.legend()
    
    plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_train_progress.png')
    plt.savefig(plt_path)
    print(f">>> 학습 진행 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # 결과 저장
    model_results[model_name] = {
        'metrics': {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        },
        'feature_importance': importance_data,
        'regression_results': scatter_data,
        'learning_curve': learning_data,
        'train_progress': evals_result
    }
    
    return model

#-----------------------------------------------------------------------------
# 모델 2: 관객 수 예측 (판매 중) XGBoost
#-----------------------------------------------------------------------------
def train_xgb_accumulated_sales_selling():
    """판매 중 단계의 누적 판매량 예측 모델 훈련"""
    model_name = 'xgb_reg_accumulated_sales_selling'
    df = load_and_preprocess_data()
    
    # 회귀 예측을 위한 피처(X)와 타깃(y) 정의
    feature_cols = [
        'genre', 
        'region', 
        'start_date_numeric',
        'capacity', 
        'star_power', 
        'ticket_price',
        'marketing_budget', 
        'sns_mention_count',
        'daily_sales', 
        'booking_rate', 
        'ad_exposure', 
        'sns_mention_daily',
        'duration'  # 공연 기간 변수 추가
    ]
    
    # 입력 변수가 모두 존재하는지 확인
    for col in feature_cols:
        if col not in df.columns:
            print(f"경고: '{col}' 컬럼이 데이터셋에 없습니다.")
    
    # 실제 존재하는 컬럼만 사용
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    y = df['accumulated_sales']
    
    # 범주형/수치형 변수 지정
    categorical_features = [col for col in ['genre', 'region'] if col in available_features]
    numerical_features = [col for col in available_features if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    # XGBoost 모델 구성
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            random_state=42,
            eval_metric='rmse'
        ))
    ])
    
    # 학습 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # eval_set 설정
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # 학습 과정 추적
    evals_result = {}
    
    # 모델 학습
    model.fit(X_train, y_train, 
              regressor__eval_set=eval_set,
              regressor__verbose=False,
              regressor__early_stopping_rounds=10,
              regressor__callbacks=[xgb.callback.EvaluationMonitor(evals_result)])
    
    # 테스트 데이터 예측
    y_pred = model.predict(X_test)
    
    # 평가 지표 계산
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"[관객 수 예측 (판매 중): XGBoost]")
    print(f" - R² Score: {r2:.4f}")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" - Mean Absolute Error (MAE): {mae:.4f}")
    
    # 모델 저장
    save_model(model, model_name)
    
    # 시각화
    importance_data = plot_feature_importance(model, available_features, model_name)
    scatter_data = plot_regression_results(y_test, y_pred, model_name)
    learning_data = plot_learning_curve(model, X, y, model_name)
    
    # 학습 과정 시각화
    plt.figure(figsize=(10, 6))
    
    if 'validation_0' in evals_result:
        # XGBoost 1.7.x 버전 이상
        plt.plot(evals_result['validation_0']['rmse'], label='Training')
        plt.plot(evals_result['validation_1']['rmse'], label='Validation')
    else:
        # evals_result가 비어있는 경우
        print("경고: 학습 과정 데이터를 가져올 수 없습니다.")
    
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.title(f'XGBoost Training Progress - {model_name}')
    plt.legend()
    
    plt_path = os.path.join(VISUALIZATION_DIR, f'{model_name}_train_progress.png')
    plt.savefig(plt_path)
    print(f">>> 학습 진행 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # 결과 저장
    model_results[model_name] = {
        'metrics': {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        },
        'feature_importance': importance_data,
        'regression_results': scatter_data,
        'learning_curve': learning_data,
        'train_progress': evals_result
    }
    
    return model

#-----------------------------------------------------------------------------
# 모델 3: ROI, BEP 예측 (기획 단계) Multi-output XGBoost
#-----------------------------------------------------------------------------
def train_xgb_roi_bep_planning():
    """기획 단계의 ROI, BEP 예측 모델 훈련"""
    model_name = 'xgb_reg_roi_bep_planning'
    df = load_and_preprocess_data()
    
    # 필요한 변수 확인
    required_vars = ['production_cost', 'marketing_budget', 'ticket_price', 
                     'accumulated_sales', 'variable_cost_rate']
    
    for var in required_vars:
        if var not in df.columns:
            # production_cost가 없는 경우 대체 로직
            if var == 'production_cost':
                print(f"경고: '{var}' 컬럼이 없습니다. 마케팅 예산의 5배를 제작비로 가정합니다.")
                df['production_cost'] = df['marketing_budget'] * 5
            else:
                print(f"경고: '{var}' 컬럼이 없습니다.")
    
    # ROI, BEP 계산 컬럼 추가
    df['total_revenue'] = df['ticket_price'] * df['accumulated_sales']
    df['total_cost'] = df['production_cost'] + df['marketing_budget'] + (
        df['ticket_price'] * df['variable_cost_rate'] * df['accumulated_sales']
    )
    df['ROI'] = (df['total_revenue'] - df['total_cost']) / df['total_cost']
    df['BEP'] = (df['production_cost'] + df['marketing_budget']) / (
        df['ticket_price'] - (df['ticket_price'] * df['variable_cost_rate'])
    )
    
    # 피처(X)와 목표(y) 설정
    feature_cols = [
        'production_cost',
        'marketing_budget',
        'ticket_price',
        'capacity',
        'variable_cost_rate',
        'duration'  # 공연 기간 변수 추가
    ]
    
    # 입력 변수가 모두 존재하는지 확인
    for col in feature_cols:
        if col not in df.columns:
            print(f"경고: '{col}' 컬럼이 데이터셋에 없습니다.")
    
    # 실제 존재하는 컬럼만 사용
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].apply(pd.to_numeric, errors='coerce')
    y = df[['ROI', 'BEP']].apply(pd.to_numeric, errors='coerce')
    
    # 전처리
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', available_features)
        ]
    )
    
    # XGBoost 모델 설정 (다중 출력 회귀를 위해 MultiOutputRegressor 사용)
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    multioutput_regressor = MultiOutputRegressor(xgb_regressor)
    
    # 파이프라인 구성
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', multioutput_regressor)
    ])
    
    # 학습 데이터 분할 및 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # 테스트 데이터 예측 및 평가 지표 계산
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("[손익 예측 (기획 단계): Multi-output XGBoost]")
    print(f" - R² Score: {r2:.4f}")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" - Mean Absolute Error (MAE): {mae:.4f}")
    
    # 모델 저장
    save_model(model, model_name)
    
    # 시각화
    importance_data = plot_feature_importance(model, available_features, model_name)
    scatter_data = plot_multi_output_regression(y_test, y_pred, ['ROI', 'BEP'], model_name)
    
    # 결과 저장
    model_results[model_name] = {
        'metrics': {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        },
        'feature_importance': importance_data,
        'regression_results': scatter_data
    }
    
    return model

#-----------------------------------------------------------------------------
# 모델 4: ROI, BEP 예측 (판매 중) Multi-output XGBoost
#-----------------------------------------------------------------------------
def train_xgb_roi_bep_selling():
    """판매 중 단계의 ROI, BEP 예측 모델 훈련"""
    model_name = 'xgb_reg_roi_bep_selling'
    df = load_and_preprocess_data()
    
    # 필요한 변수 확인
    required_vars = ['production_cost', 'marketing_budget', 'ticket_price', 
                     'accumulated_sales', 'variable_cost_rate']
    
    for var in required_vars:
        if var not in df.columns:
            # production_cost가 없는 경우 대체 로직
            if var == 'production_cost':
                print(f"경고: '{var}' 컬럼이 없습니다. 마케팅 예산의 5배를 제작비로 가정합니다.")
                df['production_cost'] = df['marketing_budget'] * 5
            else:
                print(f"경고: '{var}' 컬럼이 없습니다.")
    
    # ROI와 BEP 계산 컬럼 추가
    df['total_revenue'] = df['ticket_price'] * df['accumulated_sales']
    df['total_cost'] = df['production_cost'] + df['marketing_budget'] + (
        df['ticket_price'] * df['variable_cost_rate'] * df['accumulated_sales']
    )
    df['ROI'] = (df['total_revenue'] - df['total_cost']) / df['total_cost']
    df['BEP'] = (df['production_cost'] + df['marketing_budget']) / (
        df['ticket_price'] - (df['ticket_price'] * df['variable_cost_rate'])
    )
    
    # 피처(X)와 목표(y) 설정  
    feature_cols = [
        'production_cost',
        'marketing_budget',
        'ticket_price',
        'capacity',
        'variable_cost_rate',
        'accumulated_sales',
        'duration'  # 공연 기간 변수 추가
    ]
    
    # 입력 변수가 모두 존재하는지 확인
    for col in feature_cols:
        if col not in df.columns:
            print(f"경고: '{col}' 컬럼이 데이터셋에 없습니다.")
    
    # 실제 존재하는 컬럼만 사용
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].apply(pd.to_numeric, errors='coerce')
    y = df[['ROI', 'BEP']].apply(pd.to_numeric, errors='coerce')
    
    # 전처리
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', available_features)
        ]
    )
    
    # XGBoost 모델 설정 (다중 출력 회귀를 위해 MultiOutputRegressor 사용)
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    multioutput_regressor = MultiOutputRegressor(xgb_regressor)
    
    # 파이프라인 구성
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', multioutput_regressor)
    ])
    
    # 학습 데이터 분할 및 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # 테스트 데이터 예측 및 평가 지표 계산
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("[손익 예측 (판매 중): Multi-output XGBoost]")
    print(f" - R² Score: {r2:.4f}")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" - Mean Absolute Error (MAE): {mae:.4f}")
    
    # 모델 저장
    save_model(model, model_name)
    
    # 시각화
    importance_data = plot_feature_importance(model, available_features, model_name)
    scatter_data = plot_multi_output_regression(y_test, y_pred, ['ROI', 'BEP'], model_name)
    
    # 결과 저장
    model_results[model_name] = {
        'metrics': {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        },
        'feature_importance': importance_data,
        'regression_results': scatter_data
    }
    
    return model

#-----------------------------------------------------------------------------
# 모델 5: 티켓 판매 위험 예측 RandomForest
#-----------------------------------------------------------------------------
def train_rf_cls_ticket_risk():
    """티켓 판매 위험 예측 모델 훈련"""
    model_name = 'rf_cls_ticket_risk'
    df = load_and_preprocess_data()
    
    # booking_rate가 없으면 계산
    if 'booking_rate' not in df.columns:
        print("경고: 'booking_rate' 컬럼이 없습니다. accumulated_sales와 capacity를 이용해 계산합니다.")
        df['booking_rate'] = (df['accumulated_sales'] / df['capacity']) * 100
    
    # '위험도' 라벨 생성
    def classify_risk(rate):
        if rate >= 75:
            return 0  # 저위험
        elif rate >= 60:
            return 1  # 중위험
        else:
            return 2  # 고위험
    
    df['risk_label'] = df['booking_rate'].apply(classify_risk)
    
    # 피처와 타깃 정의
    feature_cols = [
        'genre', 
        'region', 
        'start_date_numeric',
        'capacity', 
        'star_power', 
        'daily_sales', 
        'accumulated_sales', 
        'ad_exposure', 
        'sns_mention_daily', 
        'promo_event_flag',
        'duration'  # 공연 기간 변수 추가
    ]
    
    # 입력 변수가 모두 존재하는지 확인
    for col in feature_cols:
        if col not in df.columns:
            print(f"경고: '{col}' 컬럼이 데이터셋에 없습니다.")
    
    # 실제 존재하는 컬럼만 사용
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    y = df['risk_label']
    
    # 범주형/수치형 데이터 전처리
    cat_features = ['genre', 'region', 'promo_event_flag']
    categorical_features = [col for col in cat_features if col in available_features]
    numerical_features = [col for col in available_features if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    # 모델 구성 및 학습
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[티켓 판매 위험 예측] Accuracy: {accuracy:.4f}")
    
    # 모델 저장
    save_model(model, model_name)
    
    # 혼동 행렬 시각화
    cm_data = plot_confusion_matrix(y_test, y_pred, model_name)
    
    # 특성 중요도 시각화
    importance_data = plot_feature_importance(model, available_features, model_name)
    
    # 결과 저장
    model_results[model_name] = {
        'metrics': {
            'accuracy': accuracy,
            'classification_report': cm_data['classification_report']
        },
        'confusion_matrix': cm_data['confusion_matrix'],
        'feature_importance': importance_data
    }
    
    # 추가: ground truth 데이터셋 저장 (평가용)
    ground_truth_path = os.path.join(DATA_DIR, 'ground_truth.csv')
    df[['risk_label']].to_csv(ground_truth_path, index=False)
    print(f">>> ground_truth.csv 파일이 '{ground_truth_path}' 경로에 저장되었습니다.")
    
    return model

# 모델 성능 시각화 함수
def plot_model_comparison():
    """모든 모델의 성능 지표를 비교하는 그래프 생성"""
    # 회귀 모델 지표 비교
    regression_models = [
        'xgb_reg_accumulated_sales_planning',
        'xgb_reg_accumulated_sales_selling',
        'xgb_reg_roi_bep_planning',
        'xgb_reg_roi_bep_selling'
    ]
    
    # 각 모델에서 지표 추출
    metrics = {}
    for metric in ['r2', 'rmse', 'mae']:
        metrics[metric] = []
        for model in regression_models:
            if model in model_results and 'metrics' in model_results[model]:
                metrics[metric].append(model_results[model]['metrics'].get(metric, 0))
            else:
                metrics[metric].append(0)
    
    # R² 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.bar(regression_models, metrics['r2'])
    plt.title('R² Score Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)  # R²은 일반적으로 0-1 사이
    
    # RMSE 시각화
    plt.subplot(1, 3, 2)
    plt.bar(regression_models, metrics['rmse'])
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)
    
    # MAE 시각화
    plt.subplot(1, 3, 3)
    plt.bar(regression_models, metrics['mae'])
    plt.title('MAE Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt_path = os.path.join(VISUALIZATION_DIR, 'model_comparison_regression.png')
    plt.savefig(plt_path)
    print(f">>> 회귀 모델 비교 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # 분류 모델 정확도 (RF 모델)
    if 'rf_cls_ticket_risk' in model_results and 'metrics' in model_results['rf_cls_ticket_risk']:
        accuracy = model_results['rf_cls_ticket_risk']['metrics'].get('accuracy', 0)
        
        plt.figure(figsize=(6, 4))
        plt.bar(['Ticket Risk Classifier'], [accuracy])
        plt.title('Classification Accuracy')
        plt.ylim(0, 1)
        
        plt_path = os.path.join(VISUALIZATION_DIR, 'model_comparison_classification.png')
        plt.savefig(plt_path)
        print(f">>> 분류 모델 비교 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # 종합 결과 JSON에 추가
    model_results['comparison'] = {
        'regression_metrics': {
            'models': regression_models,
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae']
        }
    }
    
    if 'rf_cls_ticket_risk' in model_results and 'metrics' in model_results['rf_cls_ticket_risk']:
        model_results['comparison']['classification_metrics'] = {
            'models': ['rf_cls_ticket_risk'],
            'accuracy': [accuracy]
        }
    
    # JSON 업데이트
    save_results_to_json()

# 데이터셋 탐색 함수 추가
def explore_dataset():
    """
    데이터셋의 구조와 내용을 탐색하여 시각화하고 JSON으로 저장합니다.
    """
    print("데이터셋 탐색 시작...")
    
    # 데이터 로드
    df = load_and_preprocess_data()
    
    # 기술 통계량 계산
    numeric_stats = df.describe().to_dict()
    
    # 주요 변수 분포 시각화
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 0:
        # 최대 9개의 히스토그램만 생성
        cols_to_plot = numeric_cols[:min(9, len(numeric_cols))]
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cols_to_plot):
            plt.subplot(3, 3, i+1)
            plt.hist(df[col].dropna(), bins=20, alpha=0.7)
            plt.title(col)
        plt.tight_layout()
        plt_path = os.path.join(VISUALIZATION_DIR, 'dataset_histograms.png')
        plt.savefig(plt_path)
        print(f">>> 데이터셋 히스토그램이 '{plt_path}' 경로에 저장되었습니다.")
    
    # 상관 행렬 계산 및 시각화
    corr_cols = [col for col in numeric_cols if col != 'performance_id']
    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt_path = os.path.join(VISUALIZATION_DIR, 'correlation_matrix.png')
        plt.savefig(plt_path)
        print(f">>> 상관 행렬 히트맵이 '{plt_path}' 경로에 저장되었습니다.")
        
        # JSON에 추가
        corr_json = corr_matrix.to_dict()
    else:
        corr_json = {}
    
    # 범주형 변수 분포
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_dist = {}
    
    if len(cat_cols) > 0:
        # 범주형 변수 분포 시각화
        plt.figure(figsize=(15, len(cat_cols) * 3))
        for i, col in enumerate(cat_cols):
            plt.subplot(len(cat_cols), 1, i+1)
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            cat_dist[col] = value_counts.to_dict()
        
        plt.tight_layout()
        plt_path = os.path.join(VISUALIZATION_DIR, 'categorical_distributions.png')
        plt.savefig(plt_path)
        print(f">>> 범주형 변수 분포가 '{plt_path}' 경로에 저장되었습니다.")
    
    # 결과 저장
    dataset_stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'numeric_stats': numeric_stats,
        'correlation_matrix': corr_json,
        'categorical_distributions': cat_dist
    }
    
    # JSON으로 저장
    dataset_path = os.path.join(VISUALIZATION_DIR, 'dataset_statistics.json')
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, indent=4, ensure_ascii=False)
    print(f">>> 데이터셋 통계가 '{dataset_path}' 경로에 JSON 형식으로 저장되었습니다.")
    
    return df

# 전체 모델 학습 실행
#-----------------------------------------------------------------------------
def train_all_models():
    """모든 모델을 순차적으로 학습하고 결과를 JSON으로 저장"""
    print("===== 모델 훈련 시작 =====")
    
    # 입력 변수 사용 정보 출력
    print("모든 모델에 'duration'(공연 기간) 변수가 추가되었습니다.")
    print("- duration: 공연 기간(일수, 정수형)")
    
    try:
        print("\n1. 기획 단계 누적 판매량 예측 모델 학습")
        train_xgb_accumulated_sales_planning()
    except Exception as e:
        print(f"기획 단계 누적 판매량 예측 모델 학습 중 오류 발생: {str(e)}")
    
    try:
        print("\n2. 판매 중 누적 판매량 예측 모델 학습")
        train_xgb_accumulated_sales_selling()
    except Exception as e:
        print(f"판매 중 누적 판매량 예측 모델 학습 중 오류 발생: {str(e)}")
    
    try:
        print("\n3. 기획 단계 ROI/BEP 예측 모델 학습")
        train_xgb_roi_bep_planning()
    except Exception as e:
        print(f"기획 단계 ROI/BEP 예측 모델 학습 중 오류 발생: {str(e)}")
    
    try:
        print("\n4. 판매 중 ROI/BEP 예측 모델 학습")
        train_xgb_roi_bep_selling()
    except Exception as e:
        print(f"판매 중 ROI/BEP 예측 모델 학습 중 오류 발생: {str(e)}")
    
    try:
        print("\n5. 티켓 판매 위험 예측 모델 학습")
        train_rf_cls_ticket_risk()
    except Exception as e:
        print(f"티켓 판매 위험 예측 모델 학습 중 오류 발생: {str(e)}")
    
    # 모든 모델 결과를 JSON으로 저장
    save_results_to_json()
    
    print("\n===== 모든 모델 학습 완료 =====")
    print(f"모델이 '{MODELS_DIR}' 디렉토리에 저장되었습니다.")
    print(f"모델 시각화 결과가 '{VISUALIZATION_DIR}' 디렉토리에 저장되었습니다.")
    print(f"시각화 데이터가 '{RESULTS_PATH}' 파일에 JSON 형식으로 저장되었습니다.")

# 모델 예측 및 시각화 함수
def predict_and_visualize(performance_data, model_type='all'):
    """
    새로운 공연 데이터에 대한 예측을 수행하고 결과를 시각화
    
    Args:
        performance_data (dict): 공연 데이터 (각 특성에 대한 값 포함)
        model_type (str): 'planning', 'selling', 'risk' 또는 'all'
    
    Returns:
        dict: 예측 결과
    """
    # 모델 로드
    models = load_models()
    
    # 입력 데이터 DataFrame으로 변환
    input_df = pd.DataFrame([performance_data])
    
    results = {}
    
    # 판매량 예측 (기획 단계)
    if model_type in ['planning', 'all']:
        planning_sales = models['planning_sales'].predict(input_df)
        results['accumulated_sales_planning'] = float(planning_sales[0])
        
        # ROI, BEP 예측 (기획 단계)
        roi_bep_planning = models['planning_roi_bep'].predict(input_df)
        results['roi_planning'] = float(roi_bep_planning[0][0])
        results['bep_planning'] = float(roi_bep_planning[0][1])
    
    # 판매량 예측 (판매 중 단계)
    if model_type in ['selling', 'all'] and 'daily_sales' in performance_data:
        selling_sales = models['selling_sales'].predict(input_df)
        results['accumulated_sales_selling'] = float(selling_sales[0])
        
        # ROI, BEP 예측 (판매 중 단계)
        roi_bep_selling = models['selling_roi_bep'].predict(input_df)
        results['roi_selling'] = float(roi_bep_selling[0][0])
        results['bep_selling'] = float(roi_bep_selling[0][1])
    
    # 티켓 판매 위험 예측
    if model_type in ['risk', 'all'] and 'booking_rate' in performance_data:
        risk_pred = models['risk'].predict(input_df)
        risk_proba = models['risk'].predict_proba(input_df)
        
        risk_level = int(risk_pred[0])
        risk_labels = ['저위험', '중위험', '고위험']
        
        results['risk_level'] = risk_level
        results['risk_label'] = risk_labels[risk_level]
        results['risk_probabilities'] = risk_proba[0].tolist()
    
    # 시각화 (선택적)
    if 'accumulated_sales_planning' in results and 'accumulated_sales_selling' in results:
        plt.figure(figsize=(10, 6))
        labels = ['기획 단계 예측', '판매 중 예측']
        values = [results['accumulated_sales_planning'], results['accumulated_sales_selling']]
        plt.bar(labels, values)
        plt.title('누적 판매량 예측 비교')
        plt.ylabel('예상 누적 판매량')
        
        plt_path = os.path.join(VISUALIZATION_DIR, 'sales_prediction_comparison.png')
        plt.savefig(plt_path)
        print(f">>> 판매량 예측 비교 그래프가 '{plt_path}' 경로에 저장되었습니다.")
    
    # JSON으로 결과 저장
    prediction_path = os.path.join(VISUALIZATION_DIR, 'prediction_results.json')
    with open(prediction_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f">>> 예측 결과가 '{prediction_path}' 경로에 JSON 형식으로 저장되었습니다.")
    
    return results

# 모델 로드 함수
def load_models():
    """모든 훈련된 모델 로드"""
    models = {
        'planning_sales': joblib.load(os.path.join(MODELS_DIR, 'xgb_reg_accumulated_sales_planning.pkl')),
        'selling_sales': joblib.load(os.path.join(MODELS_DIR, 'xgb_reg_accumulated_sales_selling.pkl')),
        'planning_roi_bep': joblib.load(os.path.join(MODELS_DIR, 'xgb_reg_roi_bep_planning.pkl')),
        'selling_roi_bep': joblib.load(os.path.join(MODELS_DIR, 'xgb_reg_roi_bep_selling.pkl')),
        'risk': joblib.load(os.path.join(MODELS_DIR, 'rf_cls_ticket_risk.pkl'))
    }
    return models

# 메인 함수로 실행 가능하도록 설정
if __name__ == "__main__":
    try:
        # 데이터셋 탐색 및 시각화
        explore_dataset()
        
        # 모든 모델 학습
        train_all_models()
        
        # 모델 성능 비교 시각화
        plot_model_comparison()
        
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()