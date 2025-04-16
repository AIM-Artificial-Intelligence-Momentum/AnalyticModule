import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

# 1) 데이터 불러오기 (절대경로 사용)
performance_df = pd.read_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\performance_tb.csv')
sales_df = pd.read_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\sales_tb.csv')

# 'performance_id' 컬럼을 기준으로 두 데이터프레임 병합
df = pd.merge(performance_df, sales_df, on='performance_id', how='inner')

# 2) start_date 컬럼 전처리: datetime으로 변환 후 기준 날짜와의 차이를 일수로 계산
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
reference_date = pd.Timestamp("2020-01-01")
df['start_date_numeric'] = (df['start_date'] - reference_date).dt.days

# 원본 start_date 컬럼 삭제
df.drop(columns=['start_date'], inplace=True)

# 3) 회귀 예측을 위한 피처(X)와 타깃(y) 정의
# 예시에서는 'accumulated_sales'(누적 판매 티켓 수)를 예측 대상으로 사용합니다.
feature_cols = [
    'genre', 
    'region', 
    'start_date_numeric',  # 변환한 숫자형 날짜 컬럼
    'capacity', 
    'star_power', 
    'ticket_price',
    'marketing_budget', 
    'sns_mention_count',
    'daily_sales', 
    'booking_rate', 
    'ad_exposure', 
    'sns_mention_daily'
]
X = df[feature_cols]
y = df['accumulated_sales']

# 4) 범주형/수치형 변수 지정 (날짜는 이미 숫자로 변환됨)
categorical_features = ['genre', 'region']
numerical_features = [col for col in feature_cols if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# 5) XGBoost 모델 구성
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        random_state=42
    ))
])

# 학습 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 6) 테스트 데이터 예측
y_pred = model.predict(X_test)

# 7) 평가 지표 계산
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"[관객 수 예측 (판매 중): XGBoost]")
print(f" - R² Score: {r2:.4f}")
print(f" - Mean Squared Error (MSE): {mse:.4f}")
print(f" - Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" - Mean Absolute Error (MAE): {mae:.4f}")

# 8) 모델 저장
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgb_reg_accumulated_sales_selling.pkl')
print(">>> 모델이 'models/xgb_reg_accumulated_sales_selling.pkl' 로 저장되었습니다.")
