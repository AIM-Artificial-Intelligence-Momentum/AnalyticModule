import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib
import os

# 1) 데이터 불러오기 (절대경로 사용)
performance_df = pd.read_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\performance_tb.csv')
sales_df = pd.read_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\sales_tb.csv')

# 'performance_id' 컬럼 기준으로 병합 (학습을 위해 과거 완성된 공연 데이터를 사용)
df = pd.merge(performance_df, sales_df, on='performance_id', how='inner')

# 2) start_date 컬럼 전처리: datetime 변환 후 기준일과의 차이를 일수로 계산
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
reference_date = pd.Timestamp("2020-01-01")
df['start_date_numeric'] = (df['start_date'] - reference_date).dt.days
df.drop(columns=['start_date'], inplace=True)

# 3) 기획 단계 ROI, BEP 계산 컬럼 추가  
# (ROI, BEP는 과거 누적 판매(accumulated_sales)와 정적 비용 변수를 이용해 산출)
df['total_revenue'] = df['ticket_price'] * df['accumulated_sales']
df['total_cost'] = df['production_cost'] + df['marketing_budget'] + (
    df['ticket_price'] * df['variable_cost_rate'] * df['accumulated_sales']
)
df['ROI'] = (df['total_revenue'] - df['total_cost']) / df['total_cost']
df['BEP'] = (df['production_cost'] + df['marketing_budget']) / (
    df['ticket_price'] - (df['ticket_price'] * df['variable_cost_rate'])
)

# 4) 기획 단계에서는 실제 판매 관련 변수는 사용하지 않고,
# performance_tb에 있는 정적 변수만을 활용함  
# (accumulated_sales는 타깃 계산용으로만 사용하며, 입력 변수에서는 제외)
feature_cols = [
    'production_cost',
    'marketing_budget',
    'ticket_price',
    'capacity',
    'variable_cost_rate'
    # 기획 단계에서는 'accumulated_sales' 등 판매 실적 변수는 미사용
]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df[['ROI', 'BEP']].apply(pd.to_numeric, errors='coerce')

# 5) 전처리: 모든 피처가 수치형이므로 특별한 인코딩 없이 passthrough로 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', feature_cols)
    ]
)

# 6) XGBoost 모델 설정 (다중 출력 회귀를 위해 MultiOutputRegressor 사용)
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
multioutput_regressor = MultiOutputRegressor(xgb_regressor)

# 7) 파이프라인 구성
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', multioutput_regressor)
])

# 8) 학습 데이터 분할 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 9) 테스트 데이터 예측 및 평가 지표 계산
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

# 10) 모델 저장 (절대경로 사용)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgb_reg_roi_bep_planning.pkl')
print(">>> 모델이 'xgb_reg_roi_bep_planning.pkl' 로 저장되었습니다.")
