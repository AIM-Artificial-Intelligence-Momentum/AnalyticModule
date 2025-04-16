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

# 'performance_id' 컬럼 기준으로 병합
df = pd.merge(performance_df, sales_df, on='performance_id', how='inner')

# 2) ROI와 BEP 계산 컬럼 추가  
# --------------------------------------------------
# ROI = (총매출 - 총비용) / 총비용
#    = ((ticket_price * accumulated_sales) - (production_cost + marketing_budget 
#       + (ticket_price * variable_cost_rate * accumulated_sales)))
#      / (production_cost + marketing_budget + (ticket_price * variable_cost_rate * accumulated_sales))
#
# BEP = (production_cost + marketing_budget)
#       / (ticket_price - (ticket_price * variable_cost_rate))
# --------------------------------------------------
df['total_revenue'] = df['ticket_price'] * df['accumulated_sales']
df['total_cost'] = df['production_cost'] + df['marketing_budget'] + (
    df['ticket_price'] * df['variable_cost_rate'] * df['accumulated_sales']
)
df['ROI'] = (df['total_revenue'] - df['total_cost']) / df['total_cost']
df['BEP'] = (df['production_cost'] + df['marketing_budget']) / (
    df['ticket_price'] - (df['ticket_price'] * df['variable_cost_rate'])
)

# 3) 피처(X)와 목표(y) 설정  
# 여기서는 모두 수치형이어야 하므로, 필요한 경우 pd.to_numeric()로 형 변환
feature_cols = [
    'production_cost',
    'marketing_budget',
    'ticket_price',
    'capacity',
    'variable_cost_rate',
    'accumulated_sales'
]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df[['ROI', 'BEP']].apply(pd.to_numeric, errors='coerce')

# 4) 전처리  
# - 모든 피처는 수치형이므로 특별한 인코딩은 없지만, ColumnTransformer를 passthrough로 구성
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', feature_cols)
    ]
)

# 5) XGBoost 모델 설정 (다중 출력 회귀를 위해 MultiOutputRegressor 사용)
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
multioutput_regressor = MultiOutputRegressor(xgb_regressor)

# 6) 파이프라인 구성
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', multioutput_regressor)
])

# 7) 학습 데이터 분할 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 8) 테스트 데이터 예측 및 평가 지표 계산
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("[손익 예측: Multi-output XGBoost]")
print(f" - R² Score: {r2:.4f}")
print(f" - Mean Squared Error (MSE): {mse:.4f}")
print(f" - Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" - Mean Absolute Error (MAE): {mae:.4f}")

# 9) 모델 저장 (절대경로 사용)
# 6) 모델 저장 (models 폴더에 pickle 파일로 저장)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgb_reg_roi_bep_selling.pkl')
print(">>> 모델이 'xgb_reg_roi_bep_selling.pkl' 로 저장되었습니다.")