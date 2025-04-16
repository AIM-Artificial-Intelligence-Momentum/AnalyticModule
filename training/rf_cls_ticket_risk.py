import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# 1) 데이터 불러오기 (절대경로 사용)
performance_df = pd.read_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\performance_tb.csv')
sales_df = pd.read_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\sales_tb.csv')

# 2) 'performance_id' 컬럼 기준으로 두 데이터프레임 병합
df = pd.merge(performance_df, sales_df, on='performance_id', how='inner')

# 병합 후 컬럼 확인 (문제 발생 시 아래 출력으로 확인)
print("병합 후 컬럼:", df.columns.tolist())

# 3) start_date 컬럼 전처리
# 만약 performance_df와 sales_df 둘 다 날짜 컬럼을 가지고 있다면, 이름이 달라질 수 있으므로 여기서는
# performance_df의 start_date를 사용한다고 가정합니다.
if 'start_date' in df.columns:
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    reference_date = pd.Timestamp("2020-01-01")
    df['start_date_numeric'] = (df['start_date'] - reference_date).dt.days
    # 문자열 형태의 start_date 컬럼 삭제 (날짜 문자열이 남아 있지 않도록)
    df.drop(columns=['start_date'], inplace=True)
else:
    print("start_date 컬럼이 존재하지 않습니다. 데이터셋을 확인하세요.")

# 4) '위험도' 라벨 생성
# booking_rate >= 75: 저위험(0), 60 <= booking_rate < 75: 중위험(1), booking_rate < 60: 고위험(2)
def classify_risk(rate):
    if rate >= 75:
        return 0  # 저위험
    elif rate >= 60:
        return 1  # 중위험
    else:
        return 2  # 고위험

df['risk_label'] = df['booking_rate'].apply(classify_risk)

# 5) 피처와 타깃 정의 - 변환된 날짜 컬럼(start_date_numeric)만 사용
feature_cols = [
    'genre', 
    'region', 
    'start_date_numeric',  # 변환한 숫자형 날짜 컬럼
    'capacity', 
    'star_power', 
    'daily_sales', 
    'accumulated_sales', 
    'ad_exposure', 
    'sns_mention_daily', 
    'promo_event_flag'
]
X = df[feature_cols]
y = df['risk_label']

# 6) 범주형/수치형 데이터 전처리
categorical_features = ['genre', 'region', 'promo_event_flag']
numerical_features = [col for col in feature_cols if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# 7) 모델 구성 및 학습
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 8) 평가 및 모델 저장
accuracy = model.score(X_test, y_test)
print(f"[티켓 판매 위험 예측] Accuracy: {accuracy:.4f}")

# 모델 저장 경로 (절대경로 사용)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_cls_ticket_risk.pkl')
print(">>> 모델이 'rf_cls_ticket_risk.pkl' 로 저장되었습니다.")

# 추가: ground truth 데이터셋 저장 (평가용)
df[['risk_label']].to_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\ground_truth.csv', index=False)
print(">>> ground_truth.csv 파일에 평가용 레이블 데이터가 저장되었습니다.")