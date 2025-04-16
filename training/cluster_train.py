import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# 1) 데이터 불러오기
audience_df = pd.read_csv(r'C:\Users\USER\Desktop\my_git\ml-analysis\data\audience_tb.csv')

# 2) 군집에 사용할 피처 선정
# 원래의 피처: last_booking, booking_count, total_amount, age
# - last_booking은 날짜 문자열이므로, 기준 날짜와의 차이를 계산하여 숫자로 변환합니다.
feature_cols = [
    'last_booking',  # 날짜 문자열 (예: "2025-06-23")
    'booking_count',
    'total_amount',
    'age'
]
X = audience_df[feature_cols].copy()

# (예시) last_booking을 날짜형으로 변환한 후,
# 기준 날짜(여기서는 '2025-01-01')로부터 경과 일수(recent days)를 계산하여 새로운 컬럼 생성
X['last_booking'] = pd.to_datetime(X['last_booking'], errors='coerce')
reference_date = pd.Timestamp("2025-01-01")
X['recency_days'] = (reference_date - X['last_booking']).dt.days

# 원래의 last_booking 컬럼은 더 이상 필요 없으므로 삭제
X.drop('last_booking', axis=1, inplace=True)

# 이제 X에는 ['booking_count', 'total_amount', 'age', 'recency_days'] 컬럼만 남음

# 3) 전처리 + KMeans 파이프라인 구성
model = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=42))  # 군집 개수는 예시입니다.
])

# 4) 모델 훈련 (비지도 학습)
model.fit(X)

# 5) 군집 레이블 확인: kmeans의 레이블을 원본 데이터프레임에 추가
cluster_labels = model['kmeans'].labels_
audience_df['cluster'] = cluster_labels
print(audience_df[['booking_count', 'total_amount', 'age', 'cluster']].head())

# 6) 모델 저장 (models 폴더에 pickle 파일로 저장)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/kmeans_audience_seg.pkl')
print(">>> 모델이 'models/kmeans_audience_seg.pkl' 로 저장되었습니다.")
