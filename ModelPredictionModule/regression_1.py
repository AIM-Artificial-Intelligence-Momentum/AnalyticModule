"""
공연 관객 수 예측 회귀 모델 모듈
- 저장된 RandomForest Regressor 모델을 불러와서 공연 관객 수 예측 수행
- 시각화 데이터 제공
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class PerformancePredictor:
    def __init__(self, model_path=None):
        """
        저장된 회귀 모델을 로드하여 공연 관객 수 예측을 수행하는 클래스
        
        Parameters:
        -----------
        model_path : str, optional
            저장된 모델 파일 경로, 기본값은 None
        """
        self.model = None
        # 경로는 환경에 맞게 조정 필요
        self.model_path = model_path or os.path.join("models", r"C:\Users\USER\Desktop\my_git\AIM 조직\IntegratedCode\backend\ModelPredictionModule\models\audience_rf_model.pkl")
        self.load_model()
    
    def load_model(self):
        """
        저장된 모델 파일 로드
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {self.model_path}")
        
        # 모델 로드
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"모델을 성공적으로 로드했습니다: {self.model_path}")
        return self.model
    
    def predict(self, features):
        """
        로드된 모델을 사용해 관객 수 예측 수행
        
        Parameters:
        -----------
        features : dict or DataFrame
            예측에 사용할 특성 데이터
            
        Returns:
        --------
        int
            예측 관객 수
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 딕셔너리를 DataFrame으로 변환
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
            
        # 예측 수행
        predictions = self.model.predict(features_df)
        return predictions.astype(int)  # 관객 수는 정수로 반환

def preprocess_performance_data(performance_data, sales_data=None):
    """
    공연 및 판매 데이터 전처리 함수
    
    Parameters:
    -----------
    performance_data : dict or DataFrame
        공연 정보 데이터
    sales_data : dict or DataFrame, optional
        판매 정보 데이터
        
    Returns:
    --------
    DataFrame
        전처리된 데이터셋
    """
    # 딕셔너리를 DataFrame으로 변환
    if isinstance(performance_data, dict):
        perf_df = pd.DataFrame([performance_data])
    else:
        perf_df = performance_data.copy()
    
    # 시작일 처리
    if 'start_date' in perf_df.columns:
        perf_df['start_date'] = pd.to_datetime(perf_df['start_date'])
        perf_df['days_since_start'] = (datetime.now() - perf_df['start_date']).dt.days
    
    # 여기서 추가적인 데이터 전처리 수행 가능
    # 범주형 변수 처리, 스케일링 등
    
    return perf_df

def predict_audience(performance_data, sales_data=None):
    """
    공연 관객 수 예측 및 시각화 데이터 생성 함수
    
    Parameters:
    -----------
    performance_data : dict
        공연 정보 (genre, region, start_date, capacity, star_power, ticket_price, marketing_budget, sns_mention_count)
    sales_data : dict, optional
        판매 정보 (daily_sales, booking_rate, ad_exposure, sns_mention_daily)
    
    Returns:
    --------
    dict
        예측 결과와 시각화 데이터를 포함한 딕셔너리
    """
    try:
        # 데이터 전처리
        processed_data = preprocess_performance_data(performance_data, sales_data)
        
        # 예측기 초기화 및 모델 로드
        predictor = PerformancePredictor()
        
        # 예측 수행
        predicted_audience = predictor.predict(processed_data)[0]
        
        # 누적 관객 추이 데이터 생성
        start_date = pd.to_datetime(performance_data.get('start_date', datetime.now() - timedelta(days=30)))
        current_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=current_date, freq='D')
        
        # 누적 판매 시뮬레이션 (로지스틱 곡선)
        days = (date_range - start_date).days.values
        accumulated_sales = [
            int(predicted_audience * (1 / (1 + np.exp(-0.15 * (day - 10)))))
            for day in days
        ]
        
        # 유사 공연 데이터 (예시)
        similar_performances = [
            {"name": "공연 A", "genre": performance_data.get('genre', ''), "actual_audience": int(predicted_audience * 0.85)},
            {"name": "공연 B", "genre": performance_data.get('genre', ''), "actual_audience": int(predicted_audience * 1.2)},
            {"name": "공연 C", "genre": performance_data.get('genre', ''), "actual_audience": int(predicted_audience * 0.95)}
        ]
        
        # 결과 구성
        result = {
            'prediction': int(predicted_audience),
            'status': 'success',
            'visualization_data': {
                'accumulated_sales_chart': {
                    'dates': [d.strftime('%Y-%m-%d') for d in date_range],
                    'accumulated_sales': accumulated_sales
                },
                'capacity_sales_scatter': {
                    'capacity': [performance_data.get('capacity', 500)],
                    'accumulated_sales': [accumulated_sales[-1] if accumulated_sales else 0]
                },
                'similar_performances': similar_performances
            }
        }
        
        return result
    
    except Exception as e:
        # 오류 발생 시 처리
        print(f"오류 발생: {e}")
        
        return {
            'prediction': None,
            'status': 'error',
            'message': str(e)
        }

def get_visualization_data(prediction_result):
    """
    시각화에 필요한 데이터만 추출
    
    Parameters:
    -----------
    prediction_result : dict
        예측 결과 및 시각화 데이터
        
    Returns:
    --------
    dict
        시각화용 데이터
    """
    if prediction_result['status'] != 'success':
        return {'error': prediction_result.get('message', '알 수 없는 오류')}
    
    return prediction_result['visualization_data']

# 직접 실행 시 사용 예시
if __name__ == "__main__":
    # 예시 공연 데이터
    example_performance = {
        'genre': '뮤지컬',
        'region': '서울',
        'start_date': '2025-03-01',
        'capacity': 800,
        'star_power': 85,
        'ticket_price': 65000,
        'marketing_budget': 50000000,
        'sns_mention_count': 12500
    }
    
    # 예측 실행
    result = predict_audience(example_performance)
    
    # 결과 출력
    if result['status'] == 'success':
        print(f"예측된 총 관객 수: {result['prediction']}")
        
        # 시각화 데이터 출력 (일부만)
        viz_data = result['visualization_data']
        print("\n시각화 데이터 예시:")
        print(f"누적 판매 날짜 수: {len(viz_data['accumulated_sales_chart']['dates'])}")
        print(f"누적 판매 최종 값: {viz_data['accumulated_sales_chart']['accumulated_sales'][-1]}")
        print(f"유사 공연 수: {len(viz_data['similar_performances'])}")
    else:
        print(f"예측 실패: {result['message']}")