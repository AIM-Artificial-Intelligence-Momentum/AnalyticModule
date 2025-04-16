import requests
import json

url = "http://127.0.0.1:8000/api/ml/accumulated_sales"
data = [
    {
        "genre": "뮤지컬",
        "region": "서울",
        "start_date_numeric": 1800,
        "capacity": 500,
        "star_power": 4,
        "ticket_price": 80000,
        "marketing_budget": 1000000,
        "sns_mention_count": 3000,
        "daily_sales": 100,
        "booking_rate": 65,
        "ad_exposure": 2000,
        "sns_mention_daily": 50
    }
]

headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers)

print("응답 코드:", response.status_code)
print("응답 텍스트:", response.text)
print("응답 내용:", response.json())