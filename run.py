import requests

url = "https://sentence-classifier-latest.onrender.com/predict"

data = {
    "text": "I am going to office"
}

response = requests.post(url, json=data)
#
for i in response.json().values():
    print(i)