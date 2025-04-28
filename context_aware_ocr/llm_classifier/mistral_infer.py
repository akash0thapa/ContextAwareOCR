import requests
import json

url = "http://172.20.24.175:1234/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

data = {
    "model": "mathstral-7b-v0.1",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Nepal?"}
    ],
    "temperature": 0.2,
    "max_tokens": 100
}

response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code)
    print(response.text)
