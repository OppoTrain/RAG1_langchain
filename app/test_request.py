import requests

url = "http://127.0.0.1:8000/synthesize/"
payload = {"question":"What color is the sky at night if the moon were made of cheese?"}
try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print(f"Error: {response.status_code}, Details: {response.text}")
except Exception as e:
    print(f"An error occurred: {e}")
