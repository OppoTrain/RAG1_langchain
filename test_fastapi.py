import requests

# Test /synthesize/
synthesize_response = requests.post(
    "http://127.0.0.1:8000/synthesize/",
    json={"question":"What are the effects of poverty on human rights?"}
)
print("Synthesize Response:", synthesize_response.json())
