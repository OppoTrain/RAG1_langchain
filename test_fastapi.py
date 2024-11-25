import requests

# Test /synthesize/
synthesize_response = requests.post(
    "http://127.0.0.1:8000/synthesize/",
    json={"question": "What rights are highlighted for women?"}
)
print("Synthesize Response:", synthesize_response.json())
