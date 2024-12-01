import requests

# Test /synthesize/
synthesize_response = requests.post(
    "http://127.0.0.1:8000/synthesize/",
    json={"question":"What protections are available under international law?"}
)
print("Synthesize Response:", synthesize_response.json())
