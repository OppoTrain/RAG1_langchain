import requests

# Test /retrieve/
retrieve_response = requests.post(
    "http://127.0.0.1:8000/retrieve/",
    json={"question": "What rights are highlighted for women?"}
)
print("Retrieve Response:", retrieve_response.json())

# Test /synthesize/
synthesize_response = requests.post(
    "http://127.0.0.1:8000/synthesize/",
    json={"question": "What rights are highlighted for women?"}
)
print("Synthesize Response:", synthesize_response.json())
