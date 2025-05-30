import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:5050")
TO_NUMBER = os.getenv("TO_NUMBER")  # The number you want to call (e.g., +919xxxxxxxxx)

def make_call():
    if not TO_NUMBER:
        print("Please set TO_NUMBER in your .env file or as an environment variable.")
        return

    url = f"{FASTAPI_URL}/make-call"
    payload = {"to": TO_NUMBER}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Call initiated successfully!")
        print("Response:", response.json())
    except Exception as e:
        print("Failed to initiate call:", e)
        if hasattr(e, 'response') and e.response is not None:
            print("Response content:", e.response.text)

if __name__ == "__main__":
    make_call()
