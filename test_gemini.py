# test_gemini.py
# Verifies Gemini API access using the new google-genai SDK

import os
from dotenv import load_dotenv
from google import genai

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    exit()

print(f"API key loaded (length: {len(api_key)} chars)")

# Create the Gemini client
client = genai.Client(api_key=api_key)

print("\nAsking Gemini a test question...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="In one sentence, what is the difference between yield strength and ultimate tensile strength?"
)

print("\n--- Gemini's response ---")
print(response.text)
print("--- End ---")

print("\nGemini is working correctly.")