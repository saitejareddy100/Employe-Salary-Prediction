import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not api_key:
    print("ERROR: Gemini API key not found!")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents="Introduce yourself as the AI assistant of SalaryIQ."
        )

        print("Gemini connected successfully!")
        print(response.text)

    except Exception as e:
        print("Gemini connection failed:")
        print(e)