from google import genai
import os

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

resp = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain what BDI measures in shipping, in one simple sentence."
)

print(resp.text)
