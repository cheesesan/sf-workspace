from google import genai
import os
from google.genai import errors

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

def ask_gemini(prompt: str, model: str = "models/gemini-flash-lite-latest") -> str:
    try:
        client = get_client()
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()
    except errors.ClientError as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return "⚠️ Gemini API quota/billing issue. Please check plan & billing."
        return f"⚠️ Gemini API error: {msg}"
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"
