# sf-workspace/ai/gemini.py
import os

def ask_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ GEMINI_API_KEY (or GOOGLE_API_KEY) is not set."

    # ===== New SDK: google-genai =====
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

        last_err = None
        for m in models:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=prompt,
                )
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                return str(resp)
            except Exception as e:
                last_err = e

        return f"⚠️ Gemini failed. Tried {models}. Last error: {last_err}"

    except Exception as e:
        return f"⚠️ Gemini SDK error: {type(e).__name__}: {e}" 
