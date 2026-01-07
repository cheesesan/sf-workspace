# ai/gemini.py
import os

def ask_gemini(prompt: str) -> str:
    """
    Simple Gemini wrapper.
    Requires env: GEMINI_API_KEY
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Please set it in .env or Streamlit Secrets.")

    # --- Option A: Google Generative AI (API Key mode) ---
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # 或 gemini-1.5-pro
    resp = model.generate_content(prompt)

    # 兼容性：有时 resp.text 为空，保险写法
    return getattr(resp, "text", "") or str(resp)
