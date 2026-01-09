# ai/gemini.py
import os
import google.generativeai as genai
from google import genai


def _pick_model(prefer_keywords=("flash", "gemini")) -> str:
    """
    Pick a model that supports generateContent.
    Prefer 'flash' models if available.
    """
    models = list(genai.list_models())

    # only models that can do generateContent
    can_generate = [
        m for m in models
        if hasattr(m, "supported_generation_methods")
        and m.supported_generation_methods
        and ("generateContent" in m.supported_generation_methods)
    ]
    if not can_generate:
        # fallback: just return something (will error with better message)
        return "gemini-1.5-flash"

    # prefer flash
    def score(m):
        name = getattr(m, "name", "").lower()  # e.g. "models/gemini-1.5-flash-latest"
        s = 0
        for kw in prefer_keywords:
            if kw in name:
                s += 10
        if "flash" in name:
            s += 30
        if "pro" in name:
            s += 10
        return s

    can_generate.sort(key=score, reverse=True)
    return can_generate[0].name  # keep full name like "models/xxx"


def ask_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ GEMINI_API_KEY (or GOOGLE_API_KEY) is not set."

    genai.configure(api_key=api_key)

    # ✅ auto-pick a working model
    model_name = _pick_model()

    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or str(resp)
    except Exception as e:
        # add more debug info
        return f"⚠️ Gemini call failed. model={model_name} error={type(e).__name__}: {e}"
