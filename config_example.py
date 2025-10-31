
# ---------------------------
# 🧠 OpenAI + Model Settings
# ---------------------------  
OPENAI_API_KEY = "your_openai_api_key"
MODEL_NAME = "gpt-4o-mini"              # or any GPT variant you’re using

# ---------------------------
# 🌐 Backend Integration
# ---------------------------
# For now, using local simulation. Replace with backend URL later.
BACKEND_URL = "backend_url"
INTENT_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/api/intent" 

# ---------------------------
# 🗣️ Voice & Language
# ---------------------------
DEFAULT_LANGUAGE = "en"
LANGUAGE_DETECTION_MODEL = "langdetect"
FALLBACK_MESSAGE = "I couldn't handle that request yet."
# ---------------------------
# 🌍 Supported Languages for TTS
# ---------------------------
SUPPORTED_LANG_CODES_FOR_TTS = {
    "en", "hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "ur"
}

# ---------------------------
# 💾 Database
# ---------------------------
DB_NAME = "memory.db"

# ---------------------------
# 🧪 Modes (for easy switching)
# ---------------------------
# If True → send to real backend
# If False → simulate backend reply
USE_REAL_BACKEND = False

# ---------------------------
# 🔊 Voice Assistant Settings
# ---------------------------