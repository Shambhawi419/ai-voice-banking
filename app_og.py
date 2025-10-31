# voice_banking_assistant_multilingual.py
# Purpose: voice-only multilingual banking assistant that:
#  - accepts spoken queries in Indian languages
#  - transcribes using OpenAI Whisper
#  - classifies intent/entities using OpenAI (strict JSON output)
#  - fetches user data from a backend URL (configured in config.py)
#  - sends intent+context+user-data to the backend intent route as JSON
#  - speaks backend's final response in the user's language (via TTS)
# IMPORTANT: this implementation does NOT hardcode business rules. All decisions
# are performed by your backend. The assistant only orchestrates speech, AI
# classification/translation, and HTTP calls to your backend.

import os
import json
import time
import traceback
import requests
import logging
from typing import Any, Dict

import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
from playsound import playsound
from langdetect import detect

from config import (
    OPENAI_API_KEY,
    BACKEND_URL,
    INTENT_ENDPOINT,
    SUPPORTED_LANG_CODES_FOR_TTS,
    DEFAULT_LANGUAGE,
    USE_REAL_BACKEND,
    AUDIO_TEMP_FILE,
    RESPONSE_AUDIO_FILE,
    MAX_CONTEXT_MESSAGES,
)
from database import (
    init_db,
    create_user_profile,
    get_user_profile,
    save_message,
    get_recent_context,
    start_session,
    end_session,
)

# --- Init ---
client = OpenAI(api_key=OPENAI_API_KEY)
init_db()
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Utilities
# ---------------------------

def speak_text(text: str, lang: str = DEFAULT_LANGUAGE) -> None:
    """Synthesize and play speech. Uses gTTS and playsound. """
    try:
        tts = gTTS(text=str(text), lang=lang, slow=False)
    except Exception:
        # fallback to english
        try:
            tts = gTTS(text=str(text), lang='en', slow=False)
        except Exception as e:
            logging.error("TTS generation failed: %s", e)
            return

    tmp = RESPONSE_AUDIO_FILE or "response.mp3"
    try:
        tts.save(tmp)
        playsound(tmp)
    except Exception as e:
        logging.error("TTS playback/save failed: %s", e)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def transcribe_audio_file(path: str) -> str:
    """Transcribe audio using OpenAI Whisper. Returns text or empty string."""
    try:
        with open(path, "rb") as f:
            res = client.audio.transcriptions.create(model="whisper-1", file=f)
        return getattr(res, 'text', '').strip() if res else ""
    except Exception as e:
        logging.error("Transcription failed: %s", e)
        return ""


def detect_language_safe(text: str, fallback: str = DEFAULT_LANGUAGE) -> str:
    try:
        lang = detect(text)
        return lang
    except Exception:
        return fallback


# ---------------------------
# OpenAI classification & translation helpers
# ---------------------------

def build_system_prompt_for_classification() -> str:
    """
    Builds a comprehensive system prompt for the AI classifier.
    This version includes expanded banking intents and support for Hindi/Hinglish queries.
    """
    return """
You are an AI language understanding module for a voice-enabled banking assistant.
Your job is to analyze a user's message and classify it into a structured JSON object.

Users may speak in English, Hindi, Hinglish, or mixed regional expressions.
Interpret colloquial or informal phrases like "paisa bhejna", "balance check karna",
"loan nahi mila", "ATM block ho gaya", "EMI kab due hai", etc., appropriately
and always produce a clean structured JSON.

### Output Format
Return output strictly in this format:
{
  "intent": "<one_of_the_intents_below>",
  "language": "<en|hi|mixed>",
  "details": { key-value pairs with extracted entities if any }
}

### Supported Intents
<intents>
check_balance |
make_payment | fund_transfer | deposit_money | withdraw_money |
view_transactions | mini_statement |
loan_inquiry | loan_status | loan_reason_inquiry | loan_eligibility |
loan_interest_rate | apply_loan |
emi_status | emi_payment | interest_query |
open_account | close_account | account_closure |
kyc_status | update_kyc |
branch_info | ifsc_lookup | atm_locator |
cheque_status | block_cheque | stop_payment |
card_block | card_unblock | card_replacement |
credit_card_status | credit_limit |
debit_card_pin_reset | credit_card_bill_payment |
fraud_report | dispute_transaction | lost_card_report |
fixed_deposit_info | open_fd | close_fd |
recurring_deposit_info | open_rd | close_rd |
insurance_status | buy_insurance | claim_insurance |
currency_exchange | international_transfer | statement_request |
complaint_registration | feedback_submission | unknown
</intents>

### Notes:
- Output must be valid, parseable JSON — no explanations or extra text.
- If the intent is unclear, classify it as "unknown".
- Use "details" to include relevant extracted entities such as:
  - amount
  - account_number or last4
  - recipient name
  - loan_type
  - date or range (for transaction inquiries)
  - location (for branch/ATM queries)
  - reason_type (e.g., insufficient_balance, missing_kyc)
- If the detected language is primarily Hindi, return "hi"; for mixed Hindi-English, return "mixed".
"""



def classify_intent_and_entities(user_text: str, detected_lang: str, max_retries: int = 3) -> Dict[str, Any]:
    prompt = build_system_prompt_for_classification()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Language: {detected_lang}\nUtterance: {user_text}"}
                ],
                temperature=0.0,
                max_tokens=450,
            )
            raw = resp.choices[0].message.content.strip()
            # extract JSON substring
            first = raw.find('{')
            last = raw.rfind('}')
            if first != -1 and last != -1:
                raw = raw[first:last+1]
            parsed = json.loads(raw)
            return {
                "intent": parsed.get('intent', 'unknown'),
                "language": parsed.get('language', detected_lang),
                "details": parsed.get('details', {}) or {}
            }
        except Exception as e:
            logging.warning("Classification attempt %s failed: %s", attempt + 1, e)
            time.sleep(0.3)
            continue
    return {"intent": "unknown", "language": detected_lang, "details": {}}


def translate_text_via_openai(text: str, target_lang: str, source_lang: str = 'en') -> str:
    """Translate assistant text to target_lang using the model. Keeps numbers unchanged."""
    if not target_lang or target_lang.startswith(source_lang):
        return text
    try:
        prompt = f"Translate the following into {target_lang}. Keep numeric values unchanged.\n\nText:\n{text}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error("Translation failed: %s", e)
        return text

# ---------------------------
# Backend communication
# ---------------------------

def fetch_user_data_from_backend(user_id: str) -> Dict[str, Any]:
    """GET user data from backend. Expects endpoint: /api/users/<user_id>"""
    try:
        url = f"{BACKEND_URL.rstrip('/')}/api/users/{user_id}"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            return resp.json()
        else:
            logging.error("User data fetch returned %s", resp.status_code)
            return {}
    except Exception as e:
        logging.error("Failed to fetch user data: %s", e)
        return {}


def send_intent_payload_to_backend(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST classified intent + context to INTENT_ENDPOINT. Backend makes decisions and returns final message."""
    try:
        if not USE_REAL_BACKEND:
            # attempt call anyway — user requested backend-driven logic. If backend offline, return informative error.
            logging.info("USE_REAL_BACKEND is False; still attempting to POST to backend for integration testing.")
        resp = requests.post(INTENT_ENDPOINT, json=payload, timeout=8)
        if resp.status_code == 200:
            return resp.json()
        else:
            logging.error("Intent endpoint returned %s", resp.status_code)
            return {"status": "error", "message": f"Backend error {resp.status_code}"}
    except Exception as e:
        logging.error("Failed to contact backend intent endpoint: %s", e)
        return {"status": "error", "message": "Backend unavailable. Please ensure the backend server is running at BACKEND_URL."}
def handle_flexible_response(response):
    """
    Dynamically handle any kind of backend response
    without hardcoding JSON structure.
    """
    if not isinstance(response, dict):
        print("Unexpected response type:", response)
        return

    if "message" in response:
        print(response["message"])

    elif "user_data" in response:
        print("\n User Data:")
        for key, value in response["user_data"].items():
            if isinstance(value, list):
                print(f"\n{key.capitalize()}:")
                for item in value:
                    print("  • " + ", ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                print(f"{key.capitalize()}: {value}")

    elif response.get("status") == "error":
        print(f" Error: {response.get('message', 'Unknown error')}")

    else:
        print("Unrecognized response format:", response)


# ---------------------------
# Main voice loop
# ---------------------------

def main():
    user_id = os.getenv('USER_ID') or input('Enter user ID (for local test): ').strip()
    if not user_id:
        print('User ID required.')
        return

    # Fetch or create profile locally for UI touches (language preference stored here)
    profile = get_user_profile(user_id)
    if not profile:
        # create with default language; assume user authenticated on site earlier and backend knows them
        create_user_profile(user_id, name=user_id.split('@')[0], language=DEFAULT_LANGUAGE, tone='neutral')
        profile = get_user_profile(user_id)

    preferred_lang = profile.get('language', DEFAULT_LANGUAGE)

    # Start session
    start_session(user_id)

    # Greet user in preferred language
    greeting = 'नमस्ते। कैसे मदद करूँ?' if preferred_lang.startswith('hi') else f'Hello {profile.get("name","")}. How can I help you?'
    speak_text(greeting, lang=preferred_lang)

    recognizer = sr.Recognizer()

    try:
        while True:
            with sr.Microphone() as source:
                logging.info('Listening...')
                audio = recognizer.listen(source, phrase_time_limit=20)

            # save temporary file
            with open(AUDIO_TEMP_FILE, 'wb') as f:
                f.write(audio.get_wav_data())

            # Transcribe
            user_text = transcribe_audio_file(AUDIO_TEMP_FILE)
            # remove temp file
            try:
                if os.path.exists(AUDIO_TEMP_FILE):
                    os.remove(AUDIO_TEMP_FILE)
            except Exception:
                pass

            if not user_text:
                speak_text("I did not hear you. Please repeat.", lang=preferred_lang)
                continue

            logging.info('User said: %s', user_text)

            # Exit phrases (multilingual common phrases)
            if user_text.strip().lower() in ['exit', 'quit', 'stop', 'bye', 'goodbye', 'band karo', 'nikal jao', 'phir milenge']:
                speak_text('Goodbye.' if preferred_lang.startswith('en') else 'अलविदा।', lang=preferred_lang)
                break

            # Detect language
            detected_lang = detect_language_safe(user_text, fallback=preferred_lang)
            if detected_lang not in SUPPORTED_LANG_CODES_FOR_TTS:
                detected_lang = preferred_lang

            # Save user utterance
            save_message(user_id, 'user', user_text)

            # Get recent context stored locally
            context_msgs = get_recent_context(user_id, limit=MAX_CONTEXT_MESSAGES)

            # Classify intent and extract entities
            classification = classify_intent_and_entities(user_text, detected_lang)
            logging.info('Classification: %s', classification)

            # Fetch authoritative user data from backend
            user_data = fetch_user_data_from_backend(user_id)

            # Build payload to send to backend intent endpoint
            payload = {
                'user_id': user_id,
                'intent': classification.get('intent', 'unknown'),
                'language': classification.get('language', detected_lang),
                'details': classification.get('details', {}),
                'conversation_context': context_msgs,
                'user_data': user_data  # provide backend with last-known state from DB
            }

            # Send to backend for final decision and message
            backend_response = send_intent_payload_to_backend(payload)
            logging.info('Backend response: %s', backend_response)
            handle_flexible_response(backend_response)

            final_message = backend_response.get('message') if isinstance(backend_response, dict) else str(backend_response)
            if not final_message:
                final_message = 'Sorry, I could not complete that request right now.'

            # Translate the backend message to the user's language if needed
            if detected_lang and not detected_lang.startswith('en'):
                try:
                    final_message_translated = translate_text_via_openai(final_message, target_lang=detected_lang, source_lang='en')
                except Exception:
                    final_message_translated = final_message
            else:
                final_message_translated = final_message

            # Save and speak
            save_message(user_id, 'assistant', final_message_translated)
            speak_text(final_message_translated, lang=detected_lang)

    except KeyboardInterrupt:
        logging.info('Interrupted by user.')
    except Exception as e:
        logging.error('Unexpected error: %s', traceback.format_exc())
    finally:
        end_session(user_id)


if __name__ == '__main__':
    main()
