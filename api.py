import os
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from gtts import gTTS
import tempfile
import speech_recognition as sr
import re

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Resilio Emergency Support API"}

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'gu': 'Gujarati',
}

# ---- SMALL TALK RESPONSES ----
def get_small_talk_response(lang):
    responses = {
        'hi': "नमस्ते! रेसिलियो इमरजेंसी डिस्पैचर। आपकी स्थिति क्या है?",
        'ta': "வணக்கம்! ரெசிலியோ அவசரநிலை அனுப்பியவர். உங்கள் நிலை என்ன?",
        'gu': "નમસ્તે! રેસિલિઓ ઇમરજન્સી ડિસ્પેચર. તમારી પરિસ્થિતિ શું છે?",
        'en': "Resilio Emergency Dispatcher. State your emergency and location."
    }
    return responses.get(lang, responses['en'])

def is_small_talk(text):
    # Detect greeting/small talk keywords
    keywords = ['hello', 'hi', 'hey', 'namaste', 'vanakkam', 'kem cho']
    return any(p == text.lower().strip() for p in keywords)

# ---- TTS/STT Logic (Kept from original) ----
def text_to_speech(text, lang):
    try:
        clean_text = re.sub(r'[\*\#\[\]\(\)]', '', text) 
        tts = gTTS(text=clean_text, lang=lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception: return None

# ---- UPDATED AI PROMPT FOR RESILIO DISPATCHER ----
def build_prompt(user_query, chat_history, target_lang):
    language_name = SUPPORTED_LANGUAGES.get(target_lang, 'English')
    
    return f"""
You are **Resilio**, a digital emergency response assistant trained to help people during urgent and life-threatening situations.

━━━━━━━━━━━━━━━━━━━━
⚠️ URGENCY PROTOCOL (SAFETY CHECK)
━━━━━━━━━━━━━━━━━━━━
- If the user's input indicates an **active emergency** (e.g., "I'm trapped", "There is a fire", "Help me") or they ask for **precautions**:
  • You MUST immediately ask: "Are you in a safe location right now?"
  • Provide 3-4 immediate, actionable safety steps (e.g., "Stay low to avoid smoke").
  • End with: "Wait for professional rescuers. Do not attempt dangerous self-rescue."

- If the user is asking **general questions** about your purpose or features (e.g., "What are you used for?"):
  • Do NOT ask about their safety location.
  • Explain that you assist in reporting emergencies and provide life-saving precautions.
  • Keep the tone professional and informative.
━━━━━━━━━━━━━━━━━━━━
🌟 IDENTITY & PURPOSE
━━━━━━━━━━━━━━━━━━━━
If the user asks who you are or what you are used for:
- Identify yourself as **Resilio**. Explain that you assist in reporting emergencies like fires, floods, or medical crises and provide immediate safety precautions.
- Mention that you use AI to analyze incident reports to ensure that life-threatening situations receive the fastest possible response.

━━━━━━━━━━━━━━━━━━━━
⚠️ PRECAUTIONARY GUIDANCE
━━━━━━━━━━━━━━━━━━━━
If the user asks for "precautions," "what to do," or "how to stay safe":
- Provide 3-4 immediate, actionable safety steps specific to the disaster mentioned (e.g., "Stay low in smoke," "Turn off power in floods").
- ALWAYS end precautions with: "Wait for professional rescuers. Do not attempt dangerous self-rescue".

━━━━━━━━━━━━━━━━━━━━
🚨 CORE DIRECTIVE: RAPID TRIAGE
━━━━━━━━━━━━━━━━━━━━
- **Brevity is Life:** Use short, clear sentences. People in danger cannot process long text.
- **Extract Information:** Prioritize identifying: 1. The type of emergency, 2. The exact location, and 3. The presence of injuries.
- **Stay Calm:** Maintain a supportive, professional, and efficient tone.

━━━━━━━━━━━━━━━━━━━━
🚫 STRICT DOMAIN LIMITS
━━━━━━━━━━━━━━━━━━━━
- You ONLY discuss emergency reporting, safety precautions, and Resilio's features.
- If asked about unrelated topics (politics, entertainment, etc.):
  • State: "I am programmed only for emergency response support".
  • Ask: "Are you in a safe location right now?".

━━━━━━━━━━━━━━━━━━━━
🧠 CONTEXT (PREVIOUS TURNS)
━━━━━━━━━━━━━━━━━━━━
{chat_history}

User Input: "{user_query}"

━━━━━━━━━━━━━━━━━━━━
🎯 RESPONSE RULE
━━━━━━━━━━━━━━━━━━━━
Respond as **Resilio** in **{language_name}**. 
Keep the response under 3-4 sentences. Prioritize immediate life safety.
"""

# --- Configure Gemini ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash") 

class ChatRequest(BaseModel):
    message: str
    target_language: str
    chat_history: list = []
    wants_audio: bool = False
    is_voice: bool = False

@app.post("/chat")
async def chat(req: ChatRequest):
    user_query = req.message
    target_lang = req.target_language.lower()
    
    if is_small_talk(user_query):
        ai_resp = get_small_talk_response(target_lang)
    else:
        history_text = ""
        for m in req.chat_history[-5:]:
            role = "Victim" if m["role"] == "user" else "Dispatcher"
            history_text += f"{role}: {m['content']}\n"

        full_prompt = build_prompt(user_query, history_text, target_lang)
        
        try:
            response = model.generate_content(full_prompt)
            ai_resp = response.text
        except Exception:
            ai_resp = "I'm having trouble connecting right now. Stay calm and find a safe location."

    tts_audio = text_to_speech(ai_resp, target_lang) if req.wants_audio else None

    return {
        "text": ai_resp,
        "language": target_lang,
        "tts_audio": tts_audio
    }