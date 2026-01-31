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
    return {"status": "ok", "service": "VIBE-CHECK Student Wellness API"}

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'gu': 'Gujarati',
}

# ---- SMALL TALK RESPONSES ----
def get_small_talk_response(lang):
    responses = {
        'hi': "नमस्ते! आप आज कैसा महसूस कर रहे हैं?",
        'ta': "வணக்கம்! இன்று உங்கள் மனநிலை எப்படி இருக்கிறது?",
        'gu': "નમસ્તે! આજે તમે કેવું અનુભવો છો?",
        'en': "Hi there! How is your mood today?"
    }
    return responses.get(lang, responses['en'])

def is_small_talk(text):
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

# ---- UPDATED AI PROMPT FOR STUDENT WELLNESS ----
def build_prompt(user_query, chat_history, target_lang):
    language_name = SUPPORTED_LANGUAGES.get(target_lang, 'English')
    
    return f"""
You are **VIBE-CHECK**, a supportive, grounded, and intuitive AI companion. 

━━━━━━━━━━━━━━━━━━━━
🚀 PROACTIVE ENGAGEMENT RULES (THE MISSING PIECES)
━━━━━━━━━━━━━━━━━━━━
- **Acknowledge the Silence:** If the user's input is very short (e.g., "idk", "...", "maybe"), don't just ask why. Say: "Sometimes words are hard to find. I'm just here to sit in the quiet with you until you're ready."
- **Direct Answers for Wellness:** If a student asks for advice (e.g., "How do I stop panicking?"), give ONE clear, immediate technique (like Box Breathing) BEFORE asking how they feel. Don't withhold help.
- **Avoid "I'm sorry" Overuse:** Instead of saying "I am sorry you feel that way" (which feels robotic), say "That sounds exhausting" or "I can see why that would be frustrating." 
- **Handle "I don't know":** When a student says they don't know what's wrong, validate that. Tell them: "That's actually very common. Sometimes stress is just a heavy cloud without a specific name."

━━━━━━━━━━━━━━━━━━━━
🛡️ SMART REFUSAL (NO HALLUCINATIONS)
━━━━━━━━━━━━━━━━━━━━
- If the user asks a non-wellness question (e.g., "Who won the game?" or "Do my homework"):
    👉 Be a helpful peer, not a strict teacher. Say: "I'd love to chat about that, but my 'brain' is currently tuned specifically to help you manage student stress and stay balanced. Want to talk about how your day is actually going instead?"

━━━━━━━━━━━━━━━━━━━━
⚖️ BALANCE RULE
━━━━━━━━━━━━━━━━━━━━
- **Mirror the User:** If the user writes a long, emotional message, give a thoughtful, long response. If they write a short message, keep yours brief and punchy.
- **One Question Max:** Only ask one question at the very end, and ONLY if it feels natural to keep the conversation flowing.

━━━━━━━━━━━━━━━━━━━━
🧠 RECENT CONTEXT
━━━━━━━━━━━━━━━━━━━━
{chat_history}

Student: "{user_query}"

Provide a warm, intuitive response in **{language_name}** that feels like a supportive friend.
"""



# --- Configure Gemini ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash") # Updated to 1.5 Flash as 2.5 is not a public version yet

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
            role = "Student" if m["role"] == "user" else "Vibe-Check"
            history_text += f"{role}: {m['content']}\n"

        full_prompt = build_prompt(user_query, history_text, target_lang)
        
        # Check if build_prompt returned the hardcoded SOS response
        if "Helpline:" in full_prompt and len(full_prompt) < 300:
            ai_resp = full_prompt
        else:
            try:
                response = model.generate_content(full_prompt)
                ai_resp = response.text
            except Exception:
                ai_resp = "I'm having trouble connecting right now. Take a deep breath, and let's try again in a moment."

    tts_audio = text_to_speech(ai_resp, target_lang) if req.wants_audio else None

    return {
        "text": ai_resp,
        "language": target_lang,
        "tts_audio": tts_audio
    }