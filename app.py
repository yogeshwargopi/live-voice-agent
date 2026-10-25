"""Local voice companion for English, Hindi, Tamil, and Telugu."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

CHAT_MODEL = "gpt-4.1-mini"
TRANSCRIBE_MODEL = "gpt-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "marin"

NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
}
ISO3 = {"eng": "en", "hin": "hi", "tam": "ta", "tel": "te"}

MISSED = {
    "en": "I didn't catch that. Could you say it once more?",
    "hi": "मुझे सुनाई नहीं दिया। एक बार फिर बोलोगे?",
    "ta": "எனக்குக் கேட்கவில்லை. இன்னொரு முறை சொல்லுங்க?",
    "te": "నాకు వినిపించలేదు. మళ్ళీ ఒకసారి చెప్తారా?",
}
FALLBACK = {
    "en": "I'm here. Say that once more?",
    "hi": "मैं यहीं हूँ। एक बार फिर बताओगे?",
    "ta": "நான் இங்கே இருக்கேன். இன்னொரு முறை சொல்லுங்க?",
    "te": "నేను ఇక్కడే ఉన్నాను. మళ్ళీ ఒకసారి చెప్తారా?",
}
UNSUPPORTED = (
    "I can chat in English, Hindi, Tamil, and Telugu. "
    "Try one of those, and I'll be right here."
)
TTS_INSTRUCTIONS = {
    "en": "Speak English in a warm, friendly, unhurried voice, like a close friend on a phone call.",
    "hi": "Speak Hindi naturally in a warm, friendly, unhurried voice, like a close friend. Use a clear Indian accent.",
    "ta": "Speak Tamil naturally in a warm, friendly, unhurried voice, like a close friend. Use a clear Tamil accent.",
    "te": "Speak Telugu naturally in a warm, friendly, unhurried voice, like a close friend. Use a clear Telugu accent.",
}
SYSTEM_PROMPT = """You are a warm friend on a voice call. You only talk. You do not browse, search, set reminders, send messages, write code, or do tasks for the person.

Speak in 1 to 3 short sentences, the way a friend talks. Use everyday words. No lists, no markdown, no headings, no emojis.

You can talk in only four languages: English, Hindi, Tamil, and Telugu.
- English uses Latin letters.
- Hindi uses Devanagari.
- Tamil uses Tamil script.
- Telugu uses Telugu script.

Follow the language of the person's latest message. If they mix languages, follow the language they are using now. If they speak Hindi, Tamil, or Telugu using Latin letters, still answer in that language's own script.

If they ask you to do a task, kindly say you are just here to talk, then stay with them in the conversation."""

MISSING_KEY = "Add your OpenAI API key to the .env file, then refresh this page."
BAD_KEY = "That OpenAI API key was rejected. Check the key in .env and restart."
MAX_AUDIO_BYTES = 15 * 1024 * 1024
MIN_AUDIO_BYTES = 800
MAX_TURNS = 20
MAX_SESSIONS = 100

log = logging.getLogger("friend")

app = FastAPI(title="Friend")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@dataclass
class Session:
    turns: list[dict[str, str]] = field(default_factory=list)
    generation: int = 0


sessions: OrderedDict[str, Session] = OrderedDict()


def api_key() -> str:
    load_dotenv(ROOT / ".env", override=True)
    return os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")


def make_client() -> OpenAI | None:
    key = api_key()
    if not key:
        return None
    return OpenAI(api_key=key, timeout=60.0)


def fail(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers={"Cache-Control": "no-store"})


def language_name(code: str | None) -> str:
    if code in NAMES:
        return NAMES[code]
    return "Another language"


def normalize_code(code: str) -> str:
    cleaned = code.strip().lower().replace("_", "-")
    if not cleaned:
        return ""
    if cleaned in ISO3:
        return ISO3[cleaned]
    primary = cleaned.split("-", 1)[0]
    return ISO3.get(primary, primary)


def script_language(text: str) -> str | None:
    counts = {
        "ta": len(re.findall(r"[\u0B80-\u0BFF]", text)),
        "te": len(re.findall(r"[\u0C00-\u0C7F]", text)),
        "hi": len(re.findall(r"[\u0900-\u097F]", text)),
        "en": len(re.findall(r"[A-Za-z]", text)),
    }
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return None
    return best


def choose_language(text: str, api_codes: list[str], locked: str | None) -> str | None:
    if locked in NAMES:
        return locked
    normalized = [normalize_code(code) for code in api_codes]
    normalized = [code for code in normalized if code]
    supported = [code for code in normalized if code in NAMES]
    if supported:
        return supported[0]
    if normalized:
        return normalized[0]
    return script_language(text)


def clean_reply(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^(friend|assistant)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def shorten(text: str, limit: int = 600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for mark in ("।", ".", "?", "!"):
        index = cut.rfind(mark)
        if index > 80:
            return cut[: index + 1].strip()
    return cut.strip()


def valid_session(session_id: str) -> str | None:
    session_id = (session_id or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", session_id):
        return session_id
    return None


def get_session(session_id: str) -> Session:
    if session_id in sessions:
        sessions.move_to_end(session_id)
        return sessions[session_id]
    if len(sessions) >= MAX_SESSIONS:
        sessions.popitem(last=False)
    session = Session()
    sessions[session_id] = session
    return session


def remember(session: Session, generation: int, user_text: str, assistant_text: str) -> None:
    if session.generation != generation:
        return
    session.turns.append({"role": "user", "content": user_text})
    session.turns.append({"role": "assistant", "content": assistant_text})
    extra = len(session.turns) - MAX_TURNS * 2
    if extra > 0:
        del session.turns[:extra]


def audio_name(upload: UploadFile) -> str:
    name = (upload.filename or "").lower()
    for ext in (".webm", ".m4a", ".mp4", ".wav", ".mp3", ".ogg"):
        if name.endswith(ext):
            return "speech.m4a" if ext == ".mp4" else f"speech{ext}"
    content = (upload.content_type or "").lower()
    if "mp4" in content or "m4a" in content or "aac" in content:
        return "speech.m4a"
    if "wav" in content:
        return "speech.wav"
    if "mpeg" in content or "mp3" in content:
        return "speech.mp3"
    if "ogg" in content:
        return "speech.ogg"
    return "speech.webm"


def read_transcript(result: object) -> tuple[str, list[str]]:
    text = (getattr(result, "text", "") or "").strip()
    codes: list[str] = []
    for item in getattr(result, "languages", None) or []:
        code = getattr(item, "code", None)
        if code is None and isinstance(item, dict):
            code = item.get("code")
        if code:
            codes.append(str(code))
    return text, codes


def transcribe_audio(client: OpenAI, audio_bytes: bytes, filename: str, languages: list[str]) -> tuple[str, list[str]]:
    def once(use_languages: bool) -> object:
        audio = BytesIO(audio_bytes)
        audio.name = filename
        if use_languages:
            return client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=audio,
                languages=languages,
            )
        return client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=audio)

    try:
        result = once(bool(languages))
    except APIStatusError as exc:
        if exc.status_code != 400 or "language" not in str(exc).lower():
            raise
        log.warning("transcription rejected the language hint; retrying without it")
        result = once(False)
    return read_transcript(result)


def system_prompt(language: str) -> str:
    name = NAMES[language]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"The person is speaking {name} right now. "
        f"Reply only in {name}, using its native script. "
        "One to three short spoken sentences."
    )


def complete_chat(client: OpenAI, history: list[dict[str, str]], user_text: str, language: str) -> str:
    messages = [{"role": "system", "content": system_prompt(language)}, *history, {"role": "user", "content": user_text}]
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=220,
    )
    reply = shorten(clean_reply(completion.choices[0].message.content or ""))
    return reply or FALLBACK[language]


def speak(client: OpenAI, text: str, language: str) -> bytes:
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=TTS_INSTRUCTIONS[language],
        response_format="mp3",
    )
    return speech.read()


def api_error_message(exc: APIStatusError) -> tuple[int, str]:
    status = exc.status_code or 502
    if status in (401, 403):
        return 401, BAD_KEY
    if status == 429:
        return 429, "OpenAI is busy right now. Wait a moment and try again."
    if status == 404:
        return 502, "OpenAI could not find a speech model for this key. Check that the key can use the API."
    return 502, "Something went wrong talking to OpenAI. Try again."


def result_body(
    user_text: str,
    assistant_text: str,
    reply_language: str,
    user_language: str | None,
    audio: bytes | None,
    error: str | None = None,
) -> dict[str, object]:
    heard = user_language or reply_language
    body: dict[str, object] = {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "language": reply_language,
        "language_name": language_name(reply_language) if reply_language in NAMES else "English",
        "user_language": heard,
        "user_language_name": language_name(heard),
        "audio_base64": base64.b64encode(audio).decode("ascii") if audio else None,
    }
    if error:
        body["error"] = error
    return body


async def voiced(
    client: OpenAI,
    user_text: str,
    assistant_text: str,
    reply_language: str,
    user_language: str | None,
) -> JSONResponse:
    speech_bytes, speech_error = await spoken(client, assistant_text, reply_language)
    if speech_error == BAD_KEY:
        return fail(401, BAD_KEY)
    return JSONResponse(
        result_body(user_text, assistant_text, reply_language, user_language, speech_bytes, speech_error)
    )


async def spoken(client: OpenAI, text: str, language: str) -> tuple[bytes | None, str | None]:
    try:
        audio = await asyncio.to_thread(speak, client, text, language)
        return audio, None
    except APIStatusError as exc:
        log.error("speech failed: %s", exc)
        _status, message = api_error_message(exc)
        if _status == 401:
            return None, BAD_KEY
        return None, "The reply is ready, but the voice didn't come through. Try again."
    except APIConnectionError:
        log.exception("speech connection failed")
        return None, "Could not reach OpenAI. Check your internet connection and try again."


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
def status() -> JSONResponse:
    if not api_key():
        return fail(503, MISSING_KEY)
    return JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})


@app.post("/api/clear")
def clear(session_id: str = Form(...)) -> JSONResponse:
    checked = valid_session(session_id)
    if not checked:
        return fail(400, "Refresh the page and try again.")
    session = get_session(checked)
    session.turns.clear()
    session.generation += 1
    return JSONResponse({"ok": True})


@app.post("/api/talk")
async def talk(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    language: str = Form("auto"),
) -> JSONResponse:
    checked = valid_session(session_id)
    if not checked:
        return fail(400, "Refresh the page and try again.")
    chosen = (language or "auto").strip().lower()
    if chosen != "auto" and chosen not in NAMES:
        return fail(400, "Pick Auto, English, Hindi, Tamil, or Telugu.")
    locked = None if chosen == "auto" else chosen

    client = make_client()
    if client is None:
        return fail(503, MISSING_KEY)

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return fail(400, "That recording is too long. Try a shorter turn.")

    filename = audio_name(audio)
    reply_language = locked or "en"
    session = get_session(checked)
    generation = session.generation
    try:
        if len(audio_bytes) < MIN_AUDIO_BYTES:
            return await voiced(client, "", MISSED[reply_language], reply_language, reply_language)

        text, codes = await asyncio.to_thread(
            transcribe_audio,
            client,
            audio_bytes,
            filename,
            [locked] if locked else list(NAMES),
        )
        if not text:
            return await voiced(client, "", MISSED[reply_language], reply_language, reply_language)

        heard = choose_language(text, codes, locked)
        if heard not in NAMES:
            reply = MISSED[reply_language] if heard is None else UNSUPPORTED
            reply_language = reply_language if heard is None else "en"
            if heard is not None:
                remember(session, generation, text, reply)
            return await voiced(client, "" if heard is None else text, reply, reply_language, heard)

        reply_language = heard
        if session.generation != generation:
            history = []
        else:
            history = list(session.turns)
        reply = await asyncio.to_thread(complete_chat, client, history, text, reply_language)
        remember(session, generation, text, reply)
        return await voiced(client, text, reply, reply_language, heard)
    except APIStatusError as exc:
        log.error("openai status %s: %s", exc.status_code, exc)
        status_code, message = api_error_message(exc)
        return fail(status_code, message)
    except APIConnectionError:
        log.exception("openai connection failed")
        return fail(502, "Could not reach OpenAI. Check your internet connection and try again.")
    except Exception:
        log.exception("talk failed")
        return fail(502, "Something went wrong talking to OpenAI. Try again.")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
