import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio_bytes(filename: str, content: bytes, content_type: str | None = None) -> str:
    transcript = client.audio.transcriptions.create(
        model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
        file=(filename, content, content_type or "application/octet-stream"),
    )

    text = getattr(transcript, "text", None) or ""
    return text.strip()
