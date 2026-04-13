from fastapi import FastAPI, HTTPException, Request, Response
import requests
from dotenv import load_dotenv
import os
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from app.db import get_connection
from app.audio_ai import transcribe_audio_bytes
from app.schemas import (
    PromptCreate,
    PromptUpdate,
    PromptGenerateVariantRequest,
    PromptGenerateVariantResponse,
    PromptGenerateVariantFromAudioResponse,
)
from app.prompt_ai import generate_prompt_variant

load_dotenv()

app = FastAPI(title="Prompt Manager API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize_prompt_row(row):
    return {
        "id": row[0],
        "name": row[1],
        "base_prompt": row[2],
        "initial_message": row[3],
        "is_active": row[4],
        "created_at": row[5].isoformat() if row[5] else None,
        "updated_at": row[6].isoformat() if row[6] else None,
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/prompts")
def list_prompts():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, base_prompt, initial_message, is_active, created_at, updated_at
        FROM prompts
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [serialize_prompt_row(row) for row in rows]


@app.get("/prompts/active")
def get_active_prompt():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, base_prompt, initial_message, is_active, created_at, updated_at
        FROM prompts
        WHERE is_active = TRUE
        LIMIT 1
    """)
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="No hay prompt activo")

    return serialize_prompt_row(row)


@app.post("/prompts")
def create_prompt(payload: PromptCreate):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO prompts (name, base_prompt, initial_message, is_active)
        VALUES (%s, %s, %s, FALSE)
        RETURNING id, name, base_prompt, initial_message, is_active, created_at, updated_at
    """, (
        payload.name.strip(),
        payload.base_prompt,
        payload.initial_message.strip(),
    ))

    row = cur.fetchone()
    conn.commit()

    cur.close()
    conn.close()

    return serialize_prompt_row(row)


@app.put("/prompts/{prompt_id}")
def update_prompt(prompt_id: int, payload: PromptUpdate):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id
        FROM prompts
        WHERE id = %s
    """, (prompt_id,))
    existing = cur.fetchone()

    if not existing:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Prompt no encontrado")

    cur.execute("""
        UPDATE prompts
        SET name = %s,
            base_prompt = %s,
            initial_message = %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING id, name, base_prompt, initial_message, is_active, created_at, updated_at
    """, (
        payload.name.strip(),
        payload.base_prompt,
        payload.initial_message.strip(),
        prompt_id,
    ))

    row = cur.fetchone()
    conn.commit()

    cur.close()
    conn.close()

    return serialize_prompt_row(row)


@app.post("/prompts/{prompt_id}/activate")
def activate_prompt(prompt_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM prompts WHERE id = %s", (prompt_id,))
    exists = cur.fetchone()

    if not exists:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Prompt no encontrado")

    cur.execute("UPDATE prompts SET is_active = FALSE WHERE is_active = TRUE")
    cur.execute("""
        UPDATE prompts
        SET is_active = TRUE, updated_at = NOW()
        WHERE id = %s
    """, (prompt_id,))

    conn.commit()

    cur.close()
    conn.close()

    return {"ok": True, "active_prompt_id": prompt_id}


@app.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, is_active
        FROM prompts
        WHERE id = %s
    """, (prompt_id,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Prompt no encontrado")

    is_active = row[1]

    if is_active:
        cur.close()
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="No se puede eliminar el prompt activo. Activa otro antes de borrarlo."
        )

    cur.execute("DELETE FROM prompts WHERE id = %s", (prompt_id,))
    conn.commit()

    cur.close()
    conn.close()

    return {"ok": True, "deleted_prompt_id": prompt_id}


@app.post(
    "/prompts/{prompt_id}/generate-variant",
    response_model=PromptGenerateVariantResponse,
)
def generate_variant(prompt_id: int, payload: PromptGenerateVariantRequest):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, base_prompt
        FROM prompts
        WHERE id = %s
    """, (prompt_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Prompt no encontrado")

    source_prompt_id = row[0]
    source_prompt_name = row[1]
    source_prompt_text = row[2]

    try:
        result = generate_prompt_variant(
            base_name=source_prompt_name,
            base_prompt=source_prompt_text,
            instruction=payload.instruction,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando variante con IA: {str(e)}"
        )

    return {
        "source_prompt_id": source_prompt_id,
        "source_prompt_name": source_prompt_name,
        "generated_name": result["generated_name"],
        "generated_prompt": result["generated_prompt"],
        "change_summary": result["change_summary"],
    }


@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    form = await request.form()

    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, base_prompt, initial_message
        FROM prompts
        WHERE is_active = TRUE
        LIMIT 1
    """)
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=400,
            detail="No hay prompt activo en la base de datos"
        )

    active_prompt_id = row[0]
    active_prompt_name = row[1]
    override_prompt = row[2]
    override_initial_message = (row[3] or "").strip()

    payload = {
        "agent_id": os.getenv("ELEVENLABS_AGENT_ID"),
        "from_number": from_number,
        "to_number": to_number,
        "direction": "inbound",
        "conversation_initiation_client_data": {
            "type": "conversation_initiation_client_data",
            "dynamic_variables": {
                "call_sid": call_sid,
                "active_prompt_id": str(active_prompt_id),
                "active_prompt_name": active_prompt_name,
            },
            "conversation_config_override": {
                "agent": {
                    "prompt": {
                        "prompt": override_prompt
                    },
                    "first_message": override_initial_message
                }
            }
        }
    }

    print("ACTIVE PROMPT:", active_prompt_id, active_prompt_name)
    print("OVERRIDE INITIAL MESSAGE:", repr(override_initial_message))
    print("PAYLOAD ELEVENLABS:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    resp = requests.post(
        "https://api.elevenlabs.io/v1/convai/twilio/register-call",
        headers={
            "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=20,
    )

    print("ELEVENLABS STATUS:", resp.status_code)
    print("ELEVENLABS RESPONSE:", resp.text)

    if not resp.ok:
        raise HTTPException(
            status_code=500,
            detail=f"ElevenLabs error: {resp.text}"
        )

    return Response(content=resp.text, media_type="application/xml")
from fastapi import UploadFile, File, HTTPException
from app.audio_ai import transcribe_audio_bytes

@app.post("/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = transcribe_audio_bytes(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
        )
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribiendo audio: {str(e)}")
@app.post(
    "/prompts/{prompt_id}/generate-variant-from-audio",
    response_model=PromptGenerateVariantFromAudioResponse,
)
async def generate_variant_from_audio(prompt_id: int, file: UploadFile = File(...)):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, base_prompt
        FROM prompts
        WHERE id = %s
    """, (prompt_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Prompt no encontrado")

    source_prompt_id = row[0]
    source_prompt_name = row[1]
    source_prompt_text = row[2]

    try:
        content = await file.read()

        transcribed_instruction = transcribe_audio_bytes(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
        )

        if not transcribed_instruction:
            raise HTTPException(
                status_code=400,
                detail="No se ha podido obtener texto del audio"
            )

        result = generate_prompt_variant(
            base_name=source_prompt_name,
            base_prompt=source_prompt_text,
            instruction=transcribed_instruction,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando variante desde audio: {str(e)}"
        )

    return {
        "source_prompt_id": source_prompt_id,
        "source_prompt_name": source_prompt_name,
        "transcribed_instruction": transcribed_instruction,
        "generated_name": result["generated_name"],
        "generated_prompt": result["generated_prompt"],
        "change_summary": result["change_summary"],
    }
