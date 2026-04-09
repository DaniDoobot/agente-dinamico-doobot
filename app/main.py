from fastapi import FastAPI, HTTPException, Request, Response
import requests
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

from app.db import get_connection
from app.schemas import (
    PromptCreate,
    PromptUpdate,
    PromptGenerateVariantRequest,
    PromptGenerateVariantResponse,
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
        "is_active": row[3],
        "created_at": row[4].isoformat(),
        "updated_at": row[5].isoformat(),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/prompts")
def list_prompts():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, base_prompt, is_active, created_at, updated_at
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
        SELECT id, name, base_prompt, is_active, created_at, updated_at
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
        INSERT INTO prompts (name, base_prompt, is_active)
        VALUES (%s, %s, FALSE)
        RETURNING id, name, base_prompt, is_active, created_at, updated_at
    """, (payload.name, payload.base_prompt))

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
            updated_at = NOW()
        WHERE id = %s
        RETURNING id, name, base_prompt, is_active, created_at, updated_at
    """, (payload.name, payload.base_prompt, prompt_id))

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
        SELECT id, name, base_prompt
        FROM prompts
        WHERE is_active = TRUE
        LIMIT 1
    """)
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=400, detail="No hay prompt activo en la base de datos")

    active_prompt_id = row[0]
    active_prompt_name = row[1]
    override_prompt = row[2]

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
                    }
                }
            }
        }
    }

    resp = requests.post(
        "https://api.elevenlabs.io/v1/convai/twilio/register-call",
        headers={
            "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=20,
    )

    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"ElevenLabs error: {resp.text}")

    return Response(content=resp.text, media_type="application/xml")
