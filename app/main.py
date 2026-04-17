from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    UploadFile,
    File,
    Depends,
    Header,
)
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
    PromptGenerateVariantFromAudioResponse,
    PromptSelectVoiceSlotRequest,
    VoiceSettingUpdate,
    LoginRequest,
    UserCreate,
    UserUpdate,
    UserDeactivateRequest,
)
from app.prompt_ai import generate_prompt_variant, build_prompt_with_ai
from app.audio_ai import transcribe_audio_bytes
from app.auth_utils import verify_password, create_access_token, decode_access_token

load_dotenv()

app = FastAPI(title="Prompt Manager API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# AUTH / USERS / AUDIT
# =========================

def log_audit_event(
    *,
    user_id: int | None,
    action: str,
    entity_type: str,
    entity_id: str | None = None,
    details_json: dict | None = None,
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO audit_logs (user_id, action, entity_type, entity_id, details_json)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            user_id,
            action,
            entity_type,
            entity_id,
            details_json or {},
        ),
    )

    conn.commit()
    cur.close()
    conn.close()


def get_user_by_id(user_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, email, role, is_active, created_at, updated_at
        FROM users
        WHERE id = %s
        """,
        (user_id,),
    )
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "email": row[1],
        "role": row[2],
        "is_active": row[3],
        "created_at": row[4].isoformat(),
        "updated_at": row[5].isoformat(),
    }


def parse_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Falta cabecera Authorization")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization debe ser Bearer token")

    return parts[1].strip()


def get_current_user(authorization: str | None = Header(default=None)):
    token = parse_bearer_token(authorization)
    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token inválido")

    try:
        user_id = int(sub)
    except ValueError:
        raise HTTPException(status_code=401, detail="Token inválido")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")

    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Usuario desactivado")

    return user


def require_roles(*allowed_roles):
    def dependency(current_user=Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="No tienes permisos para esta acción")
        return current_user
    return dependency


# =========================
# VOICES
# =========================

def get_voice_settings_map():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT slot_number, voice_id, label
        FROM voice_settings
        ORDER BY slot_number ASC
    """)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    voice_map = {}
    for row in rows:
        voice_map[row[0]] = {
            "slot_number": row[0],
            "voice_id": row[1],
            "label": row[2],
        }

    return voice_map


def get_selected_voice_info(selected_voice_slot: int):
    voice_map = get_voice_settings_map()

    selected = voice_map.get(selected_voice_slot)
    fallback = voice_map.get(1)

    if selected and selected["voice_id"]:
        return selected

    if fallback and fallback["voice_id"]:
        return fallback

    raise HTTPException(
        status_code=500,
        detail="No hay una voz global válida configurada"
    )


# =========================
# PROMPTS
# =========================

def serialize_prompt_row(row):
    selected_voice_slot = row[9]
    selected_voice = get_selected_voice_info(selected_voice_slot)

    return {
        "id": row[0],
        "name": row[1],
        "base_prompt": row[2],
        "initial_message": row[3],
        "is_active": row[4],
        "created_at": row[5].isoformat(),
        "updated_at": row[6].isoformat(),
        "anger_level": row[7],
        "prompt_change_instructions": row[8],
        "selected_voice_slot": row[9],
        "selected_voice_label": selected_voice["label"],
        "selected_voice_id": selected_voice["voice_id"],
    }


# =========================
# BASIC
# =========================

@app.get("/health")
def health():
    return {"ok": True}


# =========================
# AUTH ENDPOINTS
# =========================

@app.post("/auth/login")
def login(payload: LoginRequest):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, email, password_hash, role, is_active
        FROM users
        WHERE email = %s
        """,
        (payload.email,),
    )
    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    user_id = row[0]
    email = row[1]
    password_hash = row[2]
    role = row[3]
    is_active = row[4]

    if not is_active:
        raise HTTPException(status_code=403, detail="Usuario desactivado")

    if not verify_password(payload.password, password_hash):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    access_token = create_access_token(user_id=user_id, email=email, role=role)

    log_audit_event(
        user_id=user_id,
        action="LOGIN_SUCCESS",
        entity_type="auth",
        entity_id=str(user_id),
        details_json={"email": email, "role": role},
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": email,
            "role": role,
            "is_active": is_active,
        },
    }


@app.get("/auth/me")
def auth_me(current_user=Depends(get_current_user)):
    return current_user


# =========================
# USER ADMIN
# =========================

@app.get("/users")
def list_users(current_user=Depends(require_roles("admin"))):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, email, role, is_active, created_at, updated_at
        FROM users
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "id": row[0],
            "email": row[1],
            "role": row[2],
            "is_active": row[3],
            "created_at": row[4].isoformat(),
            "updated_at": row[5].isoformat(),
        }
        for row in rows
    ]


@app.post("/users")
def create_user(payload: UserCreate, current_user=Depends(require_roles("admin"))):
    from app.auth_utils import hash_password

    password_hash = hash_password(payload.password)

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO users (email, password_hash, role, is_active)
            VALUES (%s, %s, %s, TRUE)
            RETURNING id, email, role, is_active, created_at, updated_at
        """, (payload.email, password_hash, payload.role))

        row = cur.fetchone()
        conn.commit()
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail=f"No se pudo crear el usuario: {str(e)}")

    cur.close()
    conn.close()

    log_audit_event(
        user_id=current_user["id"],
        action="USER_CREATED",
        entity_type="user",
        entity_id=str(row[0]),
        details_json={"email": row[1], "role": row[2]},
    )

    return {
        "id": row[0],
        "email": row[1],
        "role": row[2],
        "is_active": row[3],
        "created_at": row[4].isoformat(),
        "updated_at": row[5].isoformat(),
    }


@app.put("/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate, current_user=Depends(require_roles("admin"))):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id
        FROM users
        WHERE id = %s
    """, (user_id,))
    existing = cur.fetchone()

    if not existing:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    try:
        cur.execute("""
            UPDATE users
            SET email = %s,
                role = %s,
                is_active = %s,
                updated_at = NOW()
            WHERE id = %s
            RETURNING id, email, role, is_active, created_at, updated_at
        """, (payload.email, payload.role, payload.is_active, user_id))

        row = cur.fetchone()
        conn.commit()
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail=f"No se pudo actualizar el usuario: {str(e)}")

    cur.close()
    conn.close()

    log_audit_event(
        user_id=current_user["id"],
        action="USER_UPDATED",
        entity_type="user",
        entity_id=str(row[0]),
        details_json={"email": row[1], "role": row[2], "is_active": row[3]},
    )

    return {
        "id": row[0],
        "email": row[1],
        "role": row[2],
        "is_active": row[3],
        "created_at": row[4].isoformat(),
        "updated_at": row[5].isoformat(),
    }


@app.post("/users/{user_id}/deactivate")
def deactivate_user(
    user_id: int,
    payload: UserDeactivateRequest,
    current_user=Depends(require_roles("admin")),
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, email, role, is_active
        FROM users
        WHERE id = %s
    """, (user_id,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    cur.execute("""
        UPDATE users
        SET is_active = %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING id, email, role, is_active, created_at, updated_at
    """, (payload.is_active, user_id))

    updated = cur.fetchone()
    conn.commit()

    cur.close()
    conn.close()

    log_audit_event(
        user_id=current_user["id"],
        action="USER_DEACTIVATED" if payload.is_active is False else "USER_REACTIVATED",
        entity_type="user",
        entity_id=str(updated[0]),
        details_json={"email": updated[1], "role": updated[2], "is_active": updated[3]},
    )

    return {
        "id": updated[0],
        "email": updated[1],
        "role": updated[2],
        "is_active": updated[3],
        "created_at": updated[4].isoformat(),
        "updated_at": updated[5].isoformat(),
    }


# =========================
# VOICE SETTINGS
# =========================

@app.get("/voice-settings")
def list_voice_settings(current_user=Depends(require_roles("admin", "user", "visitor"))):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT slot_number, voice_id, label
        FROM voice_settings
        ORDER BY slot_number ASC
    """)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "slot_number": row[0],
            "voice_id": row[1],
            "label": row[2],
        }
        for row in rows
    ]


@app.put("/voice-settings/{slot_number}")
def update_voice_setting(
    slot_number: int,
    payload: VoiceSettingUpdate,
    current_user=Depends(require_roles("admin", "user")),
):
    if slot_number not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Slot de voz inválido")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT slot_number, voice_id, label
        FROM voice_settings
        WHERE slot_number = %s
    """, (slot_number,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Slot de voz no encontrado")

    if slot_number == 1:
        new_voice_id = row[1]
        new_label = payload.label if payload.label.strip() else row[2]
    else:
        new_voice_id = payload.voice_id.strip()
        new_label = payload.label.strip()

    if slot_number == 1 and not new_voice_id:
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="La voz principal debe tener voice_id")

    cur.execute("""
        UPDATE voice_settings
        SET voice_id = %s,
            label = %s
        WHERE slot_number = %s
        RETURNING slot_number, voice_id, label
    """, (new_voice_id, new_label, slot_number))

    updated = cur.fetchone()
    conn.commit()

    cur.close()
    conn.close()

    log_audit_event(
        user_id=current_user["id"],
        action="VOICE_SETTING_UPDATED",
        entity_type="voice_setting",
        entity_id=str(updated[0]),
        details_json={"slot_number": updated[0], "voice_id": updated[1], "label": updated[2]},
    )

    return {
        "slot_number": updated[0],
        "voice_id": updated[1],
        "label": updated[2],
    }


# =========================
# PROMPTS
# =========================

@app.get("/prompts")
def list_prompts(current_user=Depends(require_roles("admin", "user", "visitor"))):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            name,
            base_prompt,
            initial_message,
            is_active,
            created_at,
            updated_at,
            anger_level,
            prompt_change_instructions,
            selected_voice_slot
        FROM prompts
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [serialize_prompt_row(row) for row in rows]


@app.get("/prompts/active")
def get_active_prompt(current_user=Depends(require_roles("admin", "user", "visitor"))):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            name,
            base_prompt,
            initial_message,
            is_active,
            created_at,
            updated_at,
            anger_level,
            prompt_change_instructions,
            selected_voice_slot
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
def create_prompt(payload: PromptCreate, current_user=Depends(require_roles("admin", "user", "visitor"))):
    try:
        generated_base_prompt = build_prompt_with_ai(
            name=payload.name,
            base_prompt=payload.base_prompt,
            anger_level=payload.anger_level,
            prompt_change_instructions=payload.prompt_change_instructions,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando el prompt con IA: {str(e)}"
        )

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO prompts (
            name,
            base_prompt,
            initial_message,
            is_active,
            anger_level,
            prompt_change_instructions,
            selected_voice_slot
        )
        VALUES (%s, %s, %s, FALSE, %s, %s, %s)
        RETURNING
            id,
            name,
            base_prompt,
            initial_message,
            is_active,
            created_at,
            updated_at,
            anger_level,
            prompt_change_instructions,
            selected_voice_slot
    """, (
        payload.name,
        generated_base_prompt,
        payload.initial_message,
        payload.anger_level,
        payload.prompt_change_instructions,
        payload.selected_voice_slot,
    ))

    row = cur.fetchone()
    conn.commit()

    cur.close()
    conn.close()

    serialized = serialize_prompt_row(row)

    log_audit_event(
        user_id=current_user["id"],
        action="PROMPT_CREATED",
        entity_type="prompt",
        entity_id=str(serialized["id"]),
        details_json={"name": serialized["name"]},
    )

    return serialized


@app.put("/prompts/{prompt_id}")
def update_prompt(prompt_id: int, payload: PromptUpdate, current_user=Depends(require_roles("admin", "user", "visitor"))):
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

    try:
        generated_base_prompt = build_prompt_with_ai(
            name=payload.name,
            base_prompt=payload.base_prompt,
            anger_level=payload.anger_level,
            prompt_change_instructions=payload.prompt_change_instructions,
        )
    except Exception as e:
        cur.close()
        conn.close()
        raise HTTPException(
            status_code=500,
            detail=f"Error regenerando el prompt con IA: {str(e)}"
        )

    cur.execute("""
        UPDATE prompts
        SET
            name = %s,
            base_prompt = %s,
            initial_message = %s,
            anger_level = %s,
            prompt_change_instructions = %s,
            selected_voice_slot = %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING
            id,
            name,
            base_prompt,
            initial_message,
            is_active,
            created_at,
            updated_at,
            anger_level,
            prompt_change_instructions,
            selected_voice_slot
    """, (
        payload.name,
        generated_base_prompt,
        payload.initial_message,
        payload.anger_level,
        payload.prompt_change_instructions,
        payload.selected_voice_slot,
        prompt_id,
    ))

    row = cur.fetchone()
    conn.commit()

    cur.close()
    conn.close()

    serialized = serialize_prompt_row(row)

    log_audit_event(
        user_id=current_user["id"],
        action="PROMPT_UPDATED",
        entity_type="prompt",
        entity_id=str(serialized["id"]),
        details_json={"name": serialized["name"]},
    )

    return serialized


@app.post("/prompts/{prompt_id}/activate")
def activate_prompt(prompt_id: int, current_user=Depends(require_roles("admin", "user", "visitor"))):
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

    log_audit_event(
        user_id=current_user["id"],
        action="PROMPT_ACTIVATED",
        entity_type="prompt",
        entity_id=str(prompt_id),
        details_json={"prompt_id": prompt_id},
    )

    return {"ok": True, "active_prompt_id": prompt_id}


@app.post("/prompts/{prompt_id}/select-voice-slot")
def select_voice_slot(
    prompt_id: int,
    payload: PromptSelectVoiceSlotRequest,
    current_user=Depends(require_roles("admin", "user", "visitor")),
):
    if payload.selected_voice_slot not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Slot de voz inválido")

    voice_info = get_selected_voice_info(payload.selected_voice_slot)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id
        FROM prompts
        WHERE id = %s
    """, (prompt_id,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Prompt no encontrado")

    cur.execute("""
        UPDATE prompts
        SET selected_voice_slot = %s,
            updated_at = NOW()
        WHERE id = %s
    """, (payload.selected_voice_slot, prompt_id))

    conn.commit()

    cur.close()
    conn.close()

    log_audit_event(
        user_id=current_user["id"],
        action="PROMPT_VOICE_SELECTED",
        entity_type="prompt",
        entity_id=str(prompt_id),
        details_json={
            "prompt_id": prompt_id,
            "selected_voice_slot": payload.selected_voice_slot,
            "selected_voice_label": voice_info["label"],
            "selected_voice_id": voice_info["voice_id"],
        },
    )

    return {
        "ok": True,
        "prompt_id": prompt_id,
        "selected_voice_slot": payload.selected_voice_slot,
        "selected_voice_id": voice_info["voice_id"],
        "selected_voice_label": voice_info["label"],
    }


@app.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: int, current_user=Depends(require_roles("admin", "user"))):
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

    log_audit_event(
        user_id=current_user["id"],
        action="PROMPT_DELETED",
        entity_type="prompt",
        entity_id=str(prompt_id),
        details_json={"prompt_id": prompt_id},
    )

    return {"ok": True, "deleted_prompt_id": prompt_id}


# =========================
# AI / AUDIO
# =========================

@app.post(
    "/prompts/{prompt_id}/generate-variant",
    response_model=PromptGenerateVariantResponse,
)
def generate_variant(
    prompt_id: int,
    payload: PromptGenerateVariantRequest,
    current_user=Depends(require_roles("admin", "user", "visitor")),
):
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


@app.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user=Depends(require_roles("admin", "user", "visitor")),
):
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
async def generate_variant_from_audio(
    prompt_id: int,
    file: UploadFile = File(...),
    current_user=Depends(require_roles("admin", "user", "visitor")),
):
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


# =========================
# TWILIO / ELEVENLABS
# =========================

@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    form = await request.form()

    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id,
            name,
            base_prompt,
            initial_message,
            selected_voice_slot
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
    override_initial_message = row[3]
    selected_voice_slot = row[4]

    selected_voice = get_selected_voice_info(selected_voice_slot)
    selected_voice_id = selected_voice["voice_id"]

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
                },
                "tts": {
                    "voice_id": selected_voice_id
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
