from pydantic import BaseModel, Field


class PromptCreate(BaseModel):
    name: str = Field(..., min_length=1)
    base_prompt: str = Field(..., min_length=1)
    initial_message: str = Field(..., min_length=1)

    anger_level: int = Field(0, ge=0, le=5)
    prompt_change_instructions: str = Field("", min_length=0)

    selected_voice_slot: int = Field(1, ge=1, le=3)


class PromptUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    base_prompt: str = Field(..., min_length=1)
    initial_message: str = Field(..., min_length=1)

    anger_level: int = Field(..., ge=0, le=5)
    prompt_change_instructions: str = Field("", min_length=0)

    selected_voice_slot: int = Field(..., ge=1, le=3)


class PromptGenerateVariantRequest(BaseModel):
    instruction: str = Field(..., min_length=3)


class PromptGenerateVariantResponse(BaseModel):
    source_prompt_id: int
    source_prompt_name: str
    generated_name: str
    generated_prompt: str
    change_summary: str


class PromptGenerateVariantFromAudioResponse(BaseModel):
    source_prompt_id: int
    source_prompt_name: str
    transcribed_instruction: str
    generated_name: str
    generated_prompt: str
    change_summary: str


class PromptSelectVoiceSlotRequest(BaseModel):
    selected_voice_slot: int = Field(..., ge=1, le=3)


class VoiceSettingUpdate(BaseModel):
    label: str = Field(..., min_length=0)
    voice_id: str = Field(..., min_length=0)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class UserCreate(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    role: str = Field(..., pattern="^(admin|user|visitor)$")


class UserUpdate(BaseModel):
    email: str = Field(..., min_length=3)
    role: str = Field(..., pattern="^(admin|user|visitor)$")
    is_active: bool


class UserDeactivateRequest(BaseModel):
    is_active: bool = False
