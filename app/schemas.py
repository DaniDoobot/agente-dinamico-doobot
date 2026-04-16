from pydantic import BaseModel, Field


class PromptCreate(BaseModel):
    name: str = Field(..., min_length=1)
    base_prompt: str = Field(..., min_length=1)
    initial_message: str = Field(..., min_length=1)

    anger_level: int = Field(0, ge=0, le=5)
    complaint_reasons: str = Field("", min_length=0)

    voice_slot_1: str = Field(..., min_length=1)
    voice_slot_1_label: str = Field("Voz principal", min_length=1)

    voice_slot_2: str = ""
    voice_slot_2_label: str = ""

    voice_slot_3: str = ""
    voice_slot_3_label: str = ""

    selected_voice_slot: int = Field(1, ge=1, le=3)


class PromptUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    base_prompt: str = Field(..., min_length=1)
    initial_message: str = Field(..., min_length=1)

    anger_level: int = Field(..., ge=0, le=5)
    complaint_reasons: str = Field("", min_length=0)

    voice_slot_1: str = Field(..., min_length=1)
    voice_slot_1_label: str = Field(..., min_length=1)

    voice_slot_2: str = ""
    voice_slot_2_label: str = ""

    voice_slot_3: str = ""
    voice_slot_3_label: str = ""

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
