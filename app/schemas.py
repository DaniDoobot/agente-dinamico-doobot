from pydantic import BaseModel, Field


class PromptCreate(BaseModel):
    name: str = Field(..., min_length=1)
    base_prompt: str = Field(..., min_length=1)
    initial_message: str = Field(..., min_length=1)


class PromptUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    base_prompt: str = Field(..., min_length=1)
    initial_message: str = Field(..., min_length=1)


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
