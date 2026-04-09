from pydantic import BaseModel, Field


class PromptCreate(BaseModel):
    name: str
    base_prompt: str


class PromptUpdate(BaseModel):
    name: str
    base_prompt: str


class PromptGenerateVariantRequest(BaseModel):
    instruction: str = Field(..., min_length=3)


class PromptGenerateVariantResponse(BaseModel):
    source_prompt_id: int
    source_prompt_name: str
    generated_name: str
    generated_prompt: str
    change_summary: str
