from pydantic import BaseModel


class PromptCreate(BaseModel):
    name: str
    base_prompt: str


class PromptResponse(BaseModel):
    id: int
    name: str
    base_prompt: str
    is_active: bool
    created_at: str
    updated_at: str