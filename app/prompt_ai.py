import json
import os

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_prompt_variant(base_name: str, base_prompt: str, instruction: str) -> dict:
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    developer_instructions = """
Eres un experto en redacción y adaptación de prompts para agentes conversacionales de voz.

Tu tarea consiste en generar una NUEVA variante de un prompt existente siguiendo estas reglas:

1. Conserva toda la estructura útil del prompt original siempre que siga siendo compatible.
2. Aplica únicamente los cambios pedidos por el usuario.
3. No elimines detalles importantes del prompt original salvo que el usuario lo pida.
4. El resultado debe ser un prompt operativo, claro y directamente utilizable.
5. No expliques el prompt dentro del propio prompt final.
6. Debes devolver EXCLUSIVAMENTE un JSON válido con estas claves:
   - generated_name
   - change_summary
   - generated_prompt

Reglas del JSON:
- generated_name: nombre corto y útil para la nueva variante
- change_summary: resumen breve de los cambios realizados
- generated_prompt: texto completo del nuevo prompt
"""

    user_input = f"""
NOMBRE DEL PROMPT ORIGINAL:
{base_name}

PROMPT ORIGINAL:
{base_prompt}

INSTRUCCIÓN DEL USUARIO:
{instruction}

Genera una variante del prompt.
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": developer_instructions},
            {"role": "user", "content": user_input},
        ],
    )

    text = response.output_text.strip()
    data = json.loads(text)

    return {
        "generated_name": data["generated_name"].strip(),
        "change_summary": data["change_summary"].strip(),
        "generated_prompt": data["generated_prompt"].strip(),
    }
