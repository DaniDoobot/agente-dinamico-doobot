import json
import os
from typing import Optional

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


def build_prompt_with_ai(
    *,
    name: str,
    base_prompt: str,
    anger_level: Optional[int] = None,
    complaint_reasons: Optional[str] = None,
) -> str:
    """
    Genera un base_prompt final armonizado usando IA.

    Reglas:
    - initial_message NO entra aquí y NO debe modificarse fuera de esta función.
    - Si anger_level o complaint_reasons no vienen informados, no se fuerzan.
    - Se conserva al máximo la estructura útil del prompt original.
    """

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    anger_text = (
        f"- Nivel de enfado inicial: {anger_level} en una escala de 0 a 5."
        if anger_level is not None
        else "- Nivel de enfado inicial: no especificado."
    )

    complaint_text = (
        f"- Razones de queja que deben integrarse, si encajan: {complaint_reasons}"
        if complaint_reasons and complaint_reasons.strip()
        else "- Razones de queja: no especificadas."
    )

    developer_instructions = """
Eres un experto en redacción de prompts para agentes conversacionales de voz.

Tu tarea es reconstruir y armonizar un prompt base para que quede coherente y operativo,
manteniendo la mayor parte posible de su estructura, reglas críticas y lógica interna.

REGLAS IMPORTANTES:
1. Debes trabajar ÚNICAMENTE sobre el prompt base.
2. No debes generar ni modificar ningún saludo inicial ni first message.
3. Tu salida debe ser SOLO el texto completo del prompt final.
4. Mantén intactas, salvo incompatibilidad clara, las reglas críticas, identidad del personaje,
   restricciones de rol, idioma y normas de cierre.
5. Si se informa un nivel de enfado:
   - ajústalo de forma coherente en tono, tensión inicial, resistencia, actitud defensiva
     y sistema emocional del personaje.
6. Si se informan razones de queja:
   - intégralas de forma natural en el conflicto, la personalidad, la evolución de la conversación
     o las partes del prompt donde encajen mejor.
7. Si algún campo no se informa, no lo inventes ni lo fuerces.
8. No expliques lo que haces.
9. No devuelvas JSON.
10. Devuelve EXCLUSIVAMENTE el prompt final completo.
"""

    user_input = f"""
NOMBRE DEL PROMPT:
{name}

PROMPT BASE ORIGINAL:
{base_prompt}

AJUSTES A TENER EN CUENTA:
{anger_text}
{complaint_text}

INSTRUCCIÓN GENERAL:
Reconstruye el prompt base para que quede coherente con los ajustes informados, usando solo los
que realmente estén presentes. Conserva la estructura útil del prompt original y no toques
ningún saludo inicial fuera del prompt.
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": developer_instructions},
            {"role": "user", "content": user_input},
        ],
    )

    final_prompt = response.output_text.strip()

    if not final_prompt:
        raise ValueError("La IA no devolvió un prompt final válido")

    return final_prompt
