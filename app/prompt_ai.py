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
    prompt_change_instructions: Optional[str] = None,
) -> str:
    """
    Genera un base_prompt final armonizado usando IA.

    Reglas:
    - initial_message NO entra aquí y NO debe modificarse fuera de esta función.
    - prompt_change_instructions se interpreta como texto libre principal
      para reescribir o ajustar el prompt.
    - anger_level actúa como apoyo estructurado simple.
    - se conserva la estructura principal del prompt salvo donde las
      instrucciones del usuario exijan cambio.
    """

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    anger_text = (
        f"- Nivel de enfado inicial solicitado: {anger_level} en una escala de 0 a 5."
        if anger_level is not None
        else "- Nivel de enfado inicial: no especificado."
    )

    free_change_text = (
        f"- Cambios libres solicitados por el usuario: {prompt_change_instructions}"
        if prompt_change_instructions and prompt_change_instructions.strip()
        else "- Cambios libres solicitados por el usuario: no especificados."
    )

    developer_instructions = """
Eres un experto en redacción y reescritura de prompts para agentes conversacionales de voz.

Tu tarea es reconstruir y armonizar un prompt base para que quede coherente, operativo
y bien escrito, manteniendo la estructura principal y la lógica útil del prompt original
salvo donde el usuario haya pedido cambios.

REGLAS IMPORTANTES:
1. Debes trabajar ÚNICAMENTE sobre el prompt base.
2. No debes generar ni modificar ningún saludo inicial ni first message.
3. Tu salida debe ser SOLO el texto completo del prompt final.
4. Conserva la estructura principal del prompt base siempre que siga siendo útil.
5. Si el usuario pide cambios libres:
   - trátalos como autoridad editorial principal sobre el contenido del prompt.
   - si contradicen partes del prompt base, modifica o sustituye esas partes.
   - elimina o sustituye contenido antiguo cuando ya no encaje con la nueva intención.
   - no acumules sin criterio contenido viejo y contenido nuevo.
6. Si se informa un nivel de enfado:
   - úsalo para ajustar tono, tensión inicial, resistencia y comportamiento emocional,
     pero sin romper la coherencia general.
7. Mantén, salvo que el usuario pida lo contrario, las reglas útiles de:
   - estructura general
   - rol
   - idioma
   - naturalidad
   - cierre
8. Si el usuario cambia identidad, conflicto principal, actitud o focos de queja,
   debes reflejarlo de forma coherente en todas las partes afectadas del prompt.
9. No expliques lo que haces.
10. No devuelvas JSON.
11. Devuelve EXCLUSIVAMENTE el prompt final completo.

CRITERIO DE CALIDAD:
- El resultado debe parecer un prompt reescrito con intención unitaria.
- No debe parecer una suma de parches.
- Si el usuario redefine algo importante, el prompt final debe adaptarse de verdad.
"""

    user_input = f"""
NOMBRE DEL PROMPT:
{name}

PROMPT BASE ORIGINAL:
{base_prompt}

AJUSTES A TENER EN CUENTA:
{anger_text}
{free_change_text}

INSTRUCCIÓN GENERAL:
Reescribe el prompt base conservando su estructura principal, pero aplicando de verdad los
cambios libres indicados por el usuario. Si hay conflicto entre el prompt original y los cambios
solicitados, prioriza los cambios del usuario. No toques ningún saludo inicial fuera del prompt.
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
