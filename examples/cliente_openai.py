"""
Exemplo mínimo: usar o local-ai-bridge com o SDK oficial da OpenAI.

    pip install openai
    python examples/cliente_openai.py
"""
from openai import OpenAI

# A ponte ignora a api_key, mas o SDK exige uma string
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="nao-necessaria",
)


def exemplo_basico() -> None:
    resp = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {"role": "user", "content": "Explique em uma frase o que é structured output."},
        ],
    )
    print("=== Básico ===")
    print(resp.choices[0].message.content)


def exemplo_schema() -> None:
    """Structured output: o schema é respeitado mesmo em modelo de 7B."""
    schema = {
        "type": "object",
        "properties": {
            "resumo": {"type": "string"},
            "sentimento": {"type": "string", "enum": ["positivo", "neutro", "negativo"]},
            "confianca": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["resumo", "sentimento", "confianca"],
    }

    resp = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "Você é um analisador de texto. Responda em JSON."},
            {"role": "user", "content": "Adorei o produto, chegou antes do prazo e funciona perfeitamente!"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "analise", "strict": True, "schema": schema},
        },
    )

    import json

    print("\n=== Structured output ===")
    print(json.dumps(json.loads(resp.choices[0].message.content), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    exemplo_basico()
    exemplo_schema()
