"""
Testes de robustez: entradas que quebram o contrato do jogo.
Garante que o servidor NUNCA devolve algo que o jogo rejeite.
"""
import json

import httpx

URL = "http://127.0.0.1:8000/api/v1/chat/completions"

SCHEMA = {
    "type": "object",
    "properties": {
        "dialogue": {"type": "string"},
        "trust_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "emotion": {"type": "string", "enum": ["TRUSTING", "SUSPICIOUS", "ANGRY", "NEUTRAL"]},
        "completed_objectives": {"type": "array", "items": {"type": "string"}},
        "comedy": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["dialogue", "trust_percent", "emotion", "completed_objectives", "comedy"],
}

REQUIRED = set(SCHEMA["required"])
CASOS = [
    ("Abertura de ligacao", ""),
    ("Mensagem longa", "Palavra " * 400),
    ("Com emojis e acentos", "Oxe, você tá me dizendo isso com ÁGUA na boca? 😂 Rapaz..."),
    ("Pergunta bizarra", "E se eu te disser que sou do FBI e que seu gato é um agente russo?"),
    ("Silencio (vazio)", ""),
]


def validar(content: str) -> list[str]:
    p = []
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return ["resposta nao e JSON puro"]
    if not isinstance(obj, dict):
        return ["resposta nao e objeto"]
    if set(obj.keys()) != REQUIRED:
        p.append(f"campos errados: {sorted(obj.keys())}")
    if not str(obj.get("dialogue", "")).strip():
        p.append("dialogue vazio")
    try:
        if not (0 <= int(obj.get("trust_percent", -1)) <= 100):
            p.append("trust_percent invalido")
    except (TypeError, ValueError):
        p.append("trust_percent nao numerico")
    if obj.get("emotion") not in ("TRUSTING", "SUSPICIOUS", "ANGRY", "NEUTRAL"):
        p.append(f"emotion invalida: {obj.get('emotion')}")
    if not isinstance(obj.get("completed_objectives"), list):
        p.append("completed_objectives nao e lista")
    try:
        if not (0 <= float(obj.get("comedy", -1)) <= 1):
            p.append("comedy fora de 0..1")
    except (TypeError, ValueError):
        p.append("comedy nao numerico")
    # o jogo rejeita markup solto no inicio
    d = str(obj.get("dialogue", "")).strip()
    if d and d[0] in '{["`<':
        p.append("dialogue comeca com markup")
    return p


def main() -> int:
    print("=" * 74)
    print("TESTES DE ROBUSTEZ")
    print("=" * 74)
    falhas = 0
    for nome, texto in CASOS:
        body = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "Voce e o Seu Zeca, feirante em Recife. UMA frase."},
                {"role": "user", "content": texto or "(silencio)"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "gameplay_caller_turn", "strict": True, "schema": SCHEMA},
            },
        }
        r = httpx.post(URL, json=body, timeout=300)
        if r.status_code != 200:
            print(f"\n[{nome}] HTTP {r.status_code}")
            falhas += 1
            continue
        content = r.json()["choices"][0]["message"]["content"]
        problemas = validar(content)
        if problemas:
            falhas += 1
        mark = "OK   " if not problemas else "FALHA"
        print(f"\n[{nome}] {mark}")
        try:
            obj = json.loads(content)
            print(f"   {obj.get('dialogue')}")
        except Exception:
            print(f"   {content[:150]}")
        if problemas:
            print(f"   ! {', '.join(problemas)}")

    print("\n" + "=" * 74)
    print(f"{len(CASOS) - falhas}/{len(CASOS)} casos validaram")
    print("=" * 74)
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
