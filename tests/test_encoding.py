"""
Teste do bug de ENCODING que quebrava a voz no jogo.

O jogo le o JSON da resposta como Latin-1/Windows-1252. Se o servidor mandar
UTF-8 cru, o TTS recebe lixo:
    "tô eu"  ->  "tÃ´â\x90\xa0eu"   (o espaco vira U+2420)

Este teste simula a camada do jogo: pega o content, passa por latin-1
e verifica se sobra algum caractere corrompido.
"""
import json
import sys

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


def simular_jogo(content: str) -> str:
    """
    O que o jogo faz (segundo o Player.log):
      - recebe o texto
      - trata como Latin-1 (cada char vira 1 byte)
      - reencoda em UTF-8 ao enviar ao TTS

    Se o servidor mandou ASCII puro (\\uXXXX), isso e uma operacao segura.
    Se mandou UTF-8 cru, os acentos e espacos se corrompem.
    """
    try:
        latin = content.encode("latin-1")
        return latin.decode("utf-8", errors="replace")
    except UnicodeEncodeError:
        # caracteres fora de Latin-1: o jogo os perde
        return content.encode("latin-1", "replace").decode("utf-8", "replace")


def problemas(texto: str) -> list[str]:
    """Detecta sintomas de mojibake."""
    p = []
    if "Ã" in texto or "Â" in texto:
        p.append("mojibake de acento (Ã/Â)")
    if "\u2420" in texto:
        p.append("U+2420 substituiu espacos")
    if "\ufffd" in texto:
        p.append("caractere de substituicao U+FFFD")
    if any(0x80 <= ord(c) <= 0x9F for c in texto):
        p.append("byte de controle C1 no texto")
    return p


def main() -> int:
    body = {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "Voce e a Brittany, 26, do Rio. Fala informal, com girias cariocas. UMA frase."},
            {"role": "user", "content": json.dumps([{"player_dialogue": "oi, tudo bem? o que voce queria?"}])},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "gameplay_caller_turn", "strict": True, "schema": SCHEMA}},
        "max_tokens": 200,
    }

    print("=" * 78)
    print(" TESTE DE ENCODING (bug que cortava a voz)")
    print("=" * 78)

    r = httpx.post(URL, json=body, timeout=300)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        return 1

    raw = r.content
    content = r.json()["choices"][0]["message"]["content"]

    print(f"\n1) BYTES CRUS DA RESPOSTA (primeiros 80):")
    print("   ", raw[:80])

    non_ascii = [b for b in raw if b > 127]
    print(f"\n2) BYTES NAO-ASCII: {len(non_ascii)}")
    if non_ascii:
        print("   PROBLEMA: a resposta nao e ASCII puro (acentos vai quebrar no jogo)")
        print("   ", [hex(b) for b in non_ascii[:10]])
    else:
        print("   OK: ASCII puro -> atravessa a camada Latin-1 sem perda")

    print(f"\n3) content recebido:")
    print("   ", content[:200])

    print(f"\n4) SIMULACAO DA CAMADA DO JOGO (latin-1 -> utf-8):")
    entregue = simular_jogo(content)
    print("   ", entregue[:200])

    print(f"\n5) PROBLEMAS DETECTADOS:")
    prob = problemas(entregue)
    if prob:
        for p in prob:
            print(f"   [X] {p}")
    else:
        print("   [OK] nenhum - o texto chega limpo ao TTS")

    # o JSON precisa continuar parseavel depois da ida e volta
    print(f"\n6) O JOGO CONSEGUE PARSEAR?")
    try:
        obj = json.loads(entregue)
        d = obj.get("dialogue", "")
        print(f"   [OK] dialogue: {d}")
        if any(0x80 <= ord(c) <= 0x9F for c in d):
            prob.append("controle C1 no dialogue")
            print("   [X] dialogue tem bytes de controle")
    except json.JSONDecodeError as e:
        prob.append(f"JSON invalido apos latin-1: {e}")
        print(f"   [X] JSON quebrou: {e}")

    print("\n" + "=" * 78)
    print(" SUCESSO" if not prob else f" FALHOU ({len(prob)} problema(s))")
    print("=" * 78)
    return 0 if not prob else 1


if __name__ == "__main__":
    sys.exit(main())
