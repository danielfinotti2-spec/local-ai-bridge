"""
Compara parametros do Ollama lado a lado, na MESMA conversa.

Uso:
    python comparar_params.py                 # compara presets
    python comparar_params.py --preset 0.85   # roda so um

Mede o que importa de verdade:
  - repeticao de fala (o sintoma de "burra")
  - variedade de abertura
  - coerencia do estado (trust/emotion)
  - latencia
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time

import httpx

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:7b"

SYSTEM = """## Task / Role

- You are a helpful assistant in a customer-support roleplay. The user is the customer. Answer naturally and stay in character.
KEEP REPLIES TO 1 sentence.

## Constraints / EMOTION

- Choose from TRUSTING, SUSPICIOUS, ANGRY and NEUTRAL.
"""

CTX = [
    {"language": {"name": "Portuguese", "code": "pt"}},
    {"customer_info": {"name": "Brittany", "age": 26, "location": "Rio de Janeiro",
                       "occupation": "Beauty influencer"}},
    {"initial_state": {"trust_percent": 50, "emotion": "NEUTRAL"}},
]

ROTEIRO = [
    "oi, tudo bem? o que voce queria?",
    "opa",
    "sim o que a senhora precisa?",
    "e da onde a senhora e?",
    "qual a melhor forma de pagamento pra senhora?",
    "aceita pix?",
    "mas eu nao tenho conta em banco, como faz?",
    "ta bom, manda o link entao",
]

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
    "additionalProperties": False,
}

NUDGE = (
    "\n\n[IDIOMA E ESTILO - OBRIGATORIO]"
    "\n- Portugues do Brasil falado e informal."
    "\n- UMA frase curta, no maximo 2."
    "\n- Reaja ao que o jogador disse AGORA. Nao repita falas anteriores."
)

PRESETS = {
    "ATUAL (temp 0.6, rp 1.08)": {"temperature": 0.6, "repeat_penalty": 1.08, "num_ctx": 6144},
    "ANTERIOR (temp 0.85, rp 1.2)": {"temperature": 0.85, "repeat_penalty": 1.2, "num_ctx": 6144},
    "MEIO (temp 0.75, rp 1.12)": {"temperature": 0.75, "repeat_penalty": 1.12, "num_ctx": 6144},
    "RIGIDO (temp 0.5, rp 1.02)": {"temperature": 0.5, "repeat_penalty": 1.02, "num_ctx": 6144},
}


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def rodar(preset: dict) -> dict:
    hist = [{"role": "system", "content": SYSTEM + NUDGE}]
    falas, latencias, trusts, emotions = [], [], [], []

    for fala in ROTEIRO:
        hist.append({"role": "user", "content": json.dumps(CTX + [{"player_dialogue": fala}])})
        payload = {
            "model": MODEL,
            "messages": hist,
            "stream": False,
            "format": SCHEMA,
            "keep_alive": "30m",
            "options": {
                "temperature": preset["temperature"],
                "top_p": 0.95,
                "repeat_penalty": preset["repeat_penalty"],
                "repeat_last_n": 256,
                "num_ctx": preset["num_ctx"],
                "num_predict": 220,
            },
        }
        t0 = time.time()
        try:
            r = httpx.post(OLLAMA, json=payload, timeout=300)
            r.raise_for_status()
            content = r.json()["message"]["content"]
        except Exception as e:
            return {"erro": str(e)}
        latencias.append(time.time() - t0)

        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            obj = {}
        d = str(obj.get("dialogue") or "")
        falas.append(d)
        trusts.append(obj.get("trust_percent"))
        emotions.append(obj.get("emotion"))
        hist.append({"role": "assistant", "content": content})

    # metricas
    normed = [norm(f) for f in falas]
    repetidas = len(normed) - len(set(normed))

    # repeticao de abertura (3 palavras)
    aberturas = [" ".join(f.split()[:3]).lower() for f in falas if len(f.split()) >= 3]
    ab_rep = len(aberturas) - len(set(aberturas))

    # variacao de trust: quantos valores distintos
    tr = [t for t in trusts if isinstance(t, int)]
    tr_distintos = len(set(tr))
    tr_variou = (max(tr) - min(tr)) if tr else 0

    # diversidade lexical
    palavras = [w for f in falas for w in norm(f).split()]
    diversidade = len(set(palavras)) / len(palavras) if palavras else 0

    # tamanho medio
    tam = statistics.mean(len(f.split()) for f in falas) if falas else 0

    return {
        "falas": falas,
        "trusts": trusts,
        "emotions": emotions,
        "repetidas": repetidas,
        "aberturas_repetidas": ab_rep,
        "trust_distintos": tr_distintos,
        "trust_amplitude": tr_variou,
        "diversidade": diversidade,
        "palavras_media": tam,
        "latencia_media": statistics.mean(latencias),
        "latencia_p95": sorted(latencias)[int(len(latencias) * 0.95) - 1] if len(latencias) > 1 else latencias[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", help="roda apenas um preset (nome exato)")
    args = ap.parse_args()

    alvos = {args.preset: PRESETS[args.preset]} if args.preset else PRESETS
    resultados = {}

    for nome, preset in alvos.items():
        print(f"\n{'=' * 78}\n  {nome}\n{'=' * 78}")
        res = rodar(preset)
        if "erro" in res:
            print(f"  ERRO: {res['erro']}")
            continue
        resultados[nome] = res
        for i, f in enumerate(res["falas"], 1):
            print(f"  [{i}] {f}")
        print(f"  trust: {res['trusts']}")
        print(f"  emocao: {res['emotions']}")

    if len(resultados) > 1:
        print(f"\n\n{'=' * 92}")
        print(" COMPARACAO")
        print("=" * 92)
        cab = f"{'preset':<30} {'repet':>6} {'abert':>6} {'trust_var':>10} {'divers':>8} {'lat(s)':>8}"
        print(cab)
        print("-" * 92)
        for nome, r in resultados.items():
            print(f"{nome:<30} {r['repetidas']:>6} {r['aberturas_repetidas']:>6} "
                  f"{r['trust_distintos']:>10} {r['diversidade']:>8.2f} {r['latencia_media']:>8.2f}")

        print("\n COMO LER:")
        print("   repet  = falas identicas (quanto menor, melhor)")
        print("   abert  = aberturas 'Oi, tudo...' repetidas (menor melhor)")
        print("   trust_var = quantos valores distintos de trust (mais = reage melhor)")
        print("   divers  = variedade de palavras (maior = mais variado)")
        print("   lat    = segundos por resposta (menor = menos filler no jogo)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
