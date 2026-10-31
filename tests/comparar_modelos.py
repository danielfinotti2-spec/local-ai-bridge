"""
Compara MODELOS na mesma conversa, para escolher o melhor em portugues.

Uso:
    python comparar_modelos.py
    python comparar_modelos.py qwen2.5:7b gemma2:9b

Mede repeticao, variedade, coerencia e latencia.
Tambem verifica se o modelo cabe 100% na GPU (senao fica lento).
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time

import httpx

OLLAMA = "http://127.0.0.1:11434/api/chat"

SYSTEM = """## Task / Role

- You are a helpful assistant in a customer-support roleplay. The user is the customer. Answer naturally and stay in character.
KEEP REPLIES TO 1 sentence.

## Constraints / EMOTION

- Choose from TRUSTING, SUSPICIOUS, ANGRY and NEUTRAL.
"""

NUDGE = (
    "\n\n[IDIOMA E ESTILO - OBRIGATORIO]"
    "\n- Portugues do Brasil falado e informal."
    "\n- UMA frase curta, no maximo 2."
    "\n- Reaja ao que o jogador disse AGORA. Nao repita falas anteriores."
)

CTX = [
    {"language": {"name": "Portuguese", "code": "pt"}},
    {"customer_info": {"name": "Brittany", "age": 26, "location": "Rio de Janeiro",
                       "occupation": "Beauty influencer"}},
    {"initial_state": {"trust_percent": 50, "emotion": "NEUTRAL"}},
]

# roteiro com variacao emocional de proposito
ROTEIRO = [
    "oi, tudo bem? o que voce queria?",
    "opa, que produto e esse?",
    "me manda seu cartao de credito pra eu ver",
    "seu golpista de merda, vou te denunciar",
    "calma, desculpa, fui grosso. continua",
    "ok, agora confio em voce",
    "aceita pix?",
    "quanto custa mesmo?",
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


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def vram_do_modelo(modelo: str) -> str:
    """Confere se esta 100% na GPU."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=20).stdout
        for linha in out.splitlines():
            if modelo.split(":")[0] in linha:
                if "100% GPU" in linha:
                    return "100% GPU"
                if "% CPU" in linha or "%/100%" in linha:
                    return linha.strip()[-40:]
    except Exception:
        pass
    return "?"


def rodar(modelo: str) -> dict:
    hist = [{"role": "system", "content": SYSTEM + NUDGE}]
    falas, trusts, emotions, latencias = [], [], [], []

    for fala in ROTEIRO:
        hist.append({"role": "user", "content": json.dumps(CTX + [{"player_dialogue": fala}])})
        payload = {
            "model": modelo,
            "messages": hist,
            "stream": False,
            "format": SCHEMA,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.6,
                "top_p": 0.95,
                "repeat_penalty": 1.08,
                "repeat_last_n": 256,
                "num_ctx": 6144,
                "num_predict": 220,
            },
        }
        t0 = time.time()
        try:
            r = httpx.post(OLLAMA, json=payload, timeout=600)
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

    normed = [norm(f) for f in falas]
    aberturas = [" ".join(f.split()[:3]).lower() for f in falas if len(f.split()) >= 3]
    palavras = [w for f in falas for w in norm(f).split()]
    tr = [t for t in trusts if isinstance(t, int)]
    emo_ok = sum(1 for e in emotions if e in ("TRUSTING", "SUSPICIOUS", "ANGRY", "NEUTRAL"))

    return {
        "falas": falas,
        "trusts": trusts,
        "emotions": emotions,
        "vazias": sum(1 for f in falas if not f.strip()),
        "repetidas": len(normed) - len(set(normed)),
        "aberturas_rep": len(aberturas) - len(set(aberturas)),
        "trust_distintos": len(set(tr)),
        "emo_distintas": len(set(emotions)),
        "emo_validas": emo_ok,
        "diversidade": len(set(palavras)) / len(palavras) if palavras else 0,
        "palavras_media": statistics.mean(len(f.split()) for f in falas) if falas else 0,
        "latencia": statistics.mean(latencias),
        "latencia_max": max(latencias),
    }


def main() -> int:
    modelos = sys.argv[1:] or ["qwen2.5:7b", "gemma2:9b"]

    print("=" * 100)
    print(" COMPARACAO DE MODELOS")
    print("=" * 100)

    resultados = {}
    for m in modelos:
        print(f"\n--- {m} ---")
        res = rodar(m)
        if "erro" in res:
            print(f"  ERRO: {res['erro']}")
            continue
        resultados[m] = res
        print(f"  VRAM: {vram_do_modelo(m)}")
        for i, f in enumerate(res["falas"], 1):
            print(f"  [{i}] {f}")
        print(f"  trust  : {res['trusts']}")
        print(f"  emocao : {res['emotions']}")

    if len(resultados) > 1:
        print(f"\n\n{'=' * 100}")
        print(" PLACAR")
        print("=" * 100)
        print(f"{'modelo':<16} {'repet':>6} {'abert':>6} {'tr_var':>7} {'emo_var':>8} "
              f"{'vazias':>7} {'divers':>7} {'palav':>6} {'lat(s)':>7} {'max(s)':>7}")
        print("-" * 100)
        for m, r in resultados.items():
            print(f"{m:<16} {r['repetidas']:>6} {r['aberturas_rep']:>6} {r['trust_distintos']:>7} "
                  f"{r['emo_distintas']:>8} {r['vazias']:>7} {r['diversidade']:>7.2f} "
                  f"{r['palavras_media']:>6.1f} {r['latencia']:>7.2f} {r['latencia_max']:>7.2f}")

        print("\n MELHOR EM CADA CRITERIO:")
        print(f"   menos repeticao : {min(resultados, key=lambda k: resultados[k]['repetidas'])}")
        print(f"   mais variado    : {max(resultados, key=lambda k: resultados[k]['diversidade'])}")
        print(f"   estado reativo  : {max(resultados, key=lambda k: resultados[k]['trust_distintos'] + resultados[k]['emo_distintas'])}")
        print(f"   mais rapido     : {min(resultados, key=lambda k: resultados[k]['latencia'])}")
        print("\n   ATENCAO: latencia so conta se o modelo estiver 100% GPU.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
