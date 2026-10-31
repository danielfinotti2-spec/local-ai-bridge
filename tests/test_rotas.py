"""
Testa TODAS as rotas do servidor local no formato esperado pelo jogo.
Inclui o parse do OpenRouterModelCatalog (tela de config do host).
"""
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"

# Campos que OpenRouterModelCatalog.Parse EXIGE (senao o jogo diz
# "Could not load OpenRouter models")
def parse_like_client(json_text: str) -> list[dict]:
    data = json.loads(json_text)
    src = data.get("data")
    if not isinstance(src, list):
        raise ValueError("sem 'data'")
    out = []
    for item in src:
        mid = item.get("id")
        arch = item.get("architecture") or {}
        if not mid:
            continue
        if "text" not in (arch.get("input_modalities") or []):
            continue
        if "text" not in (arch.get("output_modalities") or []):
            continue
        out.append(item)
    return out


def check(name: str, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as e:
        print(f"  [X] {name}: {type(e).__name__}: {e}")
        return False
    print(f"  [{'OK' if ok else 'X'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 76)
    print(" TESTE DE ROTAS - local-ai-bridge")
    print("=" * 76)

    results = []

    def t_root():
        r = httpx.get(f"{BASE}/", timeout=10)
        j = r.json()
        return r.status_code == 200 and "model" in j, f"{j.get('model')}"

    def t_health():
        r = httpx.get(f"{BASE}/health", timeout=20)
        j = r.json()
        return j.get("ok") is True and j.get("model_installed", False), (
            f"modelos={len(j.get('models') or [])} instalado={j.get('model_installed')}")

    def t_models():
        r = httpx.get(f"{BASE}/api/v1/models", timeout=20)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        parsed = parse_like_client(r.text)
        if not parsed:
            return False, "o parser do jogo descartaria TODOS os modelos"
        ids = [p["id"] for p in parsed]
        # valida os campos que o jogo le
        first = parsed[0]
        need = [first.get("id"), first.get("name"), first.get("pricing"),
                first.get("top_provider"), first.get("supported_parameters")]
        if any(x is None for x in need):
            return False, "faltam campos obrigatorios"
        return True, f"{len(parsed)} modelo(s) aceitos: {ids}"

    def t_models_v1():
        r = httpx.get(f"{BASE}/v1/models", timeout=20)
        return r.status_code == 200 and bool(parse_like_client(r.text)), None

    def t_404_route():
        # rota inexistente deve dar 404 (nao travar o servidor)
        r = httpx.post(f"{BASE}/api/v1/nada", json={}, timeout=10)
        return r.status_code == 404, f"HTTP {r.status_code}"

    def t_bad_json():
        r = httpx.post(f"{BASE}/api/v1/chat/completions",
                       content=b"{isso nao e json",
                       headers={"Content-Type": "application/json"}, timeout=20)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return r.status_code == 400 and "error" in j, f"HTTP {r.status_code}"

    def t_missing_messages():
        r = httpx.post(f"{BASE}/api/v1/chat/completions", json={"model": "x"}, timeout=20)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return r.status_code == 400 and "error" in j, f"HTTP {r.status_code}"

    def t_no_auth():
        # sem header Authorization: deve funcionar igual
        body = {
            "model": "qwen2.5:7b",
            "messages": [{"role": "system", "content": "Voce e o Zeca, feirante em Recife."},
                         {"role": "user", "content": "Oi, tudo bem?"}],
            "max_tokens": 120,
        }
        r = httpx.post(f"{BASE}/api/v1/chat/completions", json=body, timeout=180)
        j = r.json()
        ok = r.status_code == 200 and j["choices"][0]["message"]["content"].strip() != ""
        return ok, j["choices"][0]["message"]["content"][:80]

    def t_bearer():
        body = {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "Diga oi"}],
            "max_tokens": 40,
        }
        r = httpx.post(f"{BASE}/api/v1/chat/completions", json=body,
                       headers={"Authorization": "Bearer sk-qualquer-coisa"}, timeout=180)
        return r.status_code == 200, None

    def t_json_schema():
        schema = {
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
        body = {
            "model": "qwen2.5:7b",
            "messages": [{"role": "system", "content": "Voce e o Seu Zeca, feirante em Recife. Uma frase."},
                         {"role": "user", "content": "Oi, tudo bem?"}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "structured_turn", "strict": True, "schema": schema}},
            "max_tokens": 300,
        }
        r = httpx.post(f"{BASE}/api/v1/chat/completions", json=body, timeout=300)
        content = r.json()["choices"][0]["message"]["content"]
        obj = json.loads(content)
        exact = set(obj.keys()) == set(schema["required"])
        return exact, content[:90]

    def t_openrouter_extras():
        """Campos que so o OpenRouter entende devem ser ignorados sem erro."""
        body = {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "Oi"}],
            "provider": {"require_parameters": True, "only": ["novita"], "allow_fallbacks": False},
            "modalities": ["text"],
            "transforms": ["middle-out"],
            "session_id": "abc",
            "schema_instruction": "ignore",
            "conversation_protocol": 1,
            "route": "fallback",
            "models": ["a", "b"],
            "reasoning": {"enabled": False},
            "max_tokens": 40,
        }
        r = httpx.post(f"{BASE}/api/v1/chat/completions", json=body, timeout=180)
        return r.status_code == 200, f"HTTP {r.status_code}"

    for name, fn in [
        ("GET  /                          (raiz)", t_root),
        ("GET  /health                    (status)", t_health),
        ("GET  /api/v1/models             (tela de config)", t_models),
        ("GET  /v1/models                 (alias)", t_models_v1),
        ("POST rota inexistente           (404)", t_404_route),
        ("POST JSON invalido              (400)", t_bad_json),
        ("POST messages ausente           (400)", t_missing_messages),
        ("POST sem Authorization          (ok)", t_no_auth),
        ("POST com Bearer                 (ok)", t_bearer),
        ("POST json_schema estrito        (5 campos)", t_json_schema),
        ("POST campos do OpenRouter       (ignorados)", t_openrouter_extras),
    ]:
        results.append(check(name, fn))

    total, passed = len(results), sum(results)
    print("\n" + "=" * 76)
    print(f" {passed}/{total} rotas OK")
    print("=" * 76)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
