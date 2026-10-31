"""
local-ai-bridge - servidor compativel com a API do OpenAI/OpenRouter
                   que encaminha para o Ollama local.

    Cliente  --HTTP-->  127.0.0.1:8000  -->  Ollama  -->  GPU

Foco do projeto (educacional):
  * /v1/models devolve o formato completo que clientes OpenRouter esperam
    (architecture, pricing, supported_parameters, top_provider).
  * Aceita e ignora campos exclusivos do OpenRouter (provider, modalities,
    transforms, session_id, schema_instruction...).
  * Structured output: schema implicito + conformidade + recuperacao parcial.
  * Anti-repeticao em duas camadas (prompt + similaridade Jaccard).
  * Reparo de mojibake UTF-8 <-> Latin-1 na entrada e na saida.
  * Erros no formato OpenAI ("error": {"message", "type", "code"}).
  * keep_alive no Ollama: modelo fica na GPU entre requisicoes.

Uso:  python server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import random
import unicodedata
import re
import time
from typing import Any

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --------------------------------------------------------------------------
# Configuracao
# --------------------------------------------------------------------------

AQUI = pathlib.Path(__file__).resolve().parent


def _cfg() -> dict:
    try:
        p = AQUI / "config.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_CFG = _cfg()

OLLAMA_URL = os.environ.get("LOCALAI_OLLAMA_URL") or _CFG.get("ollama_url") or "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("LOCALAI_MODEL") or _CFG.get("model") or "qwen2.5:7b"
PORT = int(os.environ.get("LOCALAI_PORT") or _CFG.get("port") or 8000)

NUM_CTX = int(os.environ.get("LOCALAI_NUM_CTX") or _CFG.get("num_ctx") or 8192)
NUM_PREDICT = int(os.environ.get("LOCALAI_NUM_PREDICT") or _CFG.get("num_predict") or 1024)
TEMPERATURE = float(os.environ.get("LOCALAI_TEMPERATURE") or _CFG.get("temperature") or 0.85)
TOP_P = float(os.environ.get("LOCALAI_TOP_P") or _CFG.get("top_p") or 0.95)
KEEP_ALIVE = os.environ.get("LOCALAI_KEEP_ALIVE") or _CFG.get("keep_alive") or "30m"
# Penalidade de repeticao: alta demais deixa o modelo robotico e travado,
# baixa demais faz ele repetir. 1.08 ficou melhor que 1.2 nos testes.
REPEAT_PENALTY = float(os.environ.get("LOCALAI_REPEAT_PENALTY") or _CFG.get("repeat_penalty") or 1.08)
REPEAT_LAST_N = int(os.environ.get("LOCALAI_REPEAT_LAST_N") or _CFG.get("repeat_last_n") or 256)
FORCE_PT = (os.environ.get("LOCALAI_FORCE_PT") or str(_CFG.get("force_pt", 1))) not in ("0", "false", "False")
# tempo maximo de uma resposta (o jogo usa ~60-120s no provider direto)
REQUEST_TIMEOUT = float(os.environ.get("LOCALAI_TIMEOUT") or _CFG.get("timeout") or 180.0)

LOG_DIR = AQUI / "logs"

# ---------------------------------------------------------------------------
# !!! CORRECAO CRITICA DE ENCODING !!!
# ---------------------------------------------------------------------------
# O jogo le o JSON da resposta como Latin-1/Windows-1252 em algum ponto
# (NetworkWriter / TTS). Resultado observado no Player.log:
#     "Cara, eu tÃ´ ligando"   ->   "Cara,Ã¢\x90\xa0euÃ¢\x90\xa0tÃƒÂ´Ã¢\x90\xa0ligando"
# O espaco (0x20) virava o trio U+00E2 U+0090 U+00A0 (= "â " U+2420 em UTF-8),
# e "Ã´" (C3 B4) virava "ÃƒÂ´".
#
# Isso fazia o TTS receber texto cheio de lixo e a voz sair cortada/robotica,
# alem de confundir o proprio modelo nos turnos seguintes.
#
# Solucao: serializar o JSON com ensure_ascii=True. Todo caractere nao-ASCII
# vira sequencia \uXXXX (escape ASCII puro), entao nada se perde ao atravessar
# a camada Latin-1. O Newtonsoft.Json do jogo desescapa corretamente.
JSON_ESCAPE = (os.environ.get("LOCALAI_JSON_ASCII") or str(_CFG.get("json_ascii", 1))) not in ("0", "false", "False")

# Quando o jogo nao manda response_format, deduzimos este schema pelo system
# prompt. Sem isso o modelo responde texto livre e o jogo pede "repair"
# (cada repair custa uma chamada inteira a mais).
FORCE_SCHEMA = (os.environ.get("LOCALAI_FORCE_SCHEMA") or str(_CFG.get("force_schema", 1))) not in ("0", "false", "False")

TURN_SCHEMA_IMPLICITO: dict[str, Any] = {
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

app = FastAPI(title="LOCAL-AI bridge", version="2.0")

# UnityWebRequest nao valida CORS, mas navegadores/diagnosticos validam.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------
# Constantes de conteudo
# --------------------------------------------------------------------------

PT_NUDGE = (
    "\n\n[IDIOMA E ESTILO - OBRIGATORIO]"
    "\n- Responda em portugues do Brasil FALADO e informal (nao formal, nao de Portugal)."
    "\n- Use girias naturais (cara, viu, poxa, eita, ta, beleza) quando couber."
    "\n- Fale como numa LIGACAO DE TELEFONE: UMA frase curta, no maximo 2. "
    "Nada de paragrafo, nada de narracao."
    "\n- NUNCA use asteriscos, marcacao, emojis ou descricao de acao. So a fala."
    "\n- Reaja ao que o jogador disse AGORA. Nao repita falas anteriores."
    "\n- Se o jogador disse algo confuso ou sem sentido, reaja a isso "
    "(estranhe, peca pra repetir) em vez de voltar ao assunto antigo."
)

# Usado so quando o modelo devolve dialogue vazio (o jogo rejeita vazio).
PT_FALLBACKS = [
    "O que voce ta falando? Fala direito que eu nao entendi nada.",
    "Ha? Repete ai, que eu nao peguei essa.",
    "Calma la, explica de novo que eu to perdido.",
]

COMEDY_HINTS = (
    "cara", "velho", "porra", "merda", "cacete", "puts", "eita", "oxe", "oxi",
    "mano", "doidera", "sacanagem", "besteira", "ta maluco", "que isso",
)

# Opcoes do Ollama que NAO podem ir no corpo da requisicao.
_UNSUPPORTED_SCHEMA_KEYS = {
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
    "minItems", "maxItems", "uniqueItems", "pattern", "format",
    "minProperties", "maxProperties", "default", "$schema", "$id",
    "$ref", "$defs", "definitions", "dependentRequired", "propertyNames",
    "contentEncoding", "contentMediaType", "deprecated", "readOnly", "writeOnly",
    "unevaluatedProperties", "const",
}


# --------------------------------------------------------------------------
# Sanitizacao de schema
# --------------------------------------------------------------------------

def clean_schema(node: Any) -> Any:
    """Remove chaves que o llama.cpp (grammar GBNF) nao aceita."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in _UNSUPPORTED_SCHEMA_KEYS:
                continue
            out[k] = clean_schema(v)
        # const -> enum de 1 valor
        if "enum" in out and isinstance(out["enum"], list) and len(out["enum"]) == 1:
            out.setdefault("type", type(out["enum"][0]).__name__ if not isinstance(out["enum"][0], str) else "string")
        # anyOf/oneOf de 1 elemento: promove
        for combo in ("anyOf", "oneOf"):
            if isinstance(out.get(combo), list) and len(out[combo]) == 1:
                only = out.pop(combo)[0]
                if isinstance(only, dict):
                    out.update(only)
        return out
    if isinstance(node, list):
        return [clean_schema(x) for x in node]
    return node


def extract_schema(body: dict) -> dict | None:
    rf = body.get("response_format")
    if not isinstance(rf, dict):
        return None
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        sc = js.get("schema")
        if isinstance(sc, dict):
            return sc
    if rf.get("type") == "json_object":
        return {"type": "object"}
    return None


# --------------------------------------------------------------------------
# Conformidade estrita (o jogo exige campos EXATOS)
# --------------------------------------------------------------------------

def _default_for(node: dict) -> Any:
    t = node.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x not in ("null",)), "string")
    if t == "object":
        return {k: _default_for(v if isinstance(v, dict) else {}) for k, v in (node.get("properties") or {}).items()}
    if t == "array":
        return []
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    if t in ("integer", "number"):
        lo = node.get("minimum")
        return int(lo) if isinstance(lo, (int, float)) else (0 if t == "integer" else 0.5)
    if t == "boolean":
        return False
    return ""


def _coerce(value: Any, node: Any) -> Any:
    if not isinstance(node, dict):
        return value
    t = node.get("type")
    try:
        if t == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, str):
                value = re.sub(r"[^0-9.\-]", "", value) or "0"
            v = int(float(value))
            lo, hi = node.get("minimum"), node.get("maximum")
            if isinstance(lo, (int, float)):
                v = max(v, int(lo))
            if isinstance(hi, (int, float)):
                v = min(v, int(hi))
            return v
        if t == "number":
            v = float(value)
            lo, hi = node.get("minimum"), node.get("maximum")
            if isinstance(lo, (int, float)):
                v = max(v, float(lo))
            if isinstance(hi, (int, float)):
                v = min(v, float(hi))
            return v
        if t == "string":
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return "" if value is None else str(value)
        if t == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "sim", "yes")
            return bool(value)
        if t == "array":
            arr = value if isinstance(value, list) else [value]
            item = node.get("items") or {}
            return [_coerce(x, item) for x in arr]
        if t == "object":
            if isinstance(value, str):
                parsed = extract_json(value)
                value = parsed if isinstance(parsed, dict) else {}
            if not isinstance(value, dict):
                value = {}
            props = node.get("properties") or {}
            if props:
                return {k: _coerce(value.get(k, _default_for(v if isinstance(v, dict) else {})), v)
                        for k, v in props.items()}
            return value
    except (TypeError, ValueError):
        return _default_for(node)
    return value


def conform(obj: Any, schema: dict) -> dict:
    """Garante EXATAMENTE as propriedades do schema (jogo compara a contagem)."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return obj if isinstance(obj, dict) else {}
    if not isinstance(obj, dict):
        obj = {}
    props = schema.get("properties") or {}
    if not props:
        return obj
    required = schema.get("required") or []
    out: dict[str, Any] = {}
    for key, sub in props.items():
        if key in obj:
            out[key] = _coerce(obj[key], sub)
        elif key in required:
            out[key] = _default_for(sub if isinstance(sub, dict) else {})
            log(f"[!] campo obrigatorio '{key}' ausente; usando default")

    # O qwen2.5 tende a devolver 0.0 fixo em 'comedy' e 0 em 'trust_percent'.
    # O jogo usa esses campos para a barra de confianca e o placar de humor,
    # entao deixamos valores puramente inertes um pouco acima do piso.
    if "comedy" in out and isinstance(out["comedy"], (int, float)):
        d = str(out.get("dialogue") or "")
        parece_engracado = any(h in d.lower() for h in COMEDY_HINTS)
        if out["comedy"] < 0.35:
            out["comedy"] = 0.55 if parece_engracado else 0.4

    # O jogo espera "trust_percent": se o modelo devolver 0 ou 100 fixos
    # (sinal de que nao avaliou), nao deixa a barra travar num extremo.
    tp = out.get("trust_percent")
    if isinstance(tp, int) and (tp <= 0 or tp >= 100):
        log(f"[!] trust_percent={tp} suspeito; ajustando para a faixa segura")
        out["trust_percent"] = 35 if tp <= 0 else 85

    return out


def _recover_partial(text: str) -> Any:
    """JSON cortado por limite de tokens: salva o campo dialogue."""
    m = re.search(r'"dialogue"\s*:\s*"((?:[^"\\]|\\.)*)', text or "", re.S)
    if not m:
        return None
    raw = m.group(1)
    try:
        d = json.loads('"' + raw + '"')
    except json.JSONDecodeError:
        d = raw.rstrip("\\")
    return {"dialogue": d}


def extract_json(text: str) -> Any:
    """Extrai o primeiro objeto JSON balanceado, mesmo cercado de texto."""
    if not text:
        return None
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return _recover_partial(text)
    return _recover_partial(text)


# --------------------------------------------------------------------------
# Mensagens
# --------------------------------------------------------------------------

def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            else:
                parts.append(str(p))
        return "\n".join(x for x in parts if x)
    if content is None:
        return ""
    return str(content)


def prepare_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role") or "user"
        if role not in ("system", "user", "assistant"):
            role = "user"
        text = flatten_content(m.get("content"))
        # o jogo pode entregar UTF-8 lido como Latin-1 (acentos quebrados:
        # "indicaÃƒÂ§ÃƒÂµes"). Isso degrada MUITO a qualidade da resposta.
        text = fix_encoding(text)
        # o Ollama exige todo content nao-vazio
        if not text.strip() and role == "assistant":
            continue
        out.append({"role": role, "content": text})

    if FORCE_PT:
        if out and out[0]["role"] == "system":
            out[0] = {"role": "system", "content": out[0]["content"] + PT_NUDGE}
        else:
            out.insert(0, {"role": "system", "content": PT_NUDGE.strip()})
    return out


def _prev_assistant_objects(messages: list[dict]) -> list[dict]:
    """Todas as falas anteriores da vitima, em ordem cronologica."""
    out: list[dict] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        obj = extract_json(flatten_content(m.get("content")))
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _spoken_lines(messages: list[dict]) -> list[str]:
    """Falas ja ditas (texto), para deteccao de repeticao."""
    linhas = []
    for obj in _prev_assistant_objects(messages):
        d = str(obj.get("dialogue") or "").strip()
        if d:
            linhas.append(d)
    return linhas


def trust_hint(messages: list[dict], schema: dict | None) -> str | None:
    """
    Injeta o estado da conversa para o modelo:
      - trust/emotion atuais (para progredir, nao oscilar)
      - falas ja ditas (para NAO repetir)
      - numero do turno (para variar)
    """
    if not schema:
        return None
    props = schema.get("properties") or {}
    if "trust_percent" not in props:
        return None

    anteriores = _prev_assistant_objects(messages)
    falas = _spoken_lines(messages)
    turno = len(falas) + 1

    if not anteriores:
        lo = (props.get("trust_percent") or {}).get("minimum", 0)
        return (
            f"[ESTADO] Turno 1. trust_percent={lo}, emotion=NEUTRAL. "
            "Responda ao que o jogador acabou de dizer, em UMA frase curta, "
            "em portugues do Brasil, com personalidade."
        )

    prev = anteriores[-1]
    try:
        n = int(prev.get("trust_percent", 0))
    except (TypeError, ValueError):
        n = 0
    emo = prev.get("emotion", "NEUTRAL")

    # tendencia: olha os 3 ultimos para nao oscilar
    hist = []
    for o in anteriores[-3:]:
        try:
            hist.append(int(o.get("trust_percent", 0)))
        except (TypeError, ValueError):
            pass
    tendencia = ""
    if len(hist) >= 2:
        if hist[-1] > hist[0]:
            tendencia = ("Confianca vinha SUBINDO. Continue avaliando pelo merito de cada fala; "
                         "pode cair se ele falar merda.")
        elif hist[-1] < hist[0]:
            # IMPORTANTE: nao dizer "so suba se redimir" - isso congela a queda
            # e a vitima fica hostil ate desligar.
            tendencia = ("Confianca vinha CAINDO. Avalie ESTA fala pelo que ele disse agora: "
                         "se ele foi educado, coerente ou engracado, PODE subir normalmente.")
        else:
            tendencia = "Confianca ESTAVEL. Mova +/- conforme a qualidade da fala dele."

    # falas ja usadas (limita para nao estourar contexto)
    usadas = ""
    aberturas = ""
    if falas:
        ultimas = falas[-4:]
        usadas = (
            "\n[FALAS JA DITAS - NAO REPITA NEM PARAFRASEIE] "
            + " | ".join(f'"{f}"' for f in ultimas)
        )
        # primeiras 3 palavras de cada fala: evita "Oi, tudo boa?" repetido
        inicios = [ " ".join(f.split()[:3]) for f in ultimas if f.split() ]
        inicios = [i for i in inicios if i]
        if inicios:
            aberturas = " / ".join(f'"{i}"' for i in inicios)

    inicio_regra = ""
    if aberturas:
        inicio_regra = (
            f"\n2. NAO COMECE com estas aberturas ja usadas: {aberturas}. "
            "Comece de outro jeito (ou va direto ao ponto, sem cumprimentar)."
        )

    return (
        f"[ESTADO] Turno {turno}. trust_percent={n}, emotion={emo}. {tendencia}"
        f"{usadas}"
        "\n\nREGRAS:"
        "\n1. Voce LEMBRA de tudo que ja foi dito. Fatos, nomes e numeros citados "
        "pelo jogador continuam valendo - use-os normalmente."
        "\n2. Nao repita uma frase ja dita. Responda ao que ele disse AGORA com "
        "uma fala nova (mas pode se referir ao que ja foi conversado)."
        f"{inicio_regra}"
        "\n3. Se ele falou algo confuso ou mal transcrito, reaja a isso "
        "(peca pra repetir, estranhe) em vez de voltar ao assunto antigo."
        "\nResponda em UMA frase curta, em portugues do Brasil falado (nao formal)."
    )


def strip_artifacts(text: str) -> str:
    """Remove lixo comum do modelo: prefixos de papel, marcadores, aspas."""
    if not text:
        return ""
    t = text.strip()
    # marcador de desligar/acoes do jogo que vazou para o texto
    t = re.sub(r"\[\[?[A-Z_]{3,}\]?\]?", "", t)
    # prefixos tipo "Maria: " / "M | " (so com ':' ou '|'; nunca palavras reais)
    t = re.sub(r"^(?:[A-Za-z]{1,3}\s*\||[A-Za-z]{1,12}:\s+)", "", t).strip()
    # letra solta tipo "M Cara, eu..." (so se a proxima palavra comeca maiuscula)
    t = re.sub(r"^[A-Z]\s+(?=[A-Z])", "", t).strip()
    # aspas envolvendo a fala inteira
    if len(t) > 2 and t[0] in "\"'" and t[-1] == t[0]:
        t = t[1:-1].strip()
    # remove caracteres de controle (quebram o TTS) mas mantem acentos
    t = "".join(ch for ch in t if ch == "\n" or not unicodedata.category(ch).startswith("C"))
    # colapsa espacos/quebras
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def fix_encoding(text: str) -> str:
    """Corrige mojibake (UTF-8 lido como Latin-1), ex.: 'Ã§' -> 'ç'."""
    if not text or not re.search("[\u00c3\u00c2]", text):
        return text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        if len(re.findall("[\u00c3\u00c2]", fixed)) < len(re.findall("[\u00c3\u00c2]", text)):
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


RECENTES: list[str] = []  # ultimas falas devolvidas (memoria do proprio servidor)


def encurtar(text: str, limite: int = 110) -> str:
    """Fala curta pro TTS nao engasgar: ~1 frase (ate ~110 caracteres)."""
    partes = re.findall(r".+?(?:[.!?]+(?=\s|$)|$)", text.strip())
    partes = [x.strip() for x in partes if x.strip()]
    if not partes:
        return text
    out = partes[0]
    for x in partes[1:]:
        if len(out) >= 25 or len(out) + 1 + len(x) > limite:
            break
        out += " " + x
    if len(out) > limite + 40:
        out = out[:limite + 40].rsplit(" ", 1)[0].rstrip(",;: ") + "."
    return out


def dedupe_dialogue(text: str, messages: list[dict]) -> str:
    """
    Se a fala repetir literalmente (ou quase) uma anterior, tenta variar.
    O jogo aceita a repeticao, mas fica visivelmente 'burro'.
    """
    if not text:
        return text
    anteriores = list(_spoken_lines(messages)) + RECENTES[-6:]
    if not anteriores:
        return text

    norm = re.sub(r"\W+", " ", text.lower()).strip()
    for ant in anteriores:
        if not ant:
            continue
        a = re.sub(r"\W+", " ", ant.lower()).strip()
        if not a:
            continue
        # igual ou muito parecido (>=85% de sobreposicao de palavras)
        wa, wb = set(norm.split()), set(a.split())
        if not wa or not wb:
            continue
        inter = len(wa & wb)
        jac = inter / len(wa | wb) if (wa | wb) else 0.0
        if norm == a or jac >= 0.7:
            log("[!] fala repetida detectada; variando")
            # reacao natural, sem repetir: pede esclarecimento ou cobra o ponto
            return random.choice([
                "Tá, mas o que você quer de mim, afinal?",
                "Você tá falando sozinho? Me explica isso direito.",
                "Hmm, e isso muda o quê pra mim?",
                "Não entendi onde você quer chegar, fala claro.",
                "Tá bom, e daí? Fala o que você precisa.",
                "Peraí, tá confuso isso. Reformula pra mim.",
            ])

    return text


# --------------------------------------------------------------------------
# Chamada ao Ollama
# --------------------------------------------------------------------------

async def call_ollama(payload: dict) -> dict:
    r = await _client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    if r.status_code >= 400:
        detail = r.text[:400]
        raise RuntimeError(f"HTTP {r.status_code} do Ollama: {detail}")
    return r.json()


def error_json(message: str, code: str = "local_ai_error", status: int = 502) -> JSONResponse:
    """Formato de erro OpenAI/OpenRouter (o jogo le 'error')."""
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}},
        status_code=status,
    )


# --------------------------------------------------------------------------
# Rotas
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return {"service": "LOCAL-AI bridge", "model": OLLAMA_MODEL, "upstream": OLLAMA_URL, "port": PORT}


@app.get("/health")
@app.get("/api/health")
async def health():
    try:
        r = await _client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
        have = OLLAMA_MODEL in names or f"{OLLAMA_MODEL}:latest" in names
        return {"ok": have, "ollama": True, "models": names, "using": OLLAMA_MODEL, "model_installed": have}
    except Exception as e:
        return JSONResponse({"ok": False, "ollama": False, "error": str(e)}, status_code=503)


def _model_entry(model_id: str, name: str) -> dict:
    """
    Formato IDENTICO ao /api/v1/models do OpenRouter.

    OpenRouterModelCatalog.Parse exige:
      - data[] com "id"
      - architecture.input_modalities / output_modalities contendo "text"
      - supported_parameters (array)
      - pricing.prompt / pricing.completion (string numerica)
      - top_provider.max_completion_tokens
    """
    return {
        "id": model_id,
        "name": name,
        "created": int(time.time()),
        "description": "Modelo local rodando na sua GPU via Ollama. Gratis, sem nuvem.",
        "context_length": NUM_CTX,
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "other",
            "instruct_type": None,
        },
        "pricing": {"prompt": "0", "completion": "0", "request": "0", "image": "0"},
        "top_provider": {
            "context_length": NUM_CTX,
            "max_completion_tokens": NUM_PREDICT,
            "is_moderated": False,
        },
        "supported_parameters": [
            "temperature", "top_p", "max_tokens", "stop",
            "frequency_penalty", "presence_penalty", "seed",
        ],
        "per_request_limits": None,
    }


@app.get("/api/v1/models")
@app.get("/v1/models")
async def models():
    """Lista de modelos para a tela de configuracao do jogo."""
    entries = []
    try:
        r = await _client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                n = m.get("name") or ""
                if n:
                    entries.append(_model_entry(n, f"IA Local - {n} (GPU)"))
    except Exception as e:
        log(f"[!] nao consegui listar do Ollama: {e}")

    if not entries:
        entries.append(_model_entry(OLLAMA_MODEL, f"IA Local - {OLLAMA_MODEL} (GPU)"))

    # o modelo configurado vem primeiro
    entries.sort(key=lambda e: (e["id"] != OLLAMA_MODEL, e["id"]))
    return {"data": entries}


@app.post("/api/v1/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background: BackgroundTasks):
    started = time.time()

    # --- corpo ----------------------------------------------------------
    raw = await request.body()
    if len(raw) > 8 * 1024 * 1024:
        return error_json("Corpo da requisicao muito grande.", "body_too_large", 413)
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        return error_json(f"JSON invalido no corpo: {e}", "invalid_json", 400)
    if not isinstance(body, dict):
        return error_json("O corpo deve ser um objeto JSON.", "invalid_body", 400)

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return error_json("'messages' ausente ou vazio.", "missing_messages", 400)

    schema = extract_schema(body)
    requested_model = body.get("model")

    # --- schema implicito ------------------------------------------------
    # Em varios turnos o jogo NAO manda response_format, mas exige o JSON de
    # 5 campos de volta (foi o que gerou 22 pedidos de "repair" no Player.log).
    # Detectamos pelo system prompt + pelo formato das mensagens do jogo.
    if schema is None and FORCE_SCHEMA:
        texto_sys = " ".join(flatten_content(m.get("content")) for m in messages
                             if (m.get("role") or "") == "system").lower()
        texto_user = " ".join(flatten_content(m.get("content")) for m in messages
                              if (m.get("role") or "") == "user")

        # Pistas (qualquer uma basta):
        #  - system cita os campos do contrato
        #  - mensagens do jogo sao arrays JSON com chaves dele
        pistas = (
            "trust_percent" in texto_sys
            or ("dialogue" in texto_sys and "emotion" in texto_sys)
            or '"player_dialogue"' in texto_user
            or '"initial_state"' in texto_user
            or '"repair"' in texto_user
        )
        if pistas:
            schema = dict(TURN_SCHEMA_IMPLICITO)
            log("[i] schema implicito ativado (o jogo nao mandou response_format)")

    # O jogo manda o modelo que o host escolheu (ex.: qwen2.5:7b).
    # Se for o nosso modelo (ou o fake), usamos o configurado.
    model = OLLAMA_MODEL

    # --- parametros -----------------------------------------------------
    try:
        temperature = float(body.get("temperature", TEMPERATURE) or TEMPERATURE)
    except (TypeError, ValueError):
        temperature = TEMPERATURE
    try:
        top_p = float(body.get("top_p", TOP_P) or TOP_P)
    except (TypeError, ValueError):
        top_p = TOP_P
    try:
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or NUM_PREDICT)
    except (TypeError, ValueError):
        max_tokens = NUM_PREDICT
    if max_tokens <= 0:
        max_tokens = NUM_PREDICT

    payload: dict[str, Any] = {
        "model": model,
        "messages": prepare_messages(messages),
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": max(0.0, min(temperature, 2.0)),
            "top_p": max(0.01, min(top_p, 1.0)),
            # Com schema o modelo devolve um JSON de 5 campos; ~200 tokens
            # bastam. Antes forcavamos 512+, o que so aumentava a latencia
            # (o jogo tocava o filler "Hum, certo..." enquanto esperava).
            "num_predict": min(max_tokens, 220) if schema is not None else min(max_tokens, 120),
            "num_ctx": NUM_CTX,
            "repeat_penalty": REPEAT_PENALTY,
            "repeat_last_n": REPEAT_LAST_N,
        },
    }

    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if isinstance(stop, list) and stop:
        payload["options"]["stop"] = [str(s) for s in stop if s]

    if schema is not None:
        payload["format"] = clean_schema(schema)
    elif body.get("response_format", {}).get("type") == "json_object" if isinstance(body.get("response_format"), dict) else False:
        payload["format"] = "json"

    hint = trust_hint(messages, schema)
    if hint:
        payload["messages"].append({"role": "system", "content": hint})

    log(f"[>] model={requested_model} msgs={len(messages)} schema={'sim' if schema else 'nao'} "
        f"temp={temperature:.2f} max={max_tokens}")

    # --- chamada ao Ollama (com 1 retentativa curta) ---------------------
    data = None
    last_err = ""
    for tent in (1, 2):
        try:
            data = await call_ollama(payload)
            break
        except Exception as e:
            last_err = str(e)
            log(f"[x] tentativa {tent}: {last_err}")
            if tent == 1:
                await asyncio.sleep(0.6)

    if data is None:
        return error_json(f"Modelo local nao respondeu: {last_err}", "upstream_error", 502)

    content = (data.get("message") or {}).get("content") or ""

    if data.get("done_reason") == "length":
        log("[!] saida estourou o limite; refazendo com mais tokens")
        try:
            payload["options"]["num_predict"] = NUM_PREDICT * 2
            data = await call_ollama(payload)
            content = (data.get("message") or {}).get("content") or content
        except Exception as e:
            log(f"[x] retry length: {e}")

    # --- garante o contrato exato do jogo -------------------------------
    if schema is not None:
        obj = extract_json(content)
        if not isinstance(obj, dict):
            log("[!] resposta sem JSON; tentando recuperar texto")
            # o modelo pode ter respondido texto livre (ex.: "M Cara, eu...")
            obj = {"dialogue": "" if content.lstrip().startswith("{") else content}
        obj = conform(obj, schema)

        # limpa artefatos do texto
        d = strip_artifacts(str(obj.get("dialogue") or ""))
        d = fix_encoding(d)

        # evita repetir fala anterior
        d = dedupe_dialogue(d, messages)
        d = encurtar(d)
        RECENTES.append(d)
        del RECENTES[:-8]

        if not d.strip():
            d = random.choice(PT_FALLBACKS)
            log("[!] dialogue vazio do modelo -> fala de reserva pt-BR")

        obj["dialogue"] = d
        content = json.dumps(obj, ensure_ascii=JSON_ESCAPE)
    else:
        # Nesse caso o jogo NAO pediu schema (chamada de traducao/abertura).
        # Temos que devolver texto limpo no lugar de JSON.
        content = fix_encoding(strip_artifacts(content))

    eval_count = data.get("eval_count") or 0
    eval_dur = (data.get("eval_duration") or 1) / 1e9
    elapsed = time.time() - started
    log(f"[<] {elapsed:5.2f}s  {eval_count} tok  {eval_count / eval_dur:5.1f} tok/s")
    background.add_task(dump_log, messages, content)

    return {
        "id": f"chatcmpl-local-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model or model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count") or 0,
            "completion_tokens": eval_count,
            "total_tokens": (data.get("prompt_eval_count") or 0) + eval_count,
        },
    }


def dump_log(messages: list, content: str) -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        (LOG_DIR / "last_turn.json").write_text(
            json.dumps({"messages": messages, "response": content}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    log("=" * 74)
    log(" LOCAL-AI v2  |  Jogo -> 127.0.0.1:%d -> Ollama" % PORT)
    log(f"   modelo  : {OLLAMA_MODEL}")
    log(f"   ollama  : {OLLAMA_URL}")
    log(f"   contexto: {NUM_CTX} tokens | max saida {NUM_PREDICT} | keep_alive {KEEP_ALIVE}")
    log(f"   idioma  : {'portugues do Brasil (forcado)' if FORCE_PT else 'livre'}")
    log("=" * 74)
    try:
        r = await _client.get(f"{OLLAMA_URL}/api/tags", timeout=6.0)
        names = [m.get("name") for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in names and f"{OLLAMA_MODEL}:latest" not in names:
            log(f"[!] ATENCAO: '{OLLAMA_MODEL}' nao esta instalado.")
            log(f"    rode: ollama pull {OLLAMA_MODEL}")
            log(f"    instalados: {', '.join(names) or '(nenhum)'}")
        else:
            log(f"[ok] Ollama pronto; '{OLLAMA_MODEL}' disponivel.")
        # pre-carrega na GPU para a primeira fala nao ter atraso
        try:
            await _client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": "oi"}],
                    "stream": False,
                    "keep_alive": KEEP_ALIVE,
                    "options": {"num_predict": 2},
                },
                timeout=180.0,
            )
            log("[ok] modelo pre-carregado na GPU.")
        except Exception as e:
            log(f"[!] nao consegui pre-carregar: {e}")
    except Exception as e:
        log(f"[!] Ollama NAO respondeu em {OLLAMA_URL}: {e}")
        log("    inicie com: ollama serve")


@app.on_event("shutdown")
async def shutdown() -> None:
    await _client.aclose()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

