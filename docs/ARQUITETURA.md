# Arquitetura

```
┌─────────────┐   HTTP/JSON    ┌──────────────────┐   HTTP    ┌────────┐
│   Cliente   │ ─────────────> │  local-ai-bridge │ ────────> │ Ollama │
│ (SDK OpenAI)│ <───────────── │   (FastAPI)      │ <──────── │  (GPU) │
└─────────────┘                └──────────────────┘           └────────┘
                                  │
                                  ├─ schema implícito
                                  ├─ anti-repetição
                                  ├─ reparo de encoding
                                  └─ conformidade
```

O cliente acha que fala com a API do OpenAI. A ponte traduz, corrige e
encaminha para o Ollama.

---

## Fluxo de uma requisição

```
POST /v1/chat/completions
  │
  ├─ 1. Valida o corpo (JSON, messages obrigatório)
  │
  ├─ 2. Extrai o schema
  │      ├─ de response_format (se o cliente mandar)
  │      └─ INFERE do system prompt (schema implícito)
  │
  ├─ 3. Prepara as mensagens
  │      ├─ repara mojibake na entrada
  │      ├─ injeta nudge de idioma
  │      └─ injeta [ESTADO] (anti-repetição + contexto)
  │
  ├─ 4. Chama o Ollama com format=<schema>
  │
  ├─ 5. Pós-processa a resposta
  │      ├─ extrai JSON (ou recupera parcial)
  │      ├─ conforma ao schema (coage tipos, defaults)
  │      ├─ remove artefatos (prefixos, marcadores)
  │      ├─ deduplica falas (Jaccard)
  │      └─ serializa com ensure_ascii
  │
  └─ 6. Devolve no envelope OpenAI
```

---

## Módulos

### Extração de schema

```python
def extract_schema(body) -> dict | None:
    """Procura schema em response_format, em diferentes formatos."""
```

Suporta as três variações que aparecem na prática:

```json
{ "type": "json_schema", "json_schema": { "schema": {...} } }
{ "type": "json_schema", "json_schema": {...} }
{ "type": "json_object" }
```

E limpa campos que o Ollama rejeita (`$schema`, `additionalProperties`,
`title`, `default`, `format`):

```python
def clean_schema(node):
    REMOVER = {"$schema", "additionalProperties", "title", "default", "format"}
```

### Schema implícito

Modelos pequenos ignoram `response_format` com frequência. Em vez de aceitar
texto livre, a ponte deduz o contrato:

```python
if schema is None and FORCE_SCHEMA:
    texto_sys = " ".join(...)
    pistas = (
        "trust_percent" in texto_sys
        or '"player_dialogue"' in texto_user
        or '"initial_state"' in texto_user
    )
    if pistas:
        schema = TURN_SCHEMA_IMPLICITO
```

### Conformidade

Depois que o modelo responde, o contrato é forçado:

```python
def conform(obj, schema):
    for key, sub in props.items():
        if key in obj:
            out[key] = _coerce(obj[key], sub)   # "42" -> 42, "true" -> True
        elif key in required:
            out[key] = _default_for(sub)         # chave ausente -> default
```

Isso transforma respostas quase-certas em respostas válidas, evitando o ciclo
de "repair" que custa uma requisição inteira a mais.

### Recuperação de JSON truncado

Se o modelo bater no limite de tokens no meio do JSON:

```python
def _recover_partial(text):
    m = re.search(r'"dialogue"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.S)
    ...
```

Salva o campo mais importante em vez de devolver erro.

### Anti-repetição

**Camada 1 — prompt:** lista as últimas falas e proíbe repetir as aberturas.

**Camada 2 — código:** similaridade de Jaccard:

```python
def dedupe_dialogue(text, messages):
    for ant in _spoken_lines(messages):
        wa, wb = set(norm(text).split()), set(norm(ant).split())
        jac = len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0
        if jac >= 0.85:
            return random.choice(REACOES_VARIADAS)   # varia
```

**Por que duas camadas:** o prompt reduz a frequência, mas não garante. O
código garante.

### Reparo de encoding

```python
def fix_encoding(text):
    if "\u00c3" not in text and "\u00c2" not in text:
        return text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        if fixed.count("\u00c3") < text.count("\u00c3"):
            return fixed          # só aceita se melhorou
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text
```

O teste `count() < count()` evita "consertar" texto que já estava correto.

> Ver `docs/LICOES.md` §1: nem todo símbolo estranho no log é corrupção.

### Latência

| Técnica | Implementação |
|---|---|
| Preload | `POST /api/generate` com `num_predict: 2` no startup |
| `keep_alive` | `"30m"` mantém o modelo na GPU entre requisições |
| Limite de saída | `num_predict: 220` (suficiente para 1–2 frases) |
| Retry curto | 2 tentativas, não 5 |

O que **não** ajuda (medido): reduzir `num_ctx`. A latência é dominada pela
geração de tokens, não pelo tamanho da janela.

---

## Decisões de projeto

### Por que `ensure_ascii=True`?

Serializar o JSON com escapes ASCII (`\u00f4` em vez de `ô`) custa alguns bytes
e torna a resposta imune a camadas intermediárias que interpretem o corpo como
Latin-1/CP1252. É defensivo, não obrigatório.

### Por que 2 camadas de anti-repetição?

Prompt sozinho falha em ~1 de cada 5 turnos com modelo de 7B. Código sozinho
funciona, mas produz frases genéricas ("não entendi"). Juntos: o prompt gera a
resposta boa na maioria das vezes; o código garante o mínimo.

### Por que detectar schema em vez de exigir `response_format`?

Porque o cliente real não manda. A ponte existe para atender o cliente, não o
contrário.

### Por que `conform` e não só validar?

Porque devolver erro dispara uma nova requisição ("repair") que custa o dobro.
Preencher um default é mais barato que pedir de novo.

---

## Limitações conhecidas

- **Streaming:** não implementado. A resposta é sempre completa. Clientes que
  exigem `stream=true` recebem `stream=false` silenciosamente.
- **Function calling:** não suportado. `tools` é ignorado.
- **Visão:** não suportado. Só texto.
- **Múltiplos modelos simultâneos:** `keep_alive` mantém um modelo por vez na
  GPU; alternar modelos custa recarga (~9 s).
- **Sem autenticação:** destina-se a `127.0.0.1`. Não exponha na rede.
