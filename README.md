# local-ai-bridge

**Uma ponte local compatível com a API do OpenAI/OpenRouter para rodar LLMs na
sua própria GPU — feita como material de estudo sobre implementação de
proxies de API, structured output e loweração de latência.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)

---

## O que é

Um servidor FastAPI que expõe endpoints **compatíveis com a API do OpenAI** e
encaminha as requisições para o [Ollama](https://ollama.com/) local. Ele existe
para resolver problemas concretos que aparecem quando você troca uma API de
nuvem por um modelo local de 7B parâmetros:

| Problema | Como este projeto resolve |
|---|---|
| O cliente exige JSON com schema rígido, mas o modelo devolve texto livre | **Schema implícito** — infere o contrato a partir do system prompt |
| O modelo devolve JSON truncado ou com chaves faltando | **Conformidade** — preenche defaults, coage tipos, recupera resposta parcial |
| O modelo entra em loop repetindo a mesma frase | **Anti-repetição** em duas camadas (prompt + código com similaridade Jaccard) |
| Texto com mojibake ao atravessar camadas Latin-1 | **Reparo de encoding** na entrada e na saída |
| O modelo responde em inglês quando o app pede português | **Nudge de idioma** injetado no system prompt |
| Resposta lenta faz a UI desistir e trocar por fallback | **Otimização de latência** (ver abaixo) |

Tudo isso é genérico: serve para qualquer cliente que fale a API do OpenAI.

---

## ⚠️ Escopo e uso pretendido

Este projeto é **material educacional**, publicado **apenas como estudo
técnico** de:

- como implementar um proxy compatível com a API do OpenAI
- como forçar *structured output* com modelos pequenos locais
- como medir e reduzir latência em inferência local
- como diagnosticar problemas de encoding entre camadas

**O QUE ESTE REPOSITÓRIO NÃO INCLUI, DE PROPÓSITO:**

- ❌ Nenhum binário de jogo (`Assembly-CSharp.dll` ou similar)
- ❌ Nenhuma ferramenta de patch/pirataria
- ❌ Nenhum código para contornar DRM, licenciamento ou autenticação
- ❌ Nenhum asset, código-fonte descompilado ou material de terceiros

O autor **não apoia nem facilita** modificação de software de terceiros,
violação de EULA, ou uso em partidas públicas/multijogador.

O servidor é uma implementação **independente** e genérica. Se você usar com
qualquer aplicação, faça isso **na sua própria cópia, offline, com software que
você tem direito de modificar**. A responsabilidade pelo uso é inteiramente sua.

---

## Instalação

```bash
git clone https://github.com/danielfinotti2-spec/local-ai-bridge.git
cd local-ai-bridge

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Instale o [Ollama](https://ollama.com/download) e baixe um modelo:

```bash
ollama pull qwen2.5:7b
```

Rode o servidor:

```bash
python src/server.py
```

Ele sobe em `http://127.0.0.1:8000`.

---

## Uso

O servidor fala o dialeto do OpenAI. Um exemplo mínimo:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [
      {"role": "user", "content": "Explique o que é structured output em 1 frase."}
    ]
  }'
```

É compatível com o SDK oficial:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="qualquer-coisa")

resp = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Oi!"}],
)
print(resp.choices[0].message.content)
```

### Rotas

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Identificação do serviço |
| `GET` | `/health` | Status do Ollama e do modelo |
| `GET` | `/v1/models` | Lista de modelos (formato OpenRouter completo) |
| `POST` | `/v1/chat/completions` | Chat completion |
| `GET` | `/api/health` | Alias |
| `POST` | `/api/v1/chat/completions` | Alias |

---

## Os quatro problemas técnicos

### 1. Structured output com modelo pequeno

Modelos de 7B frequentemente ignoram `response_format` e devolvem texto livre.
O projeto tem três camadas de defesa:

**a) Schema implícito** — quando o cliente não manda `response_format` mas o
system prompt descreve campos, o servidor deduz o schema:

```python
pistas = (
    "trust_percent" in texto_sys
    or '"player_dialogue"' in texto_user
    or '"initial_state"' in texto_user
)
if pistas and schema is None:
    schema = TURN_SCHEMA_IMPLICITO
```

Isso eliminou **22 chamadas de "repair"** num único log de teste — cada repair
custava uma requisição inteira a mais.

**b) Conformidade** — depois da resposta, o servidor força o contrato:

```python
for key, sub in props.items():
    if key in obj:
        out[key] = _coerce(obj[key], sub)      # coage tipo
    elif key in required:
        out[key] = _default_for(sub)           # preenche default
```

**c) Recuperação de JSON truncado** — se o modelo bater no limite de tokens,
um regex salva o campo mais importante:

```python
m = re.search(r'"dialogue"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.S)
```

### 2. Anti-repetição

Duas camadas, porque prompt sozinho não basta:

**Prompt:** lista as últimas falas e as 3 primeiras palavras de cada uma:

```
[FALAS JA DITAS - NAO REPITA] "Oi, tudo bem?" | "Poxa, que isso"
REGRAS:
2. NAO COMECE com estas aberturas ja usadas: "Oi, tudo bem" / "Poxa, que isso"
```

**Código:** similaridade de Jaccard sobre as palavras:

```python
wa, wb = set(norm.split()), set(a.split())
jac = len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0
if jac >= 0.85:
    return random.choice(REACOES_VARIADAS)
```

> **Lição aprendida:** a primeira versão da instrução era tão rígida
> ("não repita nada") que o modelo passou a **negar fatos que sabia** — quando
> perguntei o nome de um cachorro citado 4 turnos antes, ele respondeu "nem
> lembro mais". Corrigi adicionando explicitamente *"você LEMBRA de tudo; só
> não repita a frase"*. Memória voltou. Instruções de "não repita" precisam ser
> pareadas com uma afirmação de memória.

### 3. Encoding entre camadas

Um achado que me custou horas: eu diagnostiquei um problema de "voz cortada"
como sendo encoding, e estava **errado**.

O log do cliente mostrava `␠` (U+2420) entre todas as palavras. Parecia
corrupção. Mas a contagem no arquivo real:

| Leitura do log | Ocorrências de U+2420 |
|---|---|
| como UTF-8 | **452** |
| como CP1252 | **0** |

O cliente **grava espaço como `␠` de propósito** — é o formato do log dele.
Não havia corrupção nenhuma.

O código de reparo continua no projeto porque mojibake de verdade existe:

```python
def fix_encoding(text: str) -> str:
    """Corrige UTF-8 lido como Latin-1: 'indicaÃ§Ãµes' -> 'indicações'."""
    if "\u00c3" not in text and "\u00c2" not in text:
        return text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        if fixed.count("\u00c3") < text.count("\u00c3"):
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text
```

**A lição real:** antes de acusar encoding, **conte as ocorrências** com cada
codec. Um símbolo estranho no log pode ser intencional.

### 4. Latência

Medição com o modelo já carregado (descartando custo de recarga):

| `num_ctx` | Latência média | Máximo |
|---|---|---|
| 6144 | 0,97 s | 1,03 s |
| 8192 | 0,97 s | 1,00 s |

**Idênticas.** Reduzir o contexto **não** acelera a geração — o gargalo é gerar
tokens, não o tamanho da janela.

O que realmente funcionou:

| Mudança | Efeito |
|---|---|
| `num_predict` de 512 → 220 | ~40% menos tempo |
| `keep_alive=30m` + preload no startup | elimina recarga de ~9 s |
| Modelo 100% na GPU | 13,95 s → 2,09 s |

### Sobre escolher modelo pela VRAM

Antes de adotar um modelo, confira se ele **cabe**:

```bash
ollama ps
```

```
NAME         SIZE     PROCESSOR
qwen2.5:7b   4.8 GB   100% GPU        <- ok
gemma2:9b    7.2 GB   24%/76% CPU/GPU <- vazou para a CPU
```

O `gemma2:9b` ocupa 7,2 GB de 8 GB de VRAM e roda **24% na CPU** — **7× mais
lento** (13,95 s vs 2,09 s). Regra prática: o modelo precisa caber com folga
para o contexto KV. Em 8 GB, fique em **≤ 5 GB**.

O script `tests/comparar_modelos.py` automatiza essa verificação.

---

## Testes

```bash
python tests/test_rotas.py          # contrato da API
python tests/test_encoding.py       # reparo de encoding
python tests/test_robustez.py       # JSON malformado, campos faltando
python tests/comparar_modelos.py    # compara modelos + alerta de VRAM
```

Saída do teste de rotas:

```
11/11 rotas OK
```

---

## Estrutura

```
local-ai-bridge/
├── src/
│   ├── server.py          # a ponte (FastAPI)
│   ├── config.json        # configuração
│   └── installer.py       # utilitário de configuração
├── tests/
│   ├── test_rotas.py
│   ├── test_encoding.py
│   ├── test_robustez.py
│   └── comparar_modelos.py
├── docs/
│   ├── ARQUITETURA.md
│   └── LICOES.md          # erros de diagnóstico que cometi
├── examples/
│   └── cliente_openai.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Configuração

`src/config.json`:

```json
{
  "model": "qwen2.5:7b",
  "port": 8000,
  "num_ctx": 6144,
  "num_predict": 220,
  "temperature": 0.6,
  "top_p": 0.95,
  "repeat_penalty": 1.08,
  "repeat_last_n": 256,
  "keep_alive": "30m",
  "force_pt": 1,
  "force_schema": 1
}
```

Qualquer valor pode ser sobrescrito por variável de ambiente:

```bash
LOCALAI_MODEL=qwen2.5:3b LOCALAI_NUM_CTX=4096 python src/server.py
```

---

## Contribuindo

Issues e PRs sobre os problemas **técnicos** (structured output, latência,
encoding, compatibilidade com a API do OpenAI) são bem-vindos.

Não são aceitos PRs que adicionem: engenharia reversa de software de terceiros,
ferramentas de patch, bypass de licenciamento, ou assets proprietários.

---

## Licença

MIT — veja [LICENSE](LICENSE).

---

## Agradecimentos

[FastAPI](https://fastapi.tiangolo.com/) ·
[Ollama](https://ollama.com/) ·
[httpx](https://www.python-httpx.org/) ·
[qwen2.5](https://github.com/QwenLM/Qwen2.5)
