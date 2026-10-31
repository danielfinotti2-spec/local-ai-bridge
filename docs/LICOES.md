# Lições de diagnóstico

Erros que cometi ao construir este projeto, registrados para não se repetirem.
Documentar o erro é mais útil que apagá-lo.

---

## 1. O bug que eu inventei

**O sintoma:** um cliente reclamava que o áudio gerado pela aplicação saía
cortado.

**Meu diagnóstico:** examinei o log do cliente e vi isso:

```
text="Desculpe,â oâ queâ disse?"
```

Todos os espaços apareciam como `â`. Em hex, os bytes eram `E2 90 A0` — que
em UTF-8 é o caractere **U+2420 `␠` (símbolo de espaço visível)**.

Reproduzi em laboratório:

```python
'Desculpe,\u2420o\u2420que'.encode('utf-8').decode('latin-1')
# -> 'Desculpe,â\x90\xa0oâ\x90\xa0que'   igual ao log, byte a byte
```

Conclui: o cliente está lendo o JSON como Latin-1 e corrompendo o texto.
"Mudei para `ensure_ascii=True`" e anunciei como corrigido.

**Estava errado.**

**O teste que deveria ter feito primeiro:**

```powershell
$txt = [System.Text.Encoding]::UTF8.GetString($bytes)
[regex]::Matches($txt, [char]0x2420).Count     # -> 452
```

Achei 452 ocorrências lendo como UTF-8 e **zero** lendo como CP1252.

O cliente **grava espaço como `␠` de propósito** — é o formato do log dele, para
tornar espaços visíveis em trace de depuração. Não havia corrupção nenhuma.

O texto sempre chegou perfeito ao sintetizador:

```
"Poxa, quem é tomando atitude?"
"Eita, pode botar para funcionar!"
"Fã do Batman? Pode ser, mas não sou eu."
```

**Lição:** antes de acusar encoding, **conte as ocorrências com cada codec**.
Compare as contagens. Se o código de erro não existe em nenhuma decodificação,
o problema é outro — provavelmente é o **formato do log**, não os dados.

**A causa real** era latência: a resposta demorava 14,5 s, a interface desistia
e tocava um áudio de espera, e quando a resposta chegava, o áudio era cancelado.
Nada a ver com bytes.

---

## 2. Quando "não repita" faz o modelo esquecer

**O sintoma:** o modelo repetia a mesma frase de abertura em turnos
consecutivos.

**A correção que fiz:** instrução no prompt:

```
REGRAS CRÍTICAS:
1. Não repita nenhuma fala anterior.
```

**O que aconteceu:** a repetição parou — e o modelo passou a **negar fatos que
conhecia**. Num teste, o usuário tinha citado um cachorro chamado "Bolota" no
turno 2; no turno 5, perguntado sobre o nome, o modelo respondeu *"nem lembro
mais do nome do meu cachorro"*.

O histórico completo estava lá. A instrução "não repita" foi interpretada como
"não traga informação do passado".

**A correção:**

```
1. Você LEMBRA de tudo que já foi dito. Fatos, nomes e números citados
   continuam valendo — use-os normalmente.
2. Não repita uma FRASE já dita. Responda ao que ele disse AGORA com
   uma fala nova (mas pode se referir ao que já foi conversado).
```

Memória voltou: *"Bolota e moro na Rua das Acacias, 47"*.

**Lição:** toda instrução negativa precisa de um par positivo explícito. "Não
repeita" precisa de "mas lembra". Modelos pequenos não inferem a distinção
entre repetir conteúdo e referenciar conteúdo.

---

## 3. Parâmetro ajustado sem medir

**O que eu tinha visto:** o modelo parecia "burro" — respostas sem graça, estado
emocional travado.

**O que eu presumi:** a penalidade de repetição estava alta demais e a
temperatura também.

**O que a medição mostrou:** rodei 4 presets na mesma conversa:

| temp | penalty | Repetiu | Trust variou | Latência |
|---|---|---|---|---|
| 0.60 | 1.08 | 0 | 8 valores | 1.24 s |
| 0.85 | 1.20 | 0 | 6 valores | 1.36 s |
| 0.75 | 1.12 | 0 | 7 valores | 1.22 s |
| 0.50 | 1.02 | 0 | 7 valores | 1.40 s |

**Repetição já era zero em todos.** O anti-repetição no código (Jaccard) já
resolvia. O ajuste fino de penalidade não era a causa.

E a temperatura? Achei que 0.6 estava travando o campo `emotion` em `NEUTRAL`.
Testei isoladamente com um roteiro que incluía insulto:

```
temp=0.6  : ['NEUTRAL','NEUTRAL','SUSPICIOUS','ANGRY','TRUSTING',...]
temp=0.85 : ['NEUTRAL','TRUSTING','SUSPICIOUS','ANGRY','TRUSTING',...]
```

**Ambos variam corretamente.** O `NEUTRAL` travado era o **roteiro de teste**
(falas neutras), não o parâmetro.

**Lição:** "o modelo está burro" é hipótese, não diagnóstico. Isole uma
variável, meça as duas condições com o mesmo input, e só então conclua. E
verifique se o seu **teste** não é a causa — um roteiro só com falas neutras
vai produzir emoções neutras em qualquer temperatura.

---

## 4. Confundir correlação de tempo com causa

**O que eu medi:** `num_ctx` 8192 → 2,83 s de média; `num_ctx` 6144 → 1,16 s.
Conclui que reduzir o contexto acelerava.

**O erro:** a primeira chamada de cada configuração incluía o custo de
**recarregar o modelo** na GPU (~9 s). Comparar médias incluindo esse valor
contamina o resultado.

**Medindo com o modelo quente:**

| num_ctx | 1ª chamada | Média (quente) |
|---|---|---|
| 6144 | 9,51 s | 0,97 s |
| 8192 | 9,56 s | 0,97 s |

**Idênticas.** O tamanho do contexto não afeta a velocidade de geração.

**Lição:** em inferência local, descarte sempre a primeira chamada de uma nova
configuração. O tempo de carga do modelo é o maior confundidor que existe.

---

## 5. Trocar de modelo sem checar se cabe

**A sugestão recebida:** "o modelo X não é bom em português, use um 9B".

**O que fiz:** baixei e testei.

```
NAME         SIZE     PROCESSOR
gemma2:9b    7.2 GB   24%/76% CPU/GPU
```

O modelo tinha 5,4 GB em disco mas ocupava **7,2 GB de VRAM** com o contexto —
não coube nos 8 GB da placa e vazou para a CPU.

| Modelo | Processador | Latência | Máximo |
|---|---|---|---|
| qwen2.5:7b | 100% GPU | 2,09 s | 8,87 s |
| gemma2:9b | 24% CPU | **13,95 s** | **84,57 s** |

**7× mais lento.** E como o problema que eu estava tentando resolver era
justamente *latência*, a "melhoria" pioraria o sintoma.

**Lição:** antes de avaliar qualidade de um modelo, verifique se ele está
**100% na GPU**:

```bash
ollama ps
```

Se aparecer `CPU/GPU` misto, qualquer comparação de qualidade é inválida — você
está medindo velocidade de CPU, não capacidade do modelo. Em 8 GB de VRAM,
mantenha o modelo em **≤ 5 GB**.

---

## Resumo

| # | Erro | Lição |
|---|---|---|
| 1 | Achar que era encoding | Conte ocorrências com cada codec antes de acusar |
| 2 | "Não repita" sem "mas lembra" | Instrução negativa precisa de par positivo |
| 3 | Ajustar parâmetro sem medir | Isole uma variável; suspeite do próprio teste |
| 4 | Incluir carga do modelo na média | Descarte a primeira chamada |
| 5 | Trocar modelo sem checar VRAM | `ollama ps` antes de comparar qualidade |
