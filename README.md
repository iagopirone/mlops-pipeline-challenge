# Desafio MLOps — Starter Kit

Kit inicial para a atividade de **construir um pipeline de MLOps** em torno de um
modelo de detecção de objetos. Aqui está tudo que você precisa para **começar**: o
enunciado, um modelo já treinado, os três scripts base e os dados.

> **Leia o enunciado primeiro:** [`docs/ATIVIDADE.pdf`](docs/ATIVIDADE.pdf)
> (versão Markdown: [`ATIVIDADE.md`](ATIVIDADE.md)).
>
> Diagrama do ciclo a implementar: [`docs/mlops_cycle.png`](docs/mlops_cycle.png).

---

## O que você recebe

| Artefato | Onde | O que é |
|---|---|---|
| Enunciado | `docs/ATIVIDADE.pdf` · `ATIVIDADE.md` | a especificação completa do desafio |
| Diagrama | `docs/mlops_cycle.png` | o ciclo de MLOps modelado |
| Modelo inicial `v0` | `models/v0/best.pt` · `best.onnx` | detector já treinado no seed |
| Scripts base | `src/prep_data.py` · `train.py` · `infer.py` | os "scripts soltos" a refatorar |
| Dados | `data/` | fonte raw (seed) + stream + oracle (ver abaixo) |

## Os dados (BCCD)

Detecção de células sanguíneas — 3 classes: **RBC**, **WBC**, **Platelets**.
O dataset foi propositalmente dividido para **simular o crescimento por coleta**:

```
data/
  raw/        # SEED — dado inicial rotulado (imagens + labels YOLO). É o "pouco que você já tem".
    images/   #   91 imagens .jpg
    labels/   #   91 .txt YOLO (class cx cy w h, normalizado)
  stream/     # STREAM — "imagens novas" que chegam ao serviço de inferência.
    images/   #   273 imagens .jpg, SEM label visível
  oracle/     # ORACLE — os labels reais do stream, ESCONDIDOS.
    labels/   #   273 .txt — revelados só na anotação simulada (substitui o humano)
  classes.txt # RBC / WBC / Platelets (ordem = id 0,1,2)
```

**Como usar no loop:** o serviço processa imagens do `stream/`; os casos
selecionados pela coleta recebem o label correspondente vindo do `oracle/`
(anotação simulada), que é **escrito na fonte raw** — fazendo o dataset crescer e
disparando um novo ciclo. Você **não** implementa ferramenta de rotulagem.

## Baseline (rode antes de começar)

O projeto usa [**uv**](https://docs.astral.sh/uv/).

```bash
uv sync                 # ambiente de inferência (leve)
uv sync --group train   # + ultralytics/torch (só p/ re-treinar)

# 1. prepara o dataset a partir da fonte raw (train/val/test + data.yaml)
uv run python src/prep_data.py

# 2. (opcional) re-treina a partir do v0 — o v0 já vem pronto em models/v0/
uv run python src/train.py --base models/v0/best.pt --epochs 20

# 3. inferência em uma imagem do stream
uv run python src/infer.py --model models/v0/best.onnx \
  --image data/stream/images/$(ls data/stream/images | head -1)
```

Se os três rodam, seu ambiente está ok. O modelo `v0` foi treinado só no seed —
ele é fraco de propósito: **melhorá-lo via o loop de coleta é a sua tarefa.**

## Sua tarefa

Transformar esses três scripts em um **pipeline de MLOps orientado a mensagens**
(produtor/consumidor sobre RabbitMQ) que feche o ciclo *dados → treino →
inferência → coleta → mais dados*. Os requisitos, o MVP, a Fase 2 (API), os
contratos das mensagens e a forma de entrega estão no **enunciado**.

## Estrutura

```
docs/        enunciado (PDF) + diagramas
src/         prep_data.py · train.py · infer.py  (scripts base)
models/v0/   checkpoint inicial (best.pt + best.onnx)
data/        raw/ (seed) · stream/ · oracle/ · classes.txt
pyproject.toml · uv.lock
```

## Créditos / licença dos dados

Dados: **BCCD Dataset** (https://github.com/Shenggan/BCCD_Dataset), licença MIT.
Convertido de Pascal VOC para o formato YOLO neste kit.
