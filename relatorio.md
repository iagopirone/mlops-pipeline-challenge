# Relatório — Pipeline de MLOps para Modelo de Detecção

## 1. Visão geral do problema

A atividade consiste em transformar três scripts soltos de preparação de dados, treinamento e inferência em um pipeline de MLOps automatizado, orientado a mensagens.

O objetivo principal não é treinar o melhor modelo possível de uma vez, mas sim construir um ciclo de vida de modelo que consiga rodar de ponta a ponta:

```text
dados → dataset versionado → treino → gate de qualidade → modelo promovido → inferência → coleta → novos dados → novo ciclo
```

A solução é construída como um MVP, priorizando o ciclo completo funcionando, com rastreabilidade básica de datasets, modelos, runs de treino e inferências.

---

## 2. Entendimento inicial dos scripts base

Antes de implementar os workers, validei os scripts fornecidos manualmente para entender suas entradas, saídas e responsabilidades.

### 2.1 Script de dados — `src/prep_data.py`

Responsabilidade atual:

* lê a fonte raw em `data/raw`;
* particiona os dados em treino, validação e teste;
* cria a estrutura de dataset no formato YOLO;
* gera o arquivo `data.yaml`.

Entrada principal:

```text
data/raw
data/classes.txt
```

Saída principal:

```text
dataset/
dataset/data.yaml
dataset/images/train
dataset/images/val
dataset/images/test
dataset/labels/train
dataset/labels/val
dataset/labels/test
```

Como esse script será usado no pipeline:

* ele será a base do Worker de Dados;
* o worker consumirá mensagens da fila `q.data.build`;
* criará uma versão de dataset, como `ds-001`;
* salvará metadados em JSON;
* publicará uma mensagem na fila `q.train.run`.

---

### 2.2 Script de treino — `src/train.py`

Responsabilidade atual:

* recebe um dataset preparado;
* recebe um checkpoint base;
* faz fine-tune do modelo;
* avalia no conjunto de teste;
* exporta o modelo para ONNX.

Entrada principal:

```text
dataset/data.yaml
models/v0/best.pt
```

Saída observada no teste inicial:

```text
runs/smoke_test/weights/best.pt
runs/smoke_test/weights/best.onnx
runs/smoke_test/weights/last.pt
```

Como esse script será usado no pipeline:

* ele será a base do Worker de Treino;
* o worker consumirá mensagens da fila `q.train.run`;
* treinará um modelo candidato;
* aplicará o gate de qualidade;
* registrará modelos promovidos e não promovidos;
* publicará `q.model.promoted` apenas se o modelo passar no gate.

---

### 2.3 Script de inferência — `src/infer.py`

Responsabilidade atual:

* carrega um modelo ONNX;
* recebe uma imagem;
* roda inferência;
* retorna detecções com classe, confiança e bounding box.

Entrada principal:

```text
models/v0/best.onnx
data/stream/images/<imagem>.jpg
```

Saída observada no teste inicial:

```text
RBC conf=0.88 box=[157, 76, 244, 171]
WBC conf=0.87 box=[256, 183, 507, 373]
RBC conf=0.83 box=[78, 330, 173, 433]
...
```

Como esse script será usado no pipeline:

* ele será a base do Worker de Inferência;
* o worker consumirá mensagens da fila `q.infer.request`;
* gerará um `inference_id`;
* salvará o resultado por `inference_id`;
* publicará `q.infer.result`;
* o mesmo evento será usado como resultado da inferência e como entrada para o Worker de Coleta.

---

## 3. Decisões iniciais de arquitetura

### 3.1 Versionamento local com JSON

Decisão:

Usar pastas locais e arquivos JSON para guardar versões de datasets, modelos, execuções de treino e inferências.

Justificativa:

Essa abordagem mantém a solução simples e fácil de inspecionar. Como o foco inicial é fazer o ciclo completo funcionar, preferi começar com uma estrutura local antes de adicionar ferramentas ou serviços mais complexos.

Trade-off:

* Vantagem: simples, fácil de entender, fácil de testar e suficiente para validar o pipeline.
* Desvantagem: em um cenário maior, essa organização poderia precisar ser substituída por uma solução mais robusta.

---

### 3.2 Modelo reprovado no gate não é erro técnico

Decisão:

Quando um modelo candidato não superar o baseline, ele será registrado como uma run válida, mas com status de não promovido.

Justificativa:

Um treino que não passa no gate é um resultado esperado do pipeline, não uma falha técnica. Ele deve ser salvo para manter histórico e rastreabilidade.

Trade-off:

* Vantagem: preserva histórico completo das tentativas de treino.
* Desvantagem: exige manter metadados também para modelos que não serão usados em produção.

---

### 3.3 Atualização simples do modelo de inferência

Decisão:

O Worker de Inferência consultará um arquivo local, como `current_model.json`, antes de cada inferência. Se a versão de produção tiver mudado, ele recarregará o modelo.

Justificativa:

Essa estratégia é simples e adequada para o MVP. Ela permite que o Worker de Inferência use sempre o modelo marcado como produção, sem criar uma lógica de atualização mais complexa neste primeiro momento.

Trade-off:

* Vantagem: simples de implementar e fácil de explicar.
* Desvantagem: a atualização do modelo ocorre na próxima inferência, não exatamente no momento em que o modelo é promovido.

---

## 4. Arquitetura planejada

A solução será organizada em workers independentes, comunicando-se por RabbitMQ. O pipeline possui dois fluxos principais: o fluxo de dados/treino e o fluxo de inferência/coleta.

Fluxo de dados e treino:

```text
q.data.build
    ↓
Worker de Dados
    ↓
q.train.run
    ↓
Worker de Treino
    ↓
q.model.promoted
    ↓
Registro/atualização do modelo de produção
```

Fluxo de inferência e coleta:

```text
q.infer.request
    ↓
Worker de Inferência
    ↓
q.infer.result
    ↓
Worker de Coleta
    ↓
q.label.task + q.data.build
```

A ideia é que o Worker de Treino registre o modelo candidato e, caso ele passe no gate de qualidade, atualize a versão de produção. O Worker de Inferência, por sua vez, atende requisições de inferência usando a versão atual de produção do modelo.

Cada worker terá uma responsabilidade clara:

| Worker               | Consome           | Produz                          | Responsabilidade                                                                  |
| -------------------- | ----------------- | ------------------------------- | --------------------------------------------------------------------------------- |
| Worker de Dados      | `q.data.build`    | `q.train.run`                   | Criar dataset versionado a partir da fonte raw                                    |
| Worker de Treino     | `q.train.run`     | `q.model.promoted` se aprovado  | Treinar modelo candidato, avaliar e aplicar gate                                  |
| Worker de Inferência | `q.infer.request` | `q.infer.result`                | Rodar inferência com o modelo atual de produção                                   |
| Worker de Coleta     | `q.infer.result`  | `q.label.task` e `q.data.build` | Coletar casos de baixa confiança, simular anotação e disparar novo ciclo de dados |


---

## 5. Estratégia de implementação

O foco é construir um pipeline de MLOps funcional, orientado a mensagens, com RabbitMQ, workers independentes, versionamento básico de datasets/modelos, gate de qualidade e fechamento do ciclo de coleta. A prioridade é garantir que o ciclo principal funcione de ponta a ponta por mensageria.

A implementação será feita de forma incremental, em três etapas internas.

### Etapa 1 — Esqueleto de mensageria

Objetivo:

Validar a comunicação entre os serviços antes de integrar a lógica real dos scripts.

Tarefas:

* subir RabbitMQ com Docker Compose;
* criar as filas principais do pipeline;
* criar publishers de teste;
* criar workers fake para dados, treino, inferência e coleta;
* validar publicação, consumo e `ack` manual;
* verificar o estado das filas no RabbitMQ;
* documentar os contratos iniciais das mensagens.

Fluxos testados nesta etapa:

```text
q.data.build → Data Worker fake → q.train.run
q.train.run → Train Worker fake → q.model.promoted
q.infer.request → Infer Worker fake → q.infer.result
q.infer.result → Collect Worker fake → q.label.task + q.data.build
```

Critério de pronto:

O pipeline fake deve provar que os eventos conseguem circular entre as filas principais, que os workers conseguem consumir e publicar mensagens, e que o ciclo de coleta consegue gerar um novo evento `q.data.build`.

---

### Etapa 2 — Dados e treino reais

Objetivo:

Substituir os workers fake de dados e treino por workers reais, reaproveitando os scripts `prep_data.py` e `train.py`.

Tarefas:

* transformar a lógica de `prep_data.py` em um Data Worker real;
* consumir mensagens da fila `q.data.build`;
* criar datasets versionados, como `ds-001`, `ds-002`, etc.;
* salvar metadados do dataset em JSON;
* publicar eventos reais em `q.train.run`;
* transformar a lógica de `train.py` em um Train Worker real;
* treinar um modelo candidato a partir do dataset versionado;
* avaliar o modelo no conjunto de teste;
* aplicar um gate de qualidade;
* registrar runs de treino promovidas e não promovidas;
* atualizar o modelo de produção quando o candidato superar o baseline.

Critério de pronto:

A partir de uma mensagem em `q.data.build`, o sistema deve criar um dataset versionado, disparar um treino real, avaliar o modelo candidato e registrar se ele foi promovido ou não.

---

### Etapa 3 — Inferência, coleta e loop fechado reais

Objetivo:

Substituir os workers fake de inferência e coleta por workers reais, reaproveitando o script `infer.py` e implementando a lógica de seleção de casos para anotação simulada.

Tarefas:

* transformar a lógica de `infer.py` em um Infer Worker real;
* consumir mensagens da fila `q.infer.request`;
* carregar o modelo de produção;
* gerar um `inference_id` único para cada inferência;
* salvar os resultados de inferência;
* publicar eventos em `q.infer.result`;
* criar um Collect Worker real;
* selecionar casos de baixa confiança;
* gerar tarefas de anotação em `q.label.task`;
* usar os labels do oracle para simular anotação;
* reincorporar imagem e label à fonte raw;
* publicar um novo evento em `q.data.build`.

Critério de pronto:

Uma imagem enviada para inferência deve gerar um resultado rastreável por `inference_id`. Caso a predição tenha baixa confiança, o sistema deve simular a anotação, reincorporar o dado à fonte raw e disparar um novo ciclo de preparação de dataset.

---

### Critério geral de sucesso

O projeto será considerado funcional quando o pipeline conseguir executar o ciclo principal:

```text
dados → dataset versionado → treino → gate de qualidade → modelo promovido → inferência → coleta → novos dados → novo ciclo
```

Além disso, deve ser possível rastrear:

* qual dataset foi usado para treinar cada modelo;
* qual modelo está em produção;
* quais métricas justificaram a promoção ou rejeição de um modelo;
* qual modelo gerou cada inferência;
* quais inferências foram selecionadas para coleta.


---

## 6. Validação inicial do baseline

Antes de modificar a arquitetura, executei os scripts base manualmente.

### 6.1 Preparação do dataset

Comando executado:

```powershell
uv run python src/prep_data.py
```

Resultado observado:

O script de preparação de dados funcionou corretamente e gerou a estrutura de dataset no formato YOLO, incluindo o arquivo `dataset/data.yaml` e as divisões de treino, validação e teste.

Essa etapa confirmou que os dados em `data/raw` conseguem ser transformados em um dataset utilizável pelo script de treino.


### 6.2 Inferência

Comando executado:

```powershell
uv run python src/infer.py --model models\v0\best.onnx --image data\stream\images\BloodImage_00000.jpg
```

Resultado observado:

O modelo carregou corretamente e retornou detecções com classe, confiança e bounding box. A menor confiança observada foi baixa o suficiente para servir como exemplo de coleta por baixa confiança no Worker de Coleta.

### 6.3 Treino curto

Comando executado:

```powershell
uv run python src/train.py --data dataset\data.yaml --base models\v0\best.pt --epochs 1 --name smoke_test
```

Resultado observado:

O treino curto funcionou e gerou os artefatos:

```text
runs/smoke_test/weights/best.pt
runs/smoke_test/weights/best.onnx
runs/smoke_test/weights/last.pt
```

Essa validação mostrou que o ambiente está configurado corretamente e que os scripts base funcionam antes da integração com RabbitMQ.

---

## 7. Diário de bordo

### 17/06/2026 — Validação do baseline

#### O que fiz
- Rodei os scripts base manualmente.
- Validei `prep_data.py`, `train.py` e `infer.py`.

#### O que aprendi
- O treino usa `.pt`.
- A inferência usa `.onnx`.
- O `infer.py` já retorna detecções estruturadas.

#### Decisões tomadas
- Decidi manter versionamento local com JSON no MVP.
- Decidi tratar modelo reprovado no gate como run válida não promovida.

#### Próximo passo
- Subir RabbitMQ e criar workers mínimos.

### 18/06/2026 — Validação da mensageria e testes de publisher/consumer

#### O que fiz

Nesta etapa, concluí a Entrega 1, focada em validar a mensageria do pipeline antes de começar a transformar os scripts reais em workers.

Primeiro, subi o RabbitMQ usando Docker Compose e confirmei que o container estava rodando corretamente com status `healthy`. Depois, criei as principais filas que serão usadas no pipeline:

```text
q.data.build
q.train.run
q.model.promoted
q.infer.request
q.infer.result
q.label.task
```

Em seguida, criei publishers e workers fake para testar a comunicação entre as etapas do pipeline.

Validei primeiro o fluxo de dados para treino:

```text
q.data.build → Data Worker fake → q.train.run
```

Depois, validei o fluxo de treino para promoção de modelo:

```text
q.train.run → Train Worker fake → q.model.promoted
```

Também testei a parte de inferência:

```text
q.infer.request → Infer Worker fake → q.infer.result
```

Por fim, validei a etapa de coleta:

```text
q.infer.result → Collect Worker fake → q.label.task + q.data.build
```

Com isso, consegui simular o loop principal do pipeline: uma mensagem gera a próxima etapa, e o Collect Worker fake consegue simular uma coleta por baixa confiança, criar uma tarefa de anotação e disparar um novo ciclo de dados.

Além dos workers fake, também fiz os testes sugeridos a partir do feedback técnico recebido para entender melhor o funcionamento básico de um publisher e de um consumer.

Criei uma fila simples chamada:

```text
q.example.dataset
```

Depois criei um publisher simples que envia uma mensagem JSON com os parâmetros necessários para criação de um dataset. A mensagem usada tinha este formato:

```json
{
  "event": "data.build",
  "trigger": "manual",
  "raw_uri": "data/raw",
  "params": {
    "val_frac": 0.15,
    "test_frac": 0.15,
    "seed": 42
  }
}
```

Também criei um consumer simples para ler essa mensagem da fila. No consumer, testei o fluxo completo:

```text
publisher → fila → consumer → JSON/string → dicionário Python → uso dos parâmetros → ack
```

A mensagem chega ao consumer como bytes. Primeiro, converti para string usando:

```python
message_as_string = body.decode("utf-8")
```

Depois, converti a string JSON para dicionário Python usando:

```python
message = json.loads(message_as_string)
```

A partir disso, consegui acessar os parâmetros da mensagem dentro do código:

```python
raw_uri = message["raw_uri"]
val_frac = message["params"]["val_frac"]
test_frac = message["params"]["test_frac"]
seed = message["params"]["seed"]
```

Depois de processar a mensagem, o consumer enviou o `ack`, confirmando ao RabbitMQ que a mensagem foi processada com sucesso. Ao verificar as filas, confirmei que `q.example.dataset` ficou com zero mensagens pendentes, ou seja, o publisher enviou, a fila armazenou, o consumer consumiu e o RabbitMQ removeu a mensagem após o processamento.

#### O que aprendi

Aprendi melhor como o RabbitMQ organiza a comunicação entre serviços. Em vez de um worker chamar diretamente outro worker, cada etapa publica uma mensagem em uma fila, e o próximo worker consome essa mensagem quando estiver disponível.

Também entendi melhor alguns conceitos importantes:

* Um **publisher** é o processo que envia uma mensagem para uma fila.
* Um **consumer** é o processo que lê mensagens de uma fila.
* Um **worker** é um tipo de consumer que, além de ler a mensagem, executa uma tarefa do pipeline.
* Uma **fila** armazena mensagens até que algum consumer as processe.
* O **ack** é a confirmação enviada pelo consumer ao RabbitMQ dizendo que a mensagem foi processada com sucesso.
* Uma mensagem em **JSON** pode ser enviada como string e depois convertida para dicionário Python com `json.loads`.
* Um **worker fake** é uma versão provisória do worker real. Ele ainda não executa a lógica pesada, mas simula o comportamento esperado para validar o fluxo.

Também observei que o RabbitMQ mantém mensagens paradas quando ainda não existe consumidor para uma fila. Isso aconteceu, por exemplo, quando uma mensagem ficou em `q.train.run` até o Train Worker fake ser executado.

Esse comportamento é importante para o pipeline, porque permite que um serviço publique uma mensagem mesmo que o próximo worker ainda não esteja rodando naquele momento.

#### Decisões tomadas

Decidi começar validando a mensageria com exemplos simples e workers fake antes de integrar os scripts reais.

A principal razão foi reduzir risco. Se eu integrasse tudo de uma vez e algo quebrasse, seria mais difícil saber se o problema estaria no RabbitMQ, no formato da mensagem, no script de dados, no treino, no modelo ou nos caminhos dos arquivos.

Com os testes isolados, consegui validar uma camada por vez:

* primeiro, uma mensagem JSON simples;
* depois, o publisher;
* depois, o consumer;
* depois, a conversão de JSON para dicionário;
* depois, o uso dos parâmetros da mensagem;
* depois, o `ack`;
* por fim, os workers fake simulando o fluxo do pipeline.

O trade-off é que essa etapa ainda não executa o pipeline real de ML, porque os workers fake apenas simulam o comportamento. Mesmo assim, ela cria uma base mais segura para substituir gradualmente o comportamento fake pela lógica real de `prep_data.py`, `train.py` e `infer.py`.

Com isso, considero a Entrega 1 concluída: o esqueleto de mensageria foi validado tanto com um exemplo simples de publisher/consumer quanto com workers fake representando as etapas principais do pipeline.

#### Próximo passo

O próximo passo é começar a substituir o Data Worker fake por um Data Worker real.

Esse worker deverá consumir mensagens da fila `q.data.build`, converter a mensagem JSON em dicionário Python, extrair parâmetros como `raw_uri`, `val_frac`, `test_frac` e `seed`, executar a lógica de preparação de dados, criar uma versão de dataset como `ds-001`, salvar metadados dessa versão e publicar um evento real em `q.train.run`.

Depois disso, o próximo avanço será transformar o Train Worker fake em um Train Worker real, usando o script de treino, registrando métricas e aplicando o gate de qualidade.
