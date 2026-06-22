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


### 19/06/2026 — Padronização dos workers e início dos workers reais

#### O que fiz

Nesta etapa, incorporei os feedbacks recebidos na daily anterior e comecei a substituir partes do pipeline fake por workers reais.

Primeiro, ajustei a lógica do `Collect Worker fake`. Na versão anterior, quando uma inferência tinha baixa confiança, o worker publicava automaticamente uma nova mensagem em `q.data.build`, iniciando um novo ciclo de construção de dataset. Depois do feedback técnico recebido, alterei esse comportamento.

Agora, o `collect_worker_fake.py` apenas registra o caso como candidato para anotação e publica uma mensagem em:

```text
q.label.task
```

O worker também salva localmente o candidato em:

```text
storage/label_candidates/candidates.jsonl
```

Com isso, o rebuild do dataset não acontece mais automaticamente a cada baixa confiança. A reconstrução do dataset passa a ser uma etapa posterior, manual ou em lote, o que torna o fluxo mais realista.

Depois disso, implementei a sugestão de modularizar a parte repetida do RabbitMQ. Criei um módulo comum chamado:

```text
src/messaging/rabbitmq.py
```

Esse módulo centraliza funções usadas por vários workers, como:

```text
create_connection
declare_queues
publish_json
parse_json_body
```

Com isso, os workers não precisam mais repetir a lógica de conexão com RabbitMQ, declaração de filas, publicação de mensagens JSON e conversão do corpo da mensagem recebida.

Também implementei contratos de mensagens usando Pydantic no arquivo:

```text
src/contracts/messages.py
```

A ideia foi deixar de trabalhar apenas com dicionários soltos e passar a validar as mensagens recebidas pelos workers. Foram criados contratos para os principais eventos do pipeline, como:

```text
DataBuildMessage
TrainRunEvent
ModelPromotedEvent
InferRequestEvent
InferResultEvent
LabelTaskEvent
```

A partir disso, os workers passaram a validar as mensagens na entrada. Por exemplo, no Data Worker, a mensagem recebida é convertida de JSON para dicionário e depois validada com:

```python
data_build_message = DataBuildMessage.model_validate(raw_message)
```

Depois de criar o helper comum de RabbitMQ e os contratos Pydantic, refatorei todos os workers fake para seguirem esse mesmo padrão:

```text
data_worker_fake.py
train_worker_fake.py
infer_worker_fake.py
collect_worker_fake.py
```

Com isso, todos os workers fake passaram a ter uma estrutura mais consistente:

```text
mensagem RabbitMQ
→ parse_json_body
→ validação com Pydantic
→ execução da lógica do worker
→ publicação do próximo evento com publish_json
→ ack
```

Depois de padronizar os workers fake, comecei a implementar os workers reais.

O primeiro foi o:

```text
src/workers/data_worker_real.py
```

Esse worker consome mensagens da fila:

```text
q.data.build
```

Em seguida, valida a mensagem com `DataBuildMessage`, extrai parâmetros como `raw_uri`, `val_frac`, `test_frac` e `seed`, executa o `prep_data.py` e cria um dataset versionado dentro de:

```text
storage/datasets/
```

No teste realizado, o worker criou corretamente uma versão de dataset e gerou os splits:

```text
train=63
val=14
test=14
```

Depois disso, o `data_worker_real.py` publicou um evento real em:

```text
q.train.run
```

Em seguida, implementei o:

```text
src/workers/train_worker_real.py
```

Esse worker consome mensagens da fila:

```text
q.train.run
```

Ele valida o evento com `TrainRunEvent`, executa o `train.py`, captura a métrica de avaliação `TEST mAP50`, compara com o baseline definido e aplica o gate de qualidade.

Durante o teste, apareceu um problema de encoding no Windows ao capturar o output do `train.py` com `subprocess.run`. O erro acontecia porque o terminal tentava decodificar alguns caracteres usando a codificação padrão do Windows. Para corrigir isso, ajustei o `subprocess.run` usando:

```python
encoding="utf-8"
errors="replace"
```

Depois dessa correção, o `train_worker_real.py` conseguiu executar o treino real, exportar o modelo em ONNX, extrair a métrica e aplicar o gate de qualidade.

No teste final, o modelo obteve:

```text
TEST mAP50=0.8208
Baseline mAP50=0.5000
```

Como a métrica ficou acima do baseline, o modelo passou no gate de qualidade. O worker registrou os artefatos do modelo em:

```text
storage/models/
```

e publicou o evento de promoção em:

```text
q.model.promoted
```

Também validei o fluxo integrado entre os dois workers reais já implementados:

```text
q.data.build → Data Worker real → q.train.run → Train Worker real → q.model.promoted
```

Esse teste mostrou que o pipeline já consegue sair de uma mensagem de construção de dataset, criar um dataset versionado, iniciar um treino real, aplicar o gate de qualidade e promover um modelo aprovado.

#### O que aprendi

Aprendi melhor a importância de separar a lógica de infraestrutura da lógica de negócio dos workers.

Antes, cada worker tinha sua própria lógica para conectar no RabbitMQ, declarar filas, publicar mensagens e converter o corpo da mensagem. Isso funcionava, mas criava repetição e deixava o código mais difícil de manter.

Com o helper comum em `src/messaging/rabbitmq.py`, os workers ficaram mais simples. Cada worker agora se concentra mais na sua própria responsabilidade:

* o Data Worker prepara e versiona datasets;
* o Train Worker treina, avalia e aplica o gate de qualidade;
* o Infer Worker simula ou executa inferência;
* o Collect Worker seleciona casos para anotação.

Também aprendi melhor o papel do Pydantic em um sistema orientado a mensagens. Em vez de confiar que todo dicionário recebido terá o formato correto, agora o worker valida explicitamente o contrato da mensagem antes de processá-la.

Isso ajuda a evitar erros silenciosos. Por exemplo, se uma mensagem `data.build` vier sem `raw_uri` ou com `val_frac` inválido, o erro aparece logo na entrada do worker, antes de executar qualquer lógica mais pesada.

Outro aprendizado importante foi sobre o gate de qualidade. O modelo treinado pode ser considerado uma run válida mesmo que não seja promovido. A promoção só acontece se a métrica superar o baseline definido. Isso separa duas ideias diferentes:

```text
treinar um modelo
promover um modelo para produção
```

Também entendi melhor como manter rastreabilidade entre dataset e modelo. O evento `train.run` carrega a versão do dataset, e o evento `model.promoted` mantém referência ao `dataset_version`. Assim, é possível saber qual dataset foi usado para gerar determinado modelo.

Além disso, aprendi que, ao rodar subprocessos no Windows, pode ser necessário controlar explicitamente a codificação do output, principalmente quando bibliotecas externas imprimem caracteres que não são compatíveis com a codificação padrão do terminal.

#### Decisões tomadas

Decidi não partir diretamente para todos os workers reais antes de padronizar os workers fake.

A razão foi reduzir risco. Antes de integrar `prep_data.py`, `train.py` e futuramente `infer.py` dentro dos workers reais, considerei melhor garantir que a arquitetura fake já estivesse organizada com:

```text
helper comum de RabbitMQ
contratos Pydantic
validação de mensagens
ack/nack
publicação padronizada de eventos
```

Também decidi manter os workers fake no projeto. Eles continuam úteis para testes rápidos da mensageria, sem precisar rodar tarefas mais pesadas como treino de modelo.

Outra decisão foi criar os workers reais de forma incremental. Primeiro implementei o `data_worker_real.py`, depois o `train_worker_real.py`. Isso permitiu testar o fluxo por partes e identificar problemas mais facilmente.

Também decidi manter o gate de qualidade no Train Worker real. O worker só publica `q.model.promoted` se o modelo passar no critério mínimo definido. Caso contrário, o treino ainda seria uma run válida, mas sem promoção do modelo.

Por fim, decidi manter o `q.data.build` fora do disparo automático do Collect Worker. O Collect Worker agora apenas cria tarefas de anotação e salva candidatos. A reconstrução do dataset deverá acontecer em uma etapa posterior, quando houver dados anotados suficientes.

Com isso, considero a Entrega 2 concluída. A etapa atual já demonstra a evolução do pipeline de um esqueleto fake de mensageria para uma estrutura mais organizada, com contratos de mensagens, helper comum de RabbitMQ e workers reais para dados e treino.

#### Próximo passo

O próximo passo será iniciar a Entrega 3.

A prioridade será implementar o `Inference Worker real`, conectando o modelo promovido ao fluxo real de inferência. Esse worker deverá consumir mensagens de:

```text
q.infer.request
```

executar inferência com o modelo ONNX ativo e publicar os resultados em:

```text
q.infer.result
```

Depois disso, o próximo avanço será transformar o fluxo de coleta em uma etapa mais real, conectando os resultados de inferência ao processo de seleção de exemplos de baixa confiança e preparação para anotação.

Com isso, o objetivo será fechar melhor o ciclo:

```text
dados → treino → promoção → inferência → coleta → anotação → novos dados
```

### 22/06/2026 — Fechamento do ciclo real de inferência, coleta, anotação simulada e rastreabilidade

#### O que fiz

Nesta etapa, avancei na Entrega 3 com o objetivo de fechar o ciclo real do pipeline de MLOps depois da promoção do modelo.

Na etapa anterior, o pipeline já possuía os workers reais de dados e treino funcionando:

```text
q.data.build
→ Data Worker real
→ q.train.run
→ Train Worker real
→ q.model.promoted
```

Hoje, a prioridade foi completar o restante do loop:

```text
modelo promovido
→ inferência
→ coleta de baixa confiança
→ anotação simulada
→ reincorporação em data/raw
→ novo ciclo de dados
```

Primeiro, ajustei o fluxo de promoção de modelo para deixar explícito qual modelo está em produção. Para isso, o `Train Worker real` passou a atualizar um ponteiro local de produção em:

```text
storage/models/production.json
```

Esse arquivo guarda informações como:

```text
model_version
model_uri
dataset_version
metrics
baseline
base_model
updated_at
```

A partir disso, o serviço de inferência não precisa depender diretamente do evento `q.model.promoted` para saber qual modelo deve usar. O evento continua existindo como registro da promoção, mas a referência operacional do modelo ativo passa a ser o `production.json`.

Depois disso, implementei o:

```text
src/workers/infer_worker_real.py
```

Esse worker consome mensagens da fila:

```text
q.infer.request
```

Ele valida a mensagem usando o contrato Pydantic `InferRequestEvent`, lê o ponteiro de produção, carrega o modelo ONNX ativo e executa inferência real usando a classe `Detector`.

O worker também verifica se o modelo ativo mudou. Antes de inferir, ele consulta o `production.json`; se a versão do modelo apontada no arquivo for diferente da versão carregada em memória, ele recarrega o modelo automaticamente.

Com isso, o fluxo do worker de inferência ficou:

```text
q.infer.request
→ lê production.json
→ carrega ou recarrega o modelo ONNX ativo
→ executa inferência real
→ gera inference_id
→ salva resultado
→ publica q.infer.result
```

No teste realizado, o worker executou inferência sobre a imagem:

```text
data/stream/images/BloodImage_00000.jpg
```

O resultado foi salvo em:

```text
storage/inference_results/results.jsonl
```

e publicado em:

```text
q.infer.result
```

A inferência gerou o identificador:

```text
inf-da3eb216
```

com `min_conf` abaixo do limite definido, o que permitiu testar a etapa de coleta.

Em seguida, implementei o:

```text
src/workers/collect_worker_real.py
```

Esse worker consome mensagens da fila:

```text
q.infer.result
```

Ele valida o resultado de inferência com `InferResultEvent`, verifica o campo `min_conf` e compara com o threshold:

```text
LOW_CONF_THRESHOLD = 0.50
```

Quando a confiança mínima fica abaixo desse limite, o worker marca o caso como candidato à anotação. No teste realizado, a inferência teve:

```text
min_conf=0.32289034128189087
```

Como esse valor é menor que `0.50`, o `Collect Worker real` criou uma tarefa em:

```text
q.label.task
```

Também salvou metadados do candidato em:

```text
storage/label_candidates/candidates.jsonl
```

e copiou a imagem candidata para:

```text
storage/label_candidates/images/
```

Mantive a decisão de o `Collect Worker real` não disparar `q.data.build` automaticamente. Ele apenas seleciona o caso incerto e cria a tarefa de anotação. A reconstrução do dataset continua sendo uma etapa posterior.

Depois disso, implementei o:

```text
src/workers/oracle_annotation_worker.py
```

Esse worker representa a anotação simulada da atividade. Como os rótulos verdadeiros das imagens de stream já existem no BCCD, mas ficam separados até a etapa de anotação, o worker funciona como um oráculo digital.

Ele consome mensagens de:

```text
q.label.task
```

valida a mensagem com `LabelTaskEvent`, identifica a imagem selecionada e procura o rótulo verdadeiro correspondente em:

```text
data/oracle/labels/
```

No primeiro teste, a imagem selecionada foi:

```text
data/stream/images/BloodImage_00000.jpg
```

e o rótulo verdadeiro correspondente foi encontrado em:

```text
data/oracle/labels/BloodImage_00000.txt
```

O worker então copiou imagem e label para a fonte `raw`:

```text
data/raw/images/oracle_label-test-oracle-001_BloodImage_00000.jpg
data/raw/labels/oracle_label-test-oracle-001_BloodImage_00000.txt
```

Também salvou um registro de rastreabilidade em:

```text
storage/oracle_annotations/annotations.jsonl
```

Com isso, a fonte `raw` cresceu com um novo exemplo anotado.

Após validar a anotação simulada, ajustei o `Oracle Annotation Worker` para fechar o loop de feedback automaticamente.

Antes, depois que o worker reinjetava imagem e label em `data/raw`, eu ainda precisava publicar manualmente uma nova mensagem em:

```text
q.data.build
```

Com o ajuste feito hoje, o `Oracle Annotation Worker` passou a publicar automaticamente um evento `data.build` com:

```text
trigger="feedback"
```

logo depois de copiar a imagem e o rótulo verdadeiro para a fonte `raw`.

Com isso, o fluxo passou a ser:

```text
q.label.task
→ Oracle Annotation Worker
→ data/raw recebe imagem + label anotados
→ q.data.build automático com trigger="feedback"
→ Data Worker real
→ novo dataset versionado
```

Esse comportamento mantém a decisão arquitetural de não deixar o `Collect Worker` disparar rebuild diretamente. O `Collect Worker` continua apenas selecionando casos de baixa confiança e criando tarefas de anotação. Quem fecha o loop é o `Oracle Annotation Worker`, mas somente depois que existe de fato um novo dado anotado em `data/raw`.

Validei esse fluxo observando que, após o `Oracle Annotation Worker` consumir uma tarefa em `q.label.task`, a fila `q.data.build` ficou com uma nova mensagem pronta enquanto o `Data Worker real` ainda não estava rodando:

```text
q.label.task    0
q.data.build    1
```

Em seguida, executei o `Data Worker real`, que consumiu esse `q.data.build` automático e criou um novo dataset versionado:

```text
ds-20260622-202001-870c13
```

O novo dataset foi criado com:

```text
train=65
val=14
test=14
```

e o log do Data Worker registrou:

```text
Added this cycle: 1
Published train event to q.train.run
```

Isso confirmou que o dado anotado pelo oráculo entrou automaticamente no próximo ciclo do pipeline.

Depois, fiz um ajuste importante no `Train Worker real`. A atividade pede que o treino faça fine-tune a partir do checkpoint vigente, e não sempre a partir do modelo inicial `v0`.

Antes, o treino usava o checkpoint inicial como base fixa. Ajustei a lógica para:

```text
se storage/models/production.json existir:
    usar o modelo promovido vigente como base
senão:
    usar models/v0/best.pt
```

Com isso, o `Train Worker real` passou a ler o ponteiro de produção e identificar o checkpoint atual. No teste, ele usou como base:

```text
base_model_version=model-20260622-160102-df2d2e
base_model_path=C:\dev\mlops-pipeline-challenge\storage\models\model-20260622-160102-df2d2e\best.pt
```

Isso confirmou que o novo treino não partiu mais diretamente do `v0`, mas sim do modelo promovido vigente.

O novo treino obteve:

```text
TEST mAP50=0.8552
Baseline mAP50=0.5000
```

Como a métrica ficou acima do baseline, o modelo passou no gate de qualidade, foi registrado em:

```text
storage/models/model-20260622-182157-34ab11
```

e o ponteiro de produção foi atualizado para esse novo modelo.

Também criei duas ferramentas auxiliares para melhorar a demonstração e atender melhor aos requisitos de rastreabilidade da inferência.

A primeira foi:

```text
src/tools/check_inference_status.py
```

Esse comando lê o `production.json` e mostra o status do serviço de inferência e do modelo ativo. No teste, ele retornou:

```text
status=healthy
model_version=model-20260622-182157-34ab11
model_file_exists=true
dataset_version=ds-20260622-172922-5d45af
mAP50=0.8552
base_model=model-20260622-160102-df2d2e
```

Com isso, ficou possível consultar rapidamente qual modelo está em produção, qual dataset gerou esse modelo, qual foi a métrica obtida e qual checkpoint serviu de base.

A segunda ferramenta foi:

```text
src/tools/get_inference_result.py
```

Ela permite recuperar um resultado de inferência pelo `inference_id`, a partir do arquivo:

```text
storage/inference_results/results.jsonl
```

Testei com:

```text
inf-da3eb216
```

e o comando retornou corretamente o JSON completo da inferência, incluindo:

```text
inference_id
model_version
status
image_uri
latency_ms
min_conf
detections
source_event
```

Também testei um ID inexistente:

```text
inf-nao-existe
```

e o comando retornou:

```json
{
  "status": "not_found",
  "inference_id": "inf-nao-existe",
  "results_file": "C:\\dev\\mlops-pipeline-challenge\\storage\\inference_results\\results.jsonl"
}
```

Isso deixou explícito que as inferências são persistidas e recuperáveis posteriormente por identificador.

Por fim, ajustei o `Data Worker real` para melhorar a rastreabilidade dos datasets versionados. Antes, o campo `added_this_cycle` estava sempre como `0`. Isso não representava bem o ciclo, porque depois da anotação oracle havia de fato um novo exemplo reincorporado à fonte `raw`.

Implementei então uma lógica para acompanhar o número de anotações oracle já processadas. O worker passou a ler:

```text
storage/oracle_annotations/annotations.jsonl
```

e manter um estado local em:

```text
storage/datasets/_data_worker_state.json
```

Com isso, o Data Worker consegue calcular quantas novas anotações entraram desde o último build.

Também passei a salvar, dentro de cada dataset versionado, um arquivo:

```text
metadata.json
```

Esse arquivo registra:

```text
dataset_version
dataset_uri
raw_uri
raw_dir
classes
counts
added_this_cycle
oracle_annotation_count
created_at
source_event
```

No primeiro build após a anotação oracle, o `metadata.json` registrou:

```json
{
  "dataset_version": "ds-20260622-183654-3b96fe",
  "counts": {
    "train": 64,
    "val": 14,
    "test": 14
  },
  "added_this_cycle": 1,
  "oracle_annotation_count": {
    "previous": 0,
    "current": 1
  }
}
```

Isso mostra que o dataset foi criado com uma nova anotação reincorporada no ciclo.

Depois, disparei um novo build sem criar nenhuma nova anotação oracle. Nesse caso, o metadata registrou:

```json
{
  "added_this_cycle": 0,
  "oracle_annotation_count": {
    "previous": 1,
    "current": 1
  }
}
```

Esse comportamento está correto, porque não houve nova anotação desde o último build.

Também preparei a atualização do `README.md`, documentando o ciclo completo, os workers, as filas RabbitMQ, os comandos de reprodução, o healthcheck, a recuperação por `inference_id`, a rastreabilidade de datasets/modelos e as limitações atuais.

Os principais commits desta etapa foram:

```text
Add real inference worker with production model reload
Add real collect worker for low-confidence samples
Add oracle annotation worker
Use current production checkpoint for training
Add inference status and result lookup tools
Track dataset metadata and added samples
```

#### O que aprendi

Aprendi melhor a diferença entre evento de promoção e estado operacional do modelo em produção.

O evento:

```text
q.model.promoted
```

é importante para registrar que um modelo passou no gate de qualidade e foi promovido. Porém, para o serviço de inferência, é mais prático consultar um ponteiro de produção:

```text
storage/models/production.json
```

Assim, o `Inference Worker real` não precisa depender diretamente da fila de promoção para funcionar. Mesmo que ele seja iniciado depois do treino, ele consegue descobrir sozinho qual modelo está ativo.

Também aprendi melhor o papel do checkpoint vigente no ciclo de vida do modelo. Em um pipeline contínuo, o próximo treino não deve sempre partir do `v0`. Depois que um modelo foi promovido, ele passa a ser o ponto de partida natural para o próximo fine-tuning.

Com isso, o ciclo deixa de ser:

```text
v0 → modelo novo
v0 → modelo novo
v0 → modelo novo
```

e passa a ser:

```text
v0 → modelo 1 → modelo 2 → modelo 3
```

Isso representa melhor um processo de melhoria incremental.

Outro aprendizado foi sobre a separação entre coleta, anotação e reconstrução do dataset.

A coleta identifica um caso útil ou incerto. A anotação simulada recupera o rótulo verdadeiro desse caso. A reconstrução do dataset só deve acontecer depois que o novo dado anotado realmente existe em `data/raw`.

Por isso, mantive a decisão de não deixar o `Collect Worker` publicar `q.data.build` diretamente. Uma inferência de baixa confiança ainda não significa que existe um novo dado pronto para treino; ela apenas indica que aquele exemplo deve virar uma tarefa de anotação.

Com o ajuste feito hoje, o gatilho automático passou a acontecer no ponto correto do fluxo: o `Oracle Annotation Worker`.

Depois que ele consome uma tarefa em:

```text
q.label.task
```

recupera o rótulo verdadeiro e reinjeta imagem + label em:

```text
data/raw
```

ele publica automaticamente uma nova mensagem em:

```text
q.data.build
```

com:

```text
trigger="feedback"
```

Assim, o loop fica automático sem reconstruir o dataset imediatamente após qualquer baixa confiança. O rebuild acontece apenas depois que a anotação simulada foi concluída.

Também aprendi a importância de persistir resultados de inferência por `inference_id`. Sem isso, a inferência seria apenas uma mensagem transitória na fila. Ao salvar os resultados em `results.jsonl` e criar uma ferramenta de recuperação, fica possível auditar o que foi inferido, qual modelo gerou a predição e quais detecções foram produzidas.

Além disso, aprendi que a rastreabilidade do dataset não depende apenas de salvar os arquivos de imagem e label. É importante registrar metadados do build, como a versão do dataset, a quantidade de imagens em cada split, o evento que originou o build e quantas novas amostras foram incorporadas naquele ciclo.

O campo `added_this_cycle` ajudou a deixar explícito quando o dataset cresceu de fato por causa da anotação simulada.

#### Decisões tomadas

Decidi manter o `q.model.promoted` como evento de rastreabilidade, mas usar o arquivo `production.json` como fonte operacional para o modelo ativo.

Essa decisão reduz o acoplamento entre treino e inferência. O Train Worker publica o evento de promoção, atualiza o ponteiro de produção, e o Inference Worker apenas consulta esse ponteiro para carregar o modelo correto.

Também decidi que o `Train Worker real` deve usar o checkpoint vigente quando ele existir, e cair para `v0` apenas se ainda não houver modelo promovido. Isso deixa o ciclo mais próximo de um pipeline real de fine-tuning contínuo.

Decidi manter o `Collect Worker real` sem publicar `q.data.build` diretamente. Ele apenas seleciona casos de baixa confiança e cria tarefas de anotação em:

```text
q.label.task
```

A razão é que baixa confiança ainda não significa novo dado pronto para treino. Ela apenas indica que aquele exemplo deve passar por uma etapa de anotação.

Por outro lado, depois que o `Oracle Annotation Worker` processa essa tarefa e reinjeta imagem e label em `data/raw`, já existe um novo exemplo anotado disponível. Por isso, decidi que o `Oracle Annotation Worker` deve publicar automaticamente um novo `q.data.build` com:

```text
trigger="feedback"
```

Essa decisão fecha o loop automaticamente, mas mantém uma separação correta de responsabilidades:

```text
Collect Worker → seleciona candidatos
Oracle Annotation Worker → simula anotação e publica novo build
Data Worker → cria novo dataset
```

Também decidi implementar a anotação simulada como um worker próprio:

```text
src/workers/oracle_annotation_worker.py
```

Isso deixa a arquitetura mais clara, porque seleção e anotação são responsabilidades diferentes.

Outra decisão foi registrar os resultados de inferência em um arquivo JSONL e criar uma ferramenta de busca por `inference_id`. Isso atende à necessidade de que o resultado da inferência seja recuperável depois e facilita a demonstração.

Também decidi criar uma ferramenta simples de status da inferência, baseada no `production.json`, em vez de implementar uma API REST neste momento. Para o objetivo atual do projeto, a ferramenta de status já permite verificar qual modelo está ativo, qual dataset originou esse modelo e se o arquivo ONNX existe.

Por fim, decidi adicionar `metadata.json` nos datasets versionados e calcular `added_this_cycle` com base nas anotações oracle processadas. Isso melhorou a rastreabilidade entre anotação simulada, crescimento da fonte raw e criação de novos datasets.

Com isso, considero que o projeto já demonstra o ciclo de MLOps de ponta a ponta, com workers independentes, comunicação por RabbitMQ, versionamento de dataset, treino com gate de qualidade, modelo em produção rastreável, inferência real, coleta de baixa confiança, anotação simulada, publicação automática de `q.data.build` após feedback e reincorporação dos dados ao ciclo.
