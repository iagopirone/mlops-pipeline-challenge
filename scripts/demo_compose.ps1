param(
    [switch]$KeepQueues,
    [int]$WaitSeconds = 12
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== MLOps Pipeline Compose Demo ==="
Write-Host ""

Write-Host "[1/5] Starting RabbitMQ and workers with Docker Compose..."
docker compose up -d --build

Write-Host ""
Write-Host "[2/5] Current services:"
docker compose ps

$queues = @(
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task"
)

if (-not $KeepQueues) {
    Write-Host ""
    Write-Host "[3/5] Purging queues for a clean demo..."
    foreach ($queue in $queues) {
        docker exec mlops-rabbitmq rabbitmqctl purge_queue $queue | Out-Null
    }
} else {
    Write-Host ""
    Write-Host "[3/5] Keeping current queue contents because -KeepQueues was used."
}

Write-Host ""
Write-Host "[4/5] Publishing a low-confidence inference result to q.infer.result..."

$tempFile = New-TemporaryFile

@'
import json
from datetime import datetime, timezone

import pika

message = {
    "event": "infer.result",
    "inference_id": "inf-compose-demo-low-conf",
    "model_version": "production",
    "status": "success",
    "image_uri": "data/stream/images/BloodImage_00000.jpg",
    "latency_ms": 35.5,
    "min_conf": 0.30,
    "ts": datetime.now(timezone.utc).isoformat(),
    "detections": [
        {
            "cls": "RBC",
            "conf": 0.30,
            "box": [10.0, 20.0, 80.0, 90.0],
        }
    ],
    "source_event": {
        "event": "infer.request",
        "image_uri": "data/stream/images/BloodImage_00000.jpg",
        "model_version": "production",
    },
}

queues = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
]

credentials = pika.PlainCredentials("guest", "guest")
parameters = pika.ConnectionParameters(
    host="localhost",
    port=5672,
    credentials=credentials,
)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()

for queue in queues:
    channel.queue_declare(queue=queue, durable=True)

channel.basic_publish(
    exchange="",
    routing_key="q.infer.result",
    body=json.dumps(message).encode("utf-8"),
    properties=pika.BasicProperties(
        delivery_mode=2,
        content_type="application/json",
    ),
)

connection.close()

print("Published low-confidence infer.result event to q.infer.result")
'@ | Set-Content -Encoding utf8 $tempFile

uv run python $tempFile
Remove-Item $tempFile

Write-Host ""
Write-Host "Waiting $WaitSeconds seconds for workers to process the event..."
Start-Sleep -Seconds $WaitSeconds

Write-Host ""
Write-Host "[5/5] Queue state:"
docker exec mlops-rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers

Write-Host ""
Write-Host "Recent worker logs:"
docker compose logs --tail=30 collect_worker
docker compose logs --tail=30 oracle_annotation_worker
docker compose logs --tail=30 data_worker
docker compose logs --tail=30 train_worker

Write-Host ""
Write-Host "Demo finished."
Write-Host ""
Write-Host "Useful follow-up commands:"
Write-Host "  docker compose ps"
Write-Host "  docker compose logs --tail=50 data_worker"
Write-Host "  docker compose logs --tail=50 train_worker"
Write-Host "  docker exec mlops-rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers"