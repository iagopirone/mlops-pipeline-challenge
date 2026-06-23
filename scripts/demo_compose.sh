bash id="sfk7qt"
#!/usr/bin/env bash
set -euo pipefail

KEEP_QUEUES="false"
WAIT_SECONDS="${WAIT_SECONDS:-12}"

if [[ "${1:-}" == "--keep-queues" ]]; then
  KEEP_QUEUES="true"
fi

echo
echo "=== MLOps Pipeline Compose Demo ==="
echo

echo "[1/5] Starting RabbitMQ and workers with Docker Compose..."
docker compose up -d --build

echo
echo "[2/5] Current services:"
docker compose ps

QUEUES=(
  "q.data.build"
  "q.train.run"
  "q.model.promoted"
  "q.infer.request"
  "q.infer.result"
  "q.label.task"
)

if [[ "$KEEP_QUEUES" == "false" ]]; then
  echo
  echo "[3/5] Purging queues for a clean demo..."
  for queue in "${QUEUES[@]}"; do
    docker exec mlops-rabbitmq rabbitmqctl purge_queue "$queue" >/dev/null
  done
else
  echo
  echo "[3/5] Keeping current queue contents because --keep-queues was used."
fi

echo
echo "[4/5] Publishing a low-confidence inference result to q.infer.result..."

TMP_FILE="$(mktemp)"

cat > "$TMP_FILE" <<'PY'
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
PY

uv run python "$TMP_FILE"
rm -f "$TMP_FILE"

echo
echo "Waiting ${WAIT_SECONDS} seconds for workers to process the event..."
sleep "$WAIT_SECONDS"

echo
echo "[5/5] Queue state:"
docker exec mlops-rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers

echo
echo "Recent worker logs:"
docker compose logs --tail=30 collect_worker
docker compose logs --tail=30 oracle_annotation_worker
docker compose logs --tail=30 data_worker
docker compose logs --tail=30 train_worker

echo
echo "Demo finished."
echo
echo "Useful follow-up commands:"
echo "  docker compose ps"
echo "  docker compose logs --tail=50 data_worker"
echo "  docker compose logs --tail=50 train_worker"
echo "  docker exec mlops-rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers"
