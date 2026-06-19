import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pika


RABBITMQ_HOST = "localhost"

INFER_RESULT_QUEUE = "q.infer.result"
LABEL_TASK_QUEUE = "q.label.task"

QUEUES = [
    INFER_RESULT_QUEUE,
    LABEL_TASK_QUEUE,
]

LOW_CONF_THRESHOLD = 0.50

ROOT_DIR = Path(__file__).resolve().parents[2]
CANDIDATES_DIR = ROOT_DIR / "storage" / "label_candidates"
CANDIDATES_FILE = CANDIDATES_DIR / "candidates.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_candidate(label_task: dict) -> None:
    """
    Saves selected low-confidence cases locally.

    This simulates accumulating candidates before rebuilding the dataset.
    The dataset rebuild should not happen for every single low-confidence case.
    """
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    with CANDIDATES_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(label_task) + "\n")


def publish_json(channel, queue: str, message: dict) -> None:
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(message).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )


def on_message(channel, method, properties, body) -> None:
    print("\n[Collect Worker Fake] Received inference result")

    try:
        message = json.loads(body.decode("utf-8"))

        inference_id = message.get("inference_id")
        image_uri = message.get("image_uri")
        model_version = message.get("model_version")
        min_conf = float(message.get("min_conf", 1.0))

        print(f"[Collect Worker Fake] inference_id={inference_id}")
        print(f"[Collect Worker Fake] image_uri={image_uri}")
        print(f"[Collect Worker Fake] model_version={model_version}")
        print(f"[Collect Worker Fake] min_conf={min_conf}")

        should_collect = min_conf < LOW_CONF_THRESHOLD

        if should_collect:
            label_task = {
                "event": "label.task",
                "label_task_id": f"label-{uuid4().hex[:8]}",
                "reason": "low_confidence",
                "inference_id": inference_id,
                "image_uri": image_uri,
                "model_version": model_version,
                "min_conf": min_conf,
                "threshold": LOW_CONF_THRESHOLD,
                "status": "pending_annotation",
                "created_at": utc_now(),
                "source_event": message,
            }

            publish_json(channel, LABEL_TASK_QUEUE, label_task)
            save_candidate(label_task)

            print("[Collect Worker Fake] Low-confidence case selected")
            print(f"[Collect Worker Fake] Published label task to {LABEL_TASK_QUEUE}")
            print(f"[Collect Worker Fake] Saved candidate to {CANDIDATES_FILE}")
            print("[Collect Worker Fake] Dataset rebuild was NOT triggered automatically")
        else:
            print("[Collect Worker Fake] Confidence is high enough; no label task created")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Collect Worker Fake] Message acknowledged")

    except Exception as error:
        print(f"[Collect Worker Fake] Error while processing message: {error}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST)
    )
    channel = connection.channel()

    for queue in QUEUES:
        channel.queue_declare(queue=queue, durable=True)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=INFER_RESULT_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )

    print(f"[Collect Worker Fake] Waiting for messages from {INFER_RESULT_QUEUE}")
    print("[Collect Worker Fake] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Collect Worker Fake] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()