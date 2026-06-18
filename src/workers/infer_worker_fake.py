import json
import random
import uuid
from datetime import datetime, timezone

import pika


QUEUES = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
]


def declare_queues(channel) -> None:
    for queue in QUEUES:
        channel.queue_declare(queue=queue, durable=True)


def on_message(channel, method, properties, body) -> None:
    print("\n[infer_worker] mensagem recebida de q.infer.request:")
    print(body.decode("utf-8"))

    message = json.loads(body)

    inference_id = f"inf-{uuid.uuid4().hex[:8]}"

    detections = [
        {
            "cls": "RBC",
            "conf": round(random.uniform(0.70, 0.95), 2),
            "box": [157, 76, 244, 171],
        },
        {
            "cls": "WBC",
            "conf": round(random.uniform(0.60, 0.90), 2),
            "box": [256, 183, 507, 373],
        },
        {
            "cls": "Platelets",
            "conf": round(random.uniform(0.20, 0.45), 2),
            "box": [80, 330, 130, 380],
        },
    ]

    min_conf = min(det["conf"] for det in detections)

    result_event = {
        "event": "infer.result",
        "inference_id": inference_id,
        "model_version": message["model_version"],
        "status": "success",
        "image_uri": message["image_uri"],
        "latency_ms": random.randint(40, 120),
        "min_conf": min_conf,
        "ts": datetime.now(timezone.utc).isoformat(),
        "detections": detections,
        "source_event": message,
    }

    channel.basic_publish(
        exchange="",
        routing_key="q.infer.result",
        body=json.dumps(result_event).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )

    channel.basic_ack(delivery_tag=method.delivery_tag)

    print("[infer_worker] inferência fake publicada em q.infer.result:")
    print(json.dumps(result_event, indent=2))


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    declare_queues(channel)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="q.infer.request",
        on_message_callback=on_message,
        auto_ack=False,
    )

    print("[infer_worker] aguardando mensagens em q.infer.request...")
    print("[infer_worker] pressione CTRL+C para parar.")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[infer_worker] encerrando...")
        channel.stop_consuming()

    connection.close()


if __name__ == "__main__":
    main()
