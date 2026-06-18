import json
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
    print("\n[train_worker] mensagem recebida de q.train.run:")
    print(body.decode("utf-8"))

    message = json.loads(body)

    model_event = {
        "event": "model.promoted",
        "model_version": "m-001",
        "base_model": "models/v0/best.pt",
        "model_uri": "storage/models/m-001/best.onnx",
        "dataset_version": message["dataset_version"],
        "metrics": {
            "mAP50": 0.51,
            "per_class": {
                "RBC": 0.70,
                "WBC": 0.55,
                "Platelets": 0.28
            }
        },
        "baseline": {
            "model_version": "m-000",
            "mAP50": 0.50
        },
        "promoted": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_event": message,
    }

    channel.basic_publish(
        exchange="",
        routing_key="q.model.promoted",
        body=json.dumps(model_event).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )

    channel.basic_ack(delivery_tag=method.delivery_tag)

    print("[train_worker] modelo promovido e mensagem publicada em q.model.promoted:")
    print(json.dumps(model_event, indent=2))


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    declare_queues(channel)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="q.train.run",
        on_message_callback=on_message,
        auto_ack=False,
    )

    print("[train_worker] aguardando mensagens em q.train.run...")
    print("[train_worker] pressione CTRL+C para parar.")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[train_worker] encerrando...")
        channel.stop_consuming()

    connection.close()


if __name__ == "__main__":
    main()
