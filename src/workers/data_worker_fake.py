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
    print("\n[data_worker] mensagem recebida de q.data.build:")
    print(body.decode("utf-8"))

    message = json.loads(body)

    train_event = {
        "event": "train.run",
        "dataset_version": "ds-001",
        "dataset_uri": "storage/datasets/ds-001",
        "classes": ["RBC", "WBC", "Platelets"],
        "counts": {
            "train": 0,
            "val": 0,
            "test": 0,
        },
        "added_this_cycle": 0,
        "source_event": message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    channel.basic_publish(
        exchange="",
        routing_key="q.train.run",
        body=json.dumps(train_event).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )

    channel.basic_ack(delivery_tag=method.delivery_tag)

    print("[data_worker] mensagem publicada em q.train.run:")
    print(json.dumps(train_event, indent=2))


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    declare_queues(channel)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="q.data.build",
        on_message_callback=on_message,
        auto_ack=False,
    )

    print("[data_worker] aguardando mensagens em q.data.build...")
    print("[data_worker] pressione CTRL+C para parar.")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[data_worker] encerrando...")
        channel.stop_consuming()

    connection.close()


if __name__ == "__main__":
    main()
