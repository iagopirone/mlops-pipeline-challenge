import json

import pika


QUEUES = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
]


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    for queue in QUEUES:
        channel.queue_declare(queue=queue, durable=True)

    message = {
        "event": "data.build",
        "trigger": "manual",
        "raw_uri": "data/raw",
        "params": {
            "val_frac": 0.15,
            "test_frac": 0.15
        }
    }

    channel.basic_publish(
        exchange="",
        routing_key="q.data.build",
        body=json.dumps(message).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )

    print("[publisher] mensagem publicada em q.data.build:")
    print(json.dumps(message, indent=2))

    connection.close()


if __name__ == "__main__":
    main()
