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
        "event": "infer.request",
        "image_uri": "data/stream/images/BloodImage_00000.jpg",
        "model_version": "production",
    }

    channel.basic_publish(
        exchange="",
        routing_key="q.infer.request",
        body=json.dumps(message).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )

    print("[publisher] mensagem publicada em q.infer.request:")
    print(json.dumps(message, indent=2))

    connection.close()


if __name__ == "__main__":
    main()
