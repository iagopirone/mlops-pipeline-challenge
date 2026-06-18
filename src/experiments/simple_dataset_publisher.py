import json

import pika


QUEUE_NAME = "q.example.dataset"


def main() -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    message = {
        "event": "data.build",
        "trigger": "manual",
        "raw_uri": "data/raw",
        "params": {
            "val_frac": 0.15,
            "test_frac": 0.15,
            "seed": 42
        }
    }

    message_as_string = json.dumps(message)

    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=message_as_string.encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )

    print("[publisher] mensagem enviada para a fila:", QUEUE_NAME)
    print(json.dumps(message, indent=2))

    connection.close()


if __name__ == "__main__":
    main()
