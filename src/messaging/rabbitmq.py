import json
from typing import Any

import pika


RABBITMQ_HOST = "localhost"


def create_connection(host: str = RABBITMQ_HOST) -> pika.BlockingConnection:
    """
    Creates a connection with RabbitMQ.

    For now, RabbitMQ runs locally through Docker Compose.
    """
    return pika.BlockingConnection(
        pika.ConnectionParameters(host=host)
    )


def declare_queues(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queues: list[str],
) -> None:
    """
    Declares all queues used by a worker.

    If the queue does not exist, RabbitMQ creates it.
    If it already exists, RabbitMQ just reuses it.
    """
    for queue in queues:
        channel.queue_declare(queue=queue, durable=True)


def publish_json(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue: str,
    message: dict[str, Any],
) -> None:
    """
    Publishes a Python dictionary as a persistent JSON message.
    """
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(message).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        ),
    )


def parse_json_body(body: bytes) -> dict[str, Any]:
    """
    Converts a RabbitMQ message body from bytes to a Python dictionary.
    """
    message_as_string = body.decode("utf-8")
    return json.loads(message_as_string)