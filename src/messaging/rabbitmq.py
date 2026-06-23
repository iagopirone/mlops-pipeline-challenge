import json
import os
from typing import Any
import pika


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")


def create_connection(host: str | None = None) -> pika.BlockingConnection:
    """
    Creates a connection with RabbitMQ.

    Locally, the default host is localhost.
    Inside Docker Compose, RABBITMQ_HOST should be set to the RabbitMQ service name.
    """
    effective_host = host or RABBITMQ_HOST

    credentials = pika.PlainCredentials(
        username=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
    )

    parameters = pika.ConnectionParameters(
        host=effective_host,
        port=RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300,
    )

    return pika.BlockingConnection(parameters)


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