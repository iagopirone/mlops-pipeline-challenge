import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pika


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.messaging.rabbitmq import (  # noqa: E402
    create_connection,
    declare_queues,
    parse_json_body,
    publish_json,
)


DATA_BUILD_QUEUE = "q.data.build"
TRAIN_RUN_QUEUE = "q.train.run"

QUEUES = [
    DATA_BUILD_QUEUE,
    TRAIN_RUN_QUEUE,
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_fake_train_event(source_event: dict) -> dict:
    """
    Builds a fake train.run event.

    This simulates the output that the real Data Worker will produce later
    after creating a versioned dataset.
    """
    return {
        "event": "train.run",
        "run_request_id": f"train-{uuid4().hex[:8]}",
        "dataset_version": "ds-001",
        "dataset_uri": "storage/datasets/ds-001",
        "classes": ["RBC", "WBC", "Platelets"],
        "counts": {
            "train": 0,
            "val": 0,
            "test": 0,
        },
        "added_this_cycle": 0,
        "created_at": utc_now(),
        "source_event": source_event,
    }


def on_message(channel, method, properties, body) -> None:
    print("\n[Data Worker Fake] Received data build request")

    try:
        message = parse_json_body(body)

        print("[Data Worker Fake] Input event:")
        print(message)

        train_event = build_fake_train_event(message)

        publish_json(
            channel=channel,
            queue=TRAIN_RUN_QUEUE,
            message=train_event,
        )

        print(f"[Data Worker Fake] Published fake train event to {TRAIN_RUN_QUEUE}")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Data Worker Fake] Message acknowledged")

    except Exception as error:
        print(f"[Data Worker Fake] Error while processing message: {error}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    connection = create_connection()
    channel = connection.channel()

    declare_queues(channel, QUEUES)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=DATA_BUILD_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )

    print(f"[Data Worker Fake] Waiting for messages from {DATA_BUILD_QUEUE}")
    print("[Data Worker Fake] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Data Worker Fake] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()