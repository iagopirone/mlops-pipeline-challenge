import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.contracts.messages import (  # noqa: E402
    ModelMetrics,
    ModelPromotedEvent,
    TrainRunEvent,
)
from src.messaging.rabbitmq import (  # noqa: E402
    create_connection,
    declare_queues,
    parse_json_body,
    publish_json,
)


TRAIN_RUN_QUEUE = "q.train.run"
MODEL_PROMOTED_QUEUE = "q.model.promoted"

BASE_MODEL = "models/v0/best.pt"
BASELINE_MAP50 = 0.50

QUEUES = [
    TRAIN_RUN_QUEUE,
    MODEL_PROMOTED_QUEUE,
    "q.data.build",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_fake_model_promoted_event(
    train_run_event: TrainRunEvent,
) -> ModelPromotedEvent | None:
    """
    Builds a fake model.promoted event.

    This simulates the output of the real Train Worker after training,
    evaluating, applying the quality gate and promoting the model.
    """
    metrics = ModelMetrics(
        mAP50=0.51,
        per_class={
            "RBC": 0.52,
            "WBC": 0.50,
            "Platelets": 0.51,
        },
    )

    if metrics.mAP50 < BASELINE_MAP50:
        return None

    model_version = f"model-{uuid4().hex[:8]}"

    return ModelPromotedEvent(
        model_version=model_version,
        base_model=BASE_MODEL,
        model_uri=f"storage/models/{model_version}",
        dataset_version=train_run_event.dataset_version,
        metrics=metrics,
        baseline=BASELINE_MAP50,
        promoted=True,
        created_at=utc_now(),
        source_event=train_run_event.model_dump(mode="json"),
    )


def on_message(channel, method, properties, body) -> None:
    print("\n[Train Worker Fake] Received train request")

    try:
        raw_message = parse_json_body(body)
        train_run_event = TrainRunEvent.model_validate(raw_message)

        print("[Train Worker Fake] Validated input event:")
        print(train_run_event.model_dump(mode="json"))

        model_promoted_event = build_fake_model_promoted_event(train_run_event)

        if model_promoted_event is None:
            print("[Train Worker Fake] Fake model did not pass the quality gate")
            print("[Train Worker Fake] No model.promoted event was published")

            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        publish_json(
            channel=channel,
            queue=MODEL_PROMOTED_QUEUE,
            message=model_promoted_event.model_dump(mode="json"),
        )

        print("[Train Worker Fake] Fake model passed the quality gate")
        print(f"[Train Worker Fake] Published model event to {MODEL_PROMOTED_QUEUE}")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Train Worker Fake] Message acknowledged")

    except ValidationError as error:
        print("[Train Worker Fake] Invalid train.run message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Train Worker Fake] Error while processing message: {error}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    connection = create_connection()
    channel = connection.channel()

    declare_queues(channel, QUEUES)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=TRAIN_RUN_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )

    print(f"[Train Worker Fake] Waiting for messages from {TRAIN_RUN_QUEUE}")
    print("[Train Worker Fake] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Train Worker Fake] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()