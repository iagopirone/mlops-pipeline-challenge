import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.contracts.messages import (  # noqa: E402
    InferResultEvent,
    LabelTaskEvent,
)
from src.messaging.rabbitmq import (  # noqa: E402
    create_connection,
    declare_queues,
    parse_json_body,
    publish_json,
)


INFER_RESULT_QUEUE = "q.infer.result"
LABEL_TASK_QUEUE = "q.label.task"

LOW_CONF_THRESHOLD = 0.50

CANDIDATES_DIR = ROOT_DIR / "storage" / "label_candidates"
CANDIDATES_FILE = CANDIDATES_DIR / "candidates.jsonl"

QUEUES = [
    INFER_RESULT_QUEUE,
    LABEL_TASK_QUEUE,
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_candidate(label_task_event: LabelTaskEvent) -> None:
    """
    Persists a label candidate locally.

    This simulates storing examples that should be reviewed by an annotator.
    """
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    with CANDIDATES_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(label_task_event.model_dump(mode="json")) + "\n")


def build_label_task_event(
    infer_result_event: InferResultEvent,
) -> LabelTaskEvent:
    """
    Builds a label.task event for low-confidence inference results.
    """
    return LabelTaskEvent(
        label_task_id=f"label-{uuid4().hex[:8]}",
        reason="low_confidence",
        inference_id=infer_result_event.inference_id,
        image_uri=infer_result_event.image_uri,
        model_version=infer_result_event.model_version,
        min_conf=infer_result_event.min_conf,
        threshold=LOW_CONF_THRESHOLD,
        status="pending_annotation",
        created_at=utc_now(),
        source_event=infer_result_event.model_dump(mode="json"),
    )


def on_message(channel, method, properties, body) -> None:
    print("\n[Collect Worker Fake] Received inference result")

    try:
        raw_message = parse_json_body(body)
        infer_result_event = InferResultEvent.model_validate(raw_message)

        print("[Collect Worker Fake] Validated input event:")
        print(infer_result_event.model_dump(mode="json"))

        print(f"[Collect Worker Fake] min_conf={infer_result_event.min_conf}")
        print(f"[Collect Worker Fake] threshold={LOW_CONF_THRESHOLD}")

        if infer_result_event.min_conf >= LOW_CONF_THRESHOLD:
            print("[Collect Worker Fake] Confidence is acceptable")
            print("[Collect Worker Fake] No label task was created")

            channel.basic_ack(delivery_tag=method.delivery_tag)
            print("[Collect Worker Fake] Message acknowledged")
            return

        label_task_event = build_label_task_event(infer_result_event)

        publish_json(
            channel=channel,
            queue=LABEL_TASK_QUEUE,
            message=label_task_event.model_dump(mode="json"),
        )

        save_candidate(label_task_event)

        print(f"[Collect Worker Fake] Published label task to {LABEL_TASK_QUEUE}")
        print(f"[Collect Worker Fake] Saved candidate to {CANDIDATES_FILE}")
        print("[Collect Worker Fake] Dataset rebuild was NOT triggered automatically")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Collect Worker Fake] Message acknowledged")

    except ValidationError as error:
        print("[Collect Worker Fake] Invalid infer.result message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Collect Worker Fake] Error while processing message: {error}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    connection = create_connection()
    channel = connection.channel()

    declare_queues(channel, QUEUES)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=INFER_RESULT_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )

    print(f"[Collect Worker Fake] Waiting for messages from {INFER_RESULT_QUEUE}")
    print("[Collect Worker Fake] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Collect Worker Fake] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()