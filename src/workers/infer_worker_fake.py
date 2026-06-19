import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.contracts.messages import (  # noqa: E402
    Detection,
    InferRequestEvent,
    InferResultEvent,
)
from src.messaging.rabbitmq import (  # noqa: E402
    create_connection,
    declare_queues,
    parse_json_body,
    publish_json,
)


INFER_REQUEST_QUEUE = "q.infer.request"
INFER_RESULT_QUEUE = "q.infer.result"

QUEUES = [
    INFER_REQUEST_QUEUE,
    INFER_RESULT_QUEUE,
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.label.task",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_fake_detections() -> list[Detection]:
    """
    Builds fake detections to simulate the output of an object detection model.
    """
    rbc_conf = round(random.uniform(0.35, 0.95), 2)
    wbc_conf = round(random.uniform(0.45, 0.98), 2)

    return [
        Detection(
            cls="RBC",
            conf=rbc_conf,
            box=[10.0, 20.0, 80.0, 90.0],
        ),
        Detection(
            cls="WBC",
            conf=wbc_conf,
            box=[100.0, 120.0, 180.0, 210.0],
        ),
    ]


def build_fake_infer_result_event(
    infer_request_event: InferRequestEvent,
) -> InferResultEvent:
    """
    Builds a fake infer.result event.

    This simulates an inference service returning detections for an image.
    """
    detections = build_fake_detections()
    min_conf = min(detection.conf for detection in detections)

    return InferResultEvent(
        inference_id=f"inf-{uuid4().hex[:8]}",
        model_version=infer_request_event.model_version,
        status="success",
        image_uri=infer_request_event.image_uri,
        latency_ms=round(random.uniform(20.0, 80.0), 2),
        min_conf=min_conf,
        ts=utc_now(),
        detections=detections,
        source_event=infer_request_event.model_dump(mode="json"),
    )


def on_message(channel, method, properties, body) -> None:
    print("\n[Inference Worker Fake] Received inference request")

    try:
        raw_message = parse_json_body(body)
        infer_request_event = InferRequestEvent.model_validate(raw_message)

        print("[Inference Worker Fake] Validated input event:")
        print(infer_request_event.model_dump(mode="json"))

        infer_result_event = build_fake_infer_result_event(infer_request_event)

        publish_json(
            channel=channel,
            queue=INFER_RESULT_QUEUE,
            message=infer_result_event.model_dump(mode="json"),
        )

        print(f"[Inference Worker Fake] Published inference result to {INFER_RESULT_QUEUE}")
        print(f"[Inference Worker Fake] min_conf={infer_result_event.min_conf}")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Inference Worker Fake] Message acknowledged")

    except ValidationError as error:
        print("[Inference Worker Fake] Invalid infer.request message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Inference Worker Fake] Error while processing message: {error}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    connection = create_connection()
    channel = connection.channel()

    declare_queues(channel, QUEUES)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=INFER_REQUEST_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )

    print(f"[Inference Worker Fake] Waiting for messages from {INFER_REQUEST_QUEUE}")
    print("[Inference Worker Fake] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Inference Worker Fake] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()