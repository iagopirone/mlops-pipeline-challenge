import json
import shutil
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
CANDIDATE_IMAGES_DIR = CANDIDATES_DIR / "images"
CANDIDATES_FILE = CANDIDATES_DIR / "candidates.jsonl"

QUEUES = [
    INFER_RESULT_QUEUE,
    LABEL_TASK_QUEUE,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(path_as_string: str) -> Path:
    path = Path(path_as_string)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def copy_candidate_image(
    image_uri: str,
    label_task_id: str,
) -> str | None:
    """
    Copies the image selected for annotation into storage/label_candidates/images.

    This makes the candidate easier to inspect later.
    """
    source_path = resolve_project_path(image_uri)

    if not source_path.exists():
        print(f"[Collect Worker Real] Warning: image not found: {source_path}")
        return None

    CANDIDATE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    target_name = f"{label_task_id}_{source_path.name}"
    target_path = CANDIDATE_IMAGES_DIR / target_name

    shutil.copy2(source_path, target_path)

    return target_path.relative_to(ROOT_DIR).as_posix()


def build_label_task_event(
    infer_result_event: InferResultEvent,
) -> LabelTaskEvent:
    """
    Builds a label.task event for a low-confidence inference result.
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


def save_candidate_record(
    label_task_event: LabelTaskEvent,
    candidate_image_uri: str | None,
) -> None:
    """
    Persists the candidate metadata locally.

    This simulates the collection store used before human annotation.
    """
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "label_task": label_task_event.model_dump(mode="json"),
        "candidate_image_uri": candidate_image_uri,
        "stored_at": utc_now(),
    }

    with CANDIDATES_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def should_create_label_task(infer_result_event: InferResultEvent) -> bool:
    return infer_result_event.min_conf < LOW_CONF_THRESHOLD


def on_message(channel, method, properties, body) -> None:
    print("\n[Collect Worker Real] Received inference result")

    try:
        raw_message = parse_json_body(body)
        infer_result_event = InferResultEvent.model_validate(raw_message)

        print("[Collect Worker Real] Validated inference result:")
        print(infer_result_event.model_dump(mode="json"))

        print(f"[Collect Worker Real] min_conf={infer_result_event.min_conf}")
        print(f"[Collect Worker Real] threshold={LOW_CONF_THRESHOLD}")

        if not should_create_label_task(infer_result_event):
            print("[Collect Worker Real] Confidence is acceptable")
            print("[Collect Worker Real] No label task was created")

            channel.basic_ack(delivery_tag=method.delivery_tag)
            print("[Collect Worker Real] Message acknowledged")
            return

        label_task_event = build_label_task_event(infer_result_event)

        candidate_image_uri = copy_candidate_image(
            image_uri=infer_result_event.image_uri,
            label_task_id=label_task_event.label_task_id,
        )

        save_candidate_record(
            label_task_event=label_task_event,
            candidate_image_uri=candidate_image_uri,
        )

        publish_json(
            channel=channel,
            queue=LABEL_TASK_QUEUE,
            message=label_task_event.model_dump(mode="json"),
        )

        print(f"[Collect Worker Real] Published label task to {LABEL_TASK_QUEUE}")
        print(f"[Collect Worker Real] Saved candidate metadata to {CANDIDATES_FILE}")

        if candidate_image_uri is not None:
            print(f"[Collect Worker Real] Copied candidate image to {candidate_image_uri}")

        print("[Collect Worker Real] Dataset rebuild was NOT triggered automatically")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Collect Worker Real] Message acknowledged")

    except ValidationError as error:
        print("[Collect Worker Real] Invalid infer.result message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Collect Worker Real] Error while processing inference result: {error}")
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

    print(f"[Collect Worker Real] Waiting for messages from {INFER_RESULT_QUEUE}")
    print("[Collect Worker Real] Press CTRL+C to stop")
    print("[Collect Worker Real] This worker does not trigger q.data.build automatically")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Collect Worker Real] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()