import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.contracts.messages import LabelTaskEvent  # noqa: E402
from src.messaging.rabbitmq import (  # noqa: E402
    create_connection,
    declare_queues,
    parse_json_body,
    publish_json,
)


LABEL_TASK_QUEUE = "q.label.task"
DATA_BUILD_QUEUE = "q.data.build"

RAW_IMAGES_DIR = ROOT_DIR / "data" / "raw" / "images"
RAW_LABELS_DIR = ROOT_DIR / "data" / "raw" / "labels"

ORACLE_LABELS_DIR = ROOT_DIR / "data" / "oracle" / "labels"

ANNOTATION_LOG_FILE = ROOT_DIR / "storage" / "oracle_annotations" / "annotations.jsonl"

QUEUES = [
    LABEL_TASK_QUEUE,
    DATA_BUILD_QUEUE,
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(path_as_string: str) -> Path:
    path = Path(path_as_string)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def append_jsonl(file_path: Path, record: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_oracle_label_for_image(image_path: Path) -> Path:
    """
    Finds the hidden/oracle label associated with a stream image.

    Example:
        data/stream/images/BloodImage_00000.jpg
        data/oracle/labels/BloodImage_00000.txt
    """
    stem = image_path.stem

    candidate_label_paths = [
        ORACLE_LABELS_DIR / f"{stem}.txt",
        ROOT_DIR / "data" / "raw" / "labels" / f"{stem}.txt",
        ROOT_DIR / "data" / "labels" / f"{stem}.txt",
        ROOT_DIR / "storage" / "datasets" / "labels" / f"{stem}.txt",
    ]

    for candidate_path in candidate_label_paths:
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Could not find oracle label for image "
        f"{image_path}. Tried: {[str(path) for path in candidate_label_paths]}"
    )


def copy_annotated_sample_to_raw(label_task_event: LabelTaskEvent) -> tuple[Path, Path]:
    """
    Copies the selected stream image and its hidden/oracle label back to data/raw.

    This simulates the result of a human annotation step.
    """
    source_image_path = resolve_project_path(label_task_event.image_uri)

    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_image_path}")

    oracle_label_path = find_oracle_label_for_image(source_image_path)

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    target_stem = f"oracle_{label_task_event.label_task_id}_{source_image_path.stem}"

    target_image_path = RAW_IMAGES_DIR / f"{target_stem}{source_image_path.suffix}"
    target_label_path = RAW_LABELS_DIR / f"{target_stem}.txt"

    shutil.copy2(source_image_path, target_image_path)
    shutil.copy2(oracle_label_path, target_label_path)

    return target_image_path, target_label_path


def build_annotation_record(
    label_task_event: LabelTaskEvent,
    target_image_path: Path,
    target_label_path: Path,
) -> dict:
    return {
        "event": "oracle.annotation.completed",
        "label_task_id": label_task_event.label_task_id,
        "inference_id": label_task_event.inference_id,
        "source_image_uri": label_task_event.image_uri,
        "target_image_uri": target_image_path.relative_to(ROOT_DIR).as_posix(),
        "target_label_uri": target_label_path.relative_to(ROOT_DIR).as_posix(),
        "model_version": label_task_event.model_version,
        "min_conf": label_task_event.min_conf,
        "threshold": label_task_event.threshold,
        "created_at": utc_now(),
        "source_event": label_task_event.model_dump(mode="json"),
    }


def build_feedback_data_build_message() -> dict:
    """
    Builds a data.build event triggered by the feedback loop.

    This is published only after the oracle annotation worker has actually
    injected a new image + label pair into data/raw.
    """
    return {
        "event": "data.build",
        "trigger": "feedback",
        "raw_uri": "data/raw",
        "params": {
            "val_frac": 0.15,
            "test_frac": 0.15,
            "seed": 42,
        },
    }


def on_message(channel, method, properties, body) -> None:
    print("\n[Oracle Annotation Worker] Received label task")

    try:
        raw_message = parse_json_body(body)
        label_task_event = LabelTaskEvent.model_validate(raw_message)

        print("[Oracle Annotation Worker] Validated input event:")
        print(label_task_event.model_dump(mode="json"))

        target_image_path, target_label_path = copy_annotated_sample_to_raw(
            label_task_event
        )

        annotation_record = build_annotation_record(
            label_task_event=label_task_event,
            target_image_path=target_image_path,
            target_label_path=target_label_path,
        )

        append_jsonl(
            file_path=ANNOTATION_LOG_FILE,
            record=annotation_record,
        )

        print("[Oracle Annotation Worker] Oracle annotation completed")
        print(f"  target_image={target_image_path}")
        print(f"  target_label={target_label_path}")
        print(f"  annotation_log={ANNOTATION_LOG_FILE}")

        data_build_message = build_feedback_data_build_message()

        publish_json(
            channel=channel,
            queue=DATA_BUILD_QUEUE,
            message=data_build_message,
        )

        print(
            "[Oracle Annotation Worker] Published feedback data build event "
            f"to {DATA_BUILD_QUEUE}"
        )

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Oracle Annotation Worker] Message acknowledged")

    except ValidationError as error:
        print("[Oracle Annotation Worker] Invalid label.task message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Oracle Annotation Worker] Error while processing message: {error}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    connection = create_connection()
    channel = connection.channel()

    declare_queues(channel, QUEUES)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=LABEL_TASK_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )

    print(f"[Oracle Annotation Worker] Waiting for messages from {LABEL_TASK_QUEUE}")
    print("[Oracle Annotation Worker] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Oracle Annotation Worker] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()