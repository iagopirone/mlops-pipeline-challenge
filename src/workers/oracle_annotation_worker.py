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
)


LABEL_TASK_QUEUE = "q.label.task"

RAW_IMAGES_DIR = ROOT_DIR / "data" / "raw" / "images"
RAW_LABELS_DIR = ROOT_DIR / "data" / "raw" / "labels"

ANNOTATION_RECORDS_DIR = ROOT_DIR / "storage" / "oracle_annotations"
ANNOTATION_RECORDS_FILE = ANNOTATION_RECORDS_DIR / "annotations.jsonl"

QUEUES = [
    LABEL_TASK_QUEUE,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(path_as_string: str) -> Path:
    path = Path(path_as_string)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def candidate_label_paths(image_path: Path) -> list[Path]:
    """
    Builds possible oracle label paths for an image.

    In this MVP, the digital annotation is simulated by recovering an existing
    YOLO label file with the same image stem.
    """
    stem = image_path.stem

    candidates: list[Path] = []

    if image_path.parent.name == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{stem}.txt")

    candidates.extend(
        [
            ROOT_DIR / "data" / "stream" / "labels" / f"{stem}.txt",
            ROOT_DIR / "data" / "oracle" / "labels" / f"{stem}.txt",
            ROOT_DIR / "data" / "raw" / "labels" / f"{stem}.txt",
            ROOT_DIR / "data" / "labels" / f"{stem}.txt",
            ROOT_DIR / "dataset" / "labels" / "train" / f"{stem}.txt",
            ROOT_DIR / "dataset" / "labels" / "val" / f"{stem}.txt",
            ROOT_DIR / "dataset" / "labels" / "test" / f"{stem}.txt",
        ]
    )

    unique_candidates = []
    seen = set()

    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    return unique_candidates


def find_oracle_label(image_path: Path) -> Path:
    """
    Finds the existing label file used as a digital oracle.
    """
    for label_path in candidate_label_paths(image_path):
        if label_path.exists():
            return label_path

    searched_paths = "\n".join(str(path) for path in candidate_label_paths(image_path))
    raise FileNotFoundError(
        "Could not find oracle label for image.\n"
        f"image_path={image_path}\n"
        f"searched_paths=\n{searched_paths}"
    )


def inject_annotated_sample(label_task_event: LabelTaskEvent) -> tuple[str, str]:
    """
    Copies the selected image and its oracle label into data/raw.

    The generated files are prefixed with oracle_<label_task_id>_ to avoid
    overwriting existing files and to make the source clear.
    """
    image_path = resolve_project_path(label_task_event.image_uri)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_path = find_oracle_label(image_path)

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    target_image_name = f"oracle_{label_task_event.label_task_id}_{image_path.name}"
    target_image_path = RAW_IMAGES_DIR / target_image_name
    target_label_path = RAW_LABELS_DIR / f"{target_image_path.stem}.txt"

    shutil.copy2(image_path, target_image_path)
    shutil.copy2(label_path, target_label_path)

    target_image_uri = target_image_path.relative_to(ROOT_DIR).as_posix()
    target_label_uri = target_label_path.relative_to(ROOT_DIR).as_posix()

    return target_image_uri, target_label_uri


def save_annotation_record(
    label_task_event: LabelTaskEvent,
    target_image_uri: str,
    target_label_uri: str,
) -> None:
    """
    Saves a traceability record for the oracle annotation.
    """
    ANNOTATION_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "event": "oracle.annotation.completed",
        "label_task_id": label_task_event.label_task_id,
        "inference_id": label_task_event.inference_id,
        "source_image_uri": label_task_event.image_uri,
        "target_image_uri": target_image_uri,
        "target_label_uri": target_label_uri,
        "model_version": label_task_event.model_version,
        "min_conf": label_task_event.min_conf,
        "threshold": label_task_event.threshold,
        "created_at": utc_now(),
    }

    with ANNOTATION_RECORDS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def on_message(channel, method, properties, body) -> None:
    print("\n[Oracle Annotation Worker] Received label task")

    try:
        raw_message = parse_json_body(body)
        label_task_event = LabelTaskEvent.model_validate(raw_message)

        print("[Oracle Annotation Worker] Validated label task:")
        print(label_task_event.model_dump(mode="json"))

        target_image_uri, target_label_uri = inject_annotated_sample(label_task_event)

        save_annotation_record(
            label_task_event=label_task_event,
            target_image_uri=target_image_uri,
            target_label_uri=target_label_uri,
        )

        print("[Oracle Annotation Worker] Digital annotation completed")
        print(f"[Oracle Annotation Worker] Injected image: {target_image_uri}")
        print(f"[Oracle Annotation Worker] Injected label: {target_label_uri}")
        print(f"[Oracle Annotation Worker] Record saved to {ANNOTATION_RECORDS_FILE}")
        print("[Oracle Annotation Worker] q.data.build was NOT triggered automatically")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Oracle Annotation Worker] Message acknowledged")

    except ValidationError as error:
        print("[Oracle Annotation Worker] Invalid label.task message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Oracle Annotation Worker] Error while processing label task: {error}")
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
    print("[Oracle Annotation Worker] This worker simulates digital annotation")
    print("[Oracle Annotation Worker] It does not trigger q.data.build automatically")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Oracle Annotation Worker] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()