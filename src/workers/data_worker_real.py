import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.contracts.messages import (  # noqa: E402
    DataBuildMessage,
    DatasetCounts,
    TrainRunEvent,
)
from src.messaging.rabbitmq import (  # noqa: E402
    create_connection,
    declare_queues,
    parse_json_body,
    publish_json,
)


DATA_BUILD_QUEUE = "q.data.build"
TRAIN_RUN_QUEUE = "q.train.run"

CLASSES = ["RBC", "WBC", "Platelets"]

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


def make_dataset_version() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_id = uuid4().hex[:6]
    return f"ds-{timestamp}-{short_id}"


def resolve_project_path(path_as_string: str) -> Path:
    """
    Converts a project-relative path into an absolute path.
    """
    path = Path(path_as_string)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def count_images_in_split(dataset_dir: Path, split: str) -> int:
    """
    Counts images in a YOLO-style split directory.

    Expected structure:
        dataset/images/train
        dataset/images/val
        dataset/images/test
    """
    images_dir = dataset_dir / "images" / split

    if not images_dir.exists():
        return 0

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    total = 0

    for extension in extensions:
        total += len(list(images_dir.glob(extension)))

    return total


def build_dataset(data_build_message: DataBuildMessage) -> tuple[str, Path, DatasetCounts]:
    """
    Runs prep_data.py to create a versioned dataset.
    """
    dataset_version = make_dataset_version()
    dataset_dir = ROOT_DIR / "storage" / "datasets" / dataset_version

    raw_dir = resolve_project_path(data_build_message.raw_uri)

    print("[Data Worker Real] Building dataset")
    print(f"  raw_dir={raw_dir}")
    print(f"  dataset_dir={dataset_dir}")

    command = [
        sys.executable,
        str(ROOT_DIR / "src" / "prep_data.py"),
        "--raw",
        str(raw_dir),
        "--out",
        str(dataset_dir),
        "--val",
        str(data_build_message.params.val_frac),
        "--test",
        str(data_build_message.params.test_frac),
        "--seed",
        str(data_build_message.params.seed),
    ]

    subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=True,
    )

    counts = DatasetCounts(
        train=count_images_in_split(dataset_dir, "train"),
        val=count_images_in_split(dataset_dir, "val"),
        test=count_images_in_split(dataset_dir, "test"),
    )

    return dataset_version, dataset_dir, counts


def build_train_event(
    data_build_message: DataBuildMessage,
    dataset_version: str,
    dataset_dir: Path,
    counts: DatasetCounts,
) -> TrainRunEvent:
    dataset_uri = dataset_dir.relative_to(ROOT_DIR).as_posix()

    return TrainRunEvent(
        run_request_id=f"train-{uuid4().hex[:8]}",
        dataset_version=dataset_version,
        dataset_uri=dataset_uri,
        classes=CLASSES,
        counts=counts,
        added_this_cycle=0,
        created_at=utc_now(),
        source_event=data_build_message.model_dump(mode="json"),
    )


def on_message(channel, method, properties, body) -> None:
    print("\n[Data Worker Real] Received data build request")

    try:
        raw_message = parse_json_body(body)
        data_build_message = DataBuildMessage.model_validate(raw_message)

        print("[Data Worker Real] Validated input event:")
        print(data_build_message.model_dump(mode="json"))

        dataset_version, dataset_dir, counts = build_dataset(data_build_message)

        train_event = build_train_event(
            data_build_message=data_build_message,
            dataset_version=dataset_version,
            dataset_dir=dataset_dir,
            counts=counts,
        )

        publish_json(
            channel=channel,
            queue=TRAIN_RUN_QUEUE,
            message=train_event.model_dump(mode="json"),
        )

        print(f"[Data Worker Real] Dataset version created: {dataset_version}")
        print(f"[Data Worker Real] Counts: {counts.model_dump(mode='json')}")
        print(f"[Data Worker Real] Published train event to {TRAIN_RUN_QUEUE}")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Data Worker Real] Message acknowledged")

    except ValidationError as error:
        print("[Data Worker Real] Invalid data.build message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except subprocess.CalledProcessError as error:
        print("[Data Worker Real] prep_data.py failed")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Data Worker Real] Error while processing message: {error}")
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

    print(f"[Data Worker Real] Waiting for messages from {DATA_BUILD_QUEUE}")
    print("[Data Worker Real] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Data Worker Real] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()