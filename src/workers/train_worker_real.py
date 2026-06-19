import re
import shutil
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

EPOCHS = 1
IMGSZ = 640
BATCH = 8

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


def make_model_version() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_id = uuid4().hex[:6]
    return f"model-{timestamp}-{short_id}"


def resolve_project_path(path_as_string: str) -> Path:
    path = Path(path_as_string)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def parse_map50(train_stdout: str) -> float:
    """
    Extracts the TEST mAP50 printed by src/train.py.

    Expected line:
        TEST mAP50=0.1234  mAP50-95=0.0567
    """
    match = re.search(r"TEST\s+mAP50=([0-9.]+)", train_stdout)

    if match is None:
        raise ValueError("Could not find TEST mAP50 in train.py output")

    return float(match.group(1))


def run_training(
    train_run_event: TrainRunEvent,
    model_version: str,
) -> tuple[float, Path]:
    """
    Runs src/train.py using the dataset received from q.train.run.
    """
    dataset_dir = resolve_project_path(train_run_event.dataset_uri)
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset data.yaml not found: {data_yaml}")

    print("[Train Worker Real] Starting training")
    print(f"  dataset_version={train_run_event.dataset_version}")
    print(f"  dataset_uri={train_run_event.dataset_uri}")
    print(f"  data_yaml={data_yaml}")
    print(f"  base_model={BASE_MODEL}")
    print(f"  epochs={EPOCHS}")

    command = [
        sys.executable,
        str(ROOT_DIR / "src" / "train.py"),
        "--data",
        str(data_yaml),
        "--base",
        str(ROOT_DIR / BASE_MODEL),
        "--epochs",
        str(EPOCHS),
        "--imgsz",
        str(IMGSZ),
        "--batch",
        str(BATCH),
        "--name",
        model_version,
    ]

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
)

    print("[Train Worker Real] train.py stdout:")
    print(result.stdout)

    if result.stderr:
        print("[Train Worker Real] train.py stderr:")
        print(result.stderr)

    map50 = parse_map50(result.stdout)

    run_dir = ROOT_DIR / "runs" / model_version
    weights_dir = run_dir / "weights"

    return map50, weights_dir


def register_model_artifacts(
    model_version: str,
    weights_dir: Path,
) -> Path:
    """
    Copies model artifacts from runs/<model_version>/weights
    to storage/models/<model_version>.
    """
    best_pt = weights_dir / "best.pt"
    best_onnx = weights_dir / "best.onnx"

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    model_dir = ROOT_DIR / "storage" / "models" / model_version
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_pt, model_dir / "best.pt")

    if best_onnx.exists():
        shutil.copy2(best_onnx, model_dir / "best.onnx")
    else:
        print(f"[Train Worker Real] Warning: best.onnx not found at {best_onnx}")

    return model_dir


def build_model_promoted_event(
    train_run_event: TrainRunEvent,
    model_version: str,
    model_dir: Path,
    map50: float,
) -> ModelPromotedEvent:
    model_uri = model_dir.relative_to(ROOT_DIR).as_posix()

    return ModelPromotedEvent(
        model_version=model_version,
        base_model=BASE_MODEL,
        model_uri=model_uri,
        dataset_version=train_run_event.dataset_version,
        metrics=ModelMetrics(
            mAP50=map50,
            per_class={},
        ),
        baseline=BASELINE_MAP50,
        promoted=True,
        created_at=utc_now(),
        source_event=train_run_event.model_dump(mode="json"),
    )


def on_message(channel, method, properties, body) -> None:
    print("\n[Train Worker Real] Received train request")

    try:
        raw_message = parse_json_body(body)
        train_run_event = TrainRunEvent.model_validate(raw_message)

        print("[Train Worker Real] Validated input event:")
        print(train_run_event.model_dump(mode="json"))

        model_version = make_model_version()

        map50, weights_dir = run_training(
            train_run_event=train_run_event,
            model_version=model_version,
        )

        print(f"[Train Worker Real] TEST mAP50={map50:.4f}")
        print(f"[Train Worker Real] Baseline mAP50={BASELINE_MAP50:.4f}")

        if map50 < BASELINE_MAP50:
            print("[Train Worker Real] Model did not pass the quality gate")
            print("[Train Worker Real] No model.promoted event was published")

            channel.basic_ack(delivery_tag=method.delivery_tag)
            print("[Train Worker Real] Message acknowledged")
            return

        model_dir = register_model_artifacts(
            model_version=model_version,
            weights_dir=weights_dir,
        )

        model_promoted_event = build_model_promoted_event(
            train_run_event=train_run_event,
            model_version=model_version,
            model_dir=model_dir,
            map50=map50,
        )

        publish_json(
            channel=channel,
            queue=MODEL_PROMOTED_QUEUE,
            message=model_promoted_event.model_dump(mode="json"),
        )

        print("[Train Worker Real] Model passed the quality gate")
        print(f"[Train Worker Real] Model registered at {model_dir}")
        print(f"[Train Worker Real] Published model event to {MODEL_PROMOTED_QUEUE}")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Train Worker Real] Message acknowledged")

    except ValidationError as error:
        print("[Train Worker Real] Invalid train.run message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except subprocess.CalledProcessError as error:
        print("[Train Worker Real] train.py failed")
        print(error)
        print(error.stdout)
        print(error.stderr)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Train Worker Real] Error while processing message: {error}")
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

    print(f"[Train Worker Real] Waiting for messages from {TRAIN_RUN_QUEUE}")
    print("[Train Worker Real] Press CTRL+C to stop")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Train Worker Real] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()