import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.contracts.messages import (  # noqa: E402
    Detection,
    InferRequestEvent,
    InferResultEvent,
)
from src.infer import Detector  # noqa: E402
from src.messaging.rabbitmq import (  # noqa: E402
    create_connection,
    declare_queues,
    parse_json_body,
    publish_json,
)


INFER_REQUEST_QUEUE = "q.infer.request"
INFER_RESULT_QUEUE = "q.infer.result"

DEFAULT_MODEL_VERSION = "v0"
DEFAULT_MODEL_URI = "models/v0"

PRODUCTION_POINTER_FILE = ROOT_DIR / "storage" / "models" / "production.json"

CONF_THRESHOLD = 0.25

RESULTS_DIR = ROOT_DIR / "storage" / "inference_results"
RESULTS_FILE = RESULTS_DIR / "results.jsonl"

QUEUES = [
    INFER_REQUEST_QUEUE,
    INFER_RESULT_QUEUE,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(path_as_string: str) -> Path:
    path = Path(path_as_string)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def resolve_model_path(model_uri: str) -> Path:
    model_dir = resolve_project_path(model_uri)
    model_path = model_dir / "best.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    return model_path


def read_production_pointer() -> dict:
    """
    Reads the current production model pointer.

    If no promoted model exists yet, the worker falls back to the baseline v0 model.
    """
    if not PRODUCTION_POINTER_FILE.exists():
        return {
            "model_version": DEFAULT_MODEL_VERSION,
            "model_uri": DEFAULT_MODEL_URI,
        }

    pointer = json.loads(PRODUCTION_POINTER_FILE.read_text(encoding="utf-8"))

    if "model_version" not in pointer or "model_uri" not in pointer:
        raise ValueError(f"Invalid production pointer: {PRODUCTION_POINTER_FILE}")

    return pointer


class ActiveModelState:
    """
    Keeps the active model loaded in memory.

    The worker checks production.json before inference. If the production model
    changed, it reloads the ONNX model automatically.
    """

    def __init__(self) -> None:
        self.model_version: str | None = None
        self.model_uri: str | None = None
        self.model_path: Path | None = None
        self.detector: Detector | None = None

    def load_model(self, model_version: str, model_uri: str) -> None:
        model_path = resolve_model_path(model_uri)

        print("[Inference Worker Real] Loading model")
        print(f"  model_version={model_version}")
        print(f"  model_uri={model_uri}")
        print(f"  model_path={model_path}")

        self.detector = Detector(str(model_path), conf=CONF_THRESHOLD)
        self.model_version = model_version
        self.model_uri = model_uri
        self.model_path = model_path

        print("[Inference Worker Real] Model loaded successfully")

    def ensure_latest_model(self) -> None:
        pointer = read_production_pointer()

        model_version = pointer["model_version"]
        model_uri = pointer["model_uri"]

        if self.detector is None:
            self.load_model(model_version=model_version, model_uri=model_uri)
            return

        if model_version != self.model_version or model_uri != self.model_uri:
            print("[Inference Worker Real] Production model changed. Reloading...")
            self.load_model(model_version=model_version, model_uri=model_uri)
            return

        print("[Inference Worker Real] Active model is already up to date")


MODEL_STATE = ActiveModelState()


def normalize_detection(raw_detection: dict) -> Detection:
    """
    Converts the dictionary returned by src/infer.py into a Pydantic Detection.
    """
    return Detection(
        cls=str(raw_detection["cls"]),
        conf=float(raw_detection["conf"]),
        box=[float(value) for value in raw_detection["box"]],
    )


def save_inference_result(infer_result_event: InferResultEvent) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(infer_result_event.model_dump(mode="json")) + "\n")


def build_infer_result_event(
    infer_request_event: InferRequestEvent,
    detections: list[Detection],
    latency_ms: float,
) -> InferResultEvent:
    if MODEL_STATE.model_version is None:
        raise RuntimeError("No active model version is loaded")

    min_conf = min((detection.conf for detection in detections), default=0.0)

    return InferResultEvent(
        inference_id=f"inf-{uuid4().hex[:8]}",
        model_version=MODEL_STATE.model_version,
        status="success",
        image_uri=infer_request_event.image_uri,
        latency_ms=latency_ms,
        min_conf=min_conf,
        ts=utc_now(),
        detections=detections,
        source_event=infer_request_event.model_dump(mode="json"),
    )


def run_inference(infer_request_event: InferRequestEvent) -> InferResultEvent:
    MODEL_STATE.ensure_latest_model()

    if MODEL_STATE.detector is None:
        raise RuntimeError("No detector loaded")

    image_path = resolve_project_path(infer_request_event.image_uri)

    print("[Inference Worker Real] Running inference")
    print(f"  image_uri={infer_request_event.image_uri}")
    print(f"  image_path={image_path}")
    print(f"  active_model_version={MODEL_STATE.model_version}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    start = time.perf_counter()
    raw_detections = MODEL_STATE.detector(image)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    detections = [
        normalize_detection(raw_detection)
        for raw_detection in raw_detections
    ]

    return build_infer_result_event(
        infer_request_event=infer_request_event,
        detections=detections,
        latency_ms=latency_ms,
    )


def on_infer_request(channel, method, properties, body) -> None:
    print("\n[Inference Worker Real] Received inference request")

    try:
        raw_message = parse_json_body(body)
        infer_request_event = InferRequestEvent.model_validate(raw_message)

        print("[Inference Worker Real] Validated inference request:")
        print(infer_request_event.model_dump(mode="json"))

        infer_result_event = run_inference(infer_request_event)

        publish_json(
            channel=channel,
            queue=INFER_RESULT_QUEUE,
            message=infer_result_event.model_dump(mode="json"),
        )

        save_inference_result(infer_result_event)

        print(f"[Inference Worker Real] Published inference result to {INFER_RESULT_QUEUE}")
        print(f"[Inference Worker Real] Saved result to {RESULTS_FILE}")
        print(f"[Inference Worker Real] detections={len(infer_result_event.detections)}")
        print(f"[Inference Worker Real] min_conf={infer_result_event.min_conf}")

        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("[Inference Worker Real] Inference request acknowledged")

    except ValidationError as error:
        print("[Inference Worker Real] Invalid infer.request message")
        print(error)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as error:
        print(f"[Inference Worker Real] Error while processing inference: {error}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    MODEL_STATE.ensure_latest_model()

    connection = create_connection()
    channel = connection.channel()

    declare_queues(channel, QUEUES)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=INFER_REQUEST_QUEUE,
        on_message_callback=on_infer_request,
        auto_ack=False,
    )

    print(f"[Inference Worker Real] Waiting for messages from {INFER_REQUEST_QUEUE}")
    print("[Inference Worker Real] Press CTRL+C to stop")
    print("[Inference Worker Real] This worker does not consume q.model.promoted directly")
    print(f"[Inference Worker Real] It checks {PRODUCTION_POINTER_FILE} before inference")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Inference Worker Real] Stopping...")
    finally:
        connection.close()


if __name__ == "__main__":
    main()