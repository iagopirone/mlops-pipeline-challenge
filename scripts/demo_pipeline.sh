#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-feedback}"
KEEP_QUEUES="false"

for arg in "$@"; do
    case "$arg" in
        setup|feedback|build|infer|full)
            MODE="$arg"
            ;;
        --keep-queues)
            KEEP_QUEUES="true"
            ;;
    esac
done

QUEUES=(
    "q.data.build"
    "q.train.run"
    "q.model.promoted"
    "q.infer.request"
    "q.infer.result"
    "q.label.task"
    "q.example.dataset"
)

REPO_ROOT="$(pwd)"
LOG_DIR=".demo_logs"
PIDS=()

write_section() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Command not found: $1"
        echo "Please install it or add it to PATH."
        exit 1
    fi
}

assert_repo_root() {
    if [[ ! -f "docker-compose.yml" ]]; then
        echo "docker-compose.yml not found. Run this script from the project root."
        exit 1
    fi

    if [[ ! -f "src/messaging/rabbitmq.py" ]]; then
        echo "src/messaging/rabbitmq.py not found. Run this script from the project root."
        exit 1
    fi
}

cleanup_workers() {
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        write_section "Stopping demo workers"

        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" >/dev/null 2>&1; then
                echo "Stopping process $pid"
                kill "$pid" >/dev/null 2>&1 || true
            fi
        done
    fi
}

trap cleanup_workers EXIT

invoke_demo_python() {
    local code="$1"
    local error_message="$2"

    local temp_file
    temp_file="$(mktemp)"

    cat > "$temp_file" <<PY_PREFIX
import sys
from pathlib import Path

sys.path.insert(0, r'''$REPO_ROOT''')

PY_PREFIX

    printf "%s\n" "$code" >> "$temp_file"

    if ! uv run python "$temp_file"; then
        rm -f "$temp_file"
        echo "$error_message"
        exit 1
    fi

    rm -f "$temp_file"
}

start_rabbitmq() {
    write_section "Starting RabbitMQ with Docker Compose"

    docker compose up -d

    echo "Waiting for RabbitMQ to become ready..."

    for _ in $(seq 1 30); do
        if docker exec mlops-rabbitmq rabbitmq-diagnostics -q ping >/dev/null 2>&1; then
            echo "RabbitMQ is ready."
            return
        fi

        sleep 2
    done

    echo "RabbitMQ did not become ready in time."
    exit 1
}

declare_queues() {
    write_section "Declaring RabbitMQ queues"

    local python_code
    python_code=$(cat <<'PY'
from src.messaging.rabbitmq import create_connection, declare_queues

queues = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
    "q.example.dataset",
]

connection = create_connection()
channel = connection.channel()

declare_queues(channel, queues)

connection.close()

print("Queues declared:")
for queue in queues:
    print(f"  {queue}")
PY
)

    invoke_demo_python "$python_code" "Queue declaration failed."
}

purge_queues() {
    write_section "Purging RabbitMQ queues"

    for queue in "${QUEUES[@]}"; do
        echo "Purging $queue"
        docker exec mlops-rabbitmq rabbitmqctl purge_queue "$queue" || true
    done
}

show_queues() {
    write_section "Current RabbitMQ queue state"

    docker exec mlops-rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers
}

start_worker() {
    local name="$1"
    shift

    mkdir -p "$LOG_DIR"

    local log_file="$LOG_DIR/${name}.log"

    echo "Starting $name"
    echo "Log: $log_file"

    nohup "$@" > "$log_file" 2>&1 &
    local pid="$!"

    PIDS+=("$pid")

    echo "$pid" > "$LOG_DIR/${name}.pid"
    echo "Started $name with pid=$pid"
}

publish_data_build() {
    write_section "Publishing q.data.build"

    local python_code
    python_code=$(cat <<'PY'
from src.messaging.rabbitmq import create_connection, declare_queues, publish_json

queues = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
    "q.example.dataset",
]

message = {
    "event": "data.build",
    "trigger": "manual",
    "raw_uri": "data/raw",
    "params": {
        "val_frac": 0.15,
        "test_frac": 0.15,
        "seed": 42,
    },
}

connection = create_connection()
channel = connection.channel()

declare_queues(channel, queues)
publish_json(
    channel=channel,
    queue="q.data.build",
    message=message,
)

connection.close()

print("Published q.data.build:")
print(message)
PY
)

    invoke_demo_python "$python_code" "Failed to publish q.data.build."
}

publish_infer_request() {
    write_section "Publishing q.infer.request"

    local python_code
    python_code=$(cat <<'PY'
from src.messaging.rabbitmq import create_connection, declare_queues, publish_json

queues = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
    "q.example.dataset",
]

message = {
    "event": "infer.request",
    "image_uri": "data/stream/images/BloodImage_00000.jpg",
    "model_version": "production",
}

connection = create_connection()
channel = connection.channel()

declare_queues(channel, queues)
publish_json(
    channel=channel,
    queue="q.infer.request",
    message=message,
)

connection.close()

print("Published q.infer.request:")
print(message)
PY
)

    invoke_demo_python "$python_code" "Failed to publish q.infer.request."
}

publish_fake_low_confidence_infer_result() {
    write_section "Publishing fake low-confidence q.infer.result"

    local python_code
    python_code=$(cat <<'PY'
from datetime import datetime, timezone
from uuid import uuid4

from src.messaging.rabbitmq import create_connection, declare_queues, publish_json

queues = [
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
    "q.example.dataset",
]

inference_id = "inf-demo-feedback-" + uuid4().hex[:8]

message = {
    "event": "infer.result",
    "inference_id": inference_id,
    "model_version": "production",
    "status": "success",
    "image_uri": "data/stream/images/BloodImage_00000.jpg",
    "latency_ms": 35.5,
    "min_conf": 0.30,
    "ts": datetime.now(timezone.utc).isoformat(),
    "detections": [
        {
            "cls": "RBC",
            "conf": 0.30,
            "box": [10.0, 20.0, 80.0, 90.0],
        }
    ],
    "source_event": {
        "event": "infer.request",
        "image_uri": "data/stream/images/BloodImage_00000.jpg",
        "model_version": "production",
    },
}

connection = create_connection()
channel = connection.channel()

declare_queues(channel, queues)
publish_json(
    channel=channel,
    queue="q.infer.result",
    message=message,
)

connection.close()

print("Published q.infer.result:")
print(message)
PY
)

    invoke_demo_python "$python_code" "Failed to publish fake q.infer.result."
}

run_setup() {
    require_command docker
    require_command uv

    assert_repo_root
    start_rabbitmq
    declare_queues

    if [[ "$KEEP_QUEUES" != "true" ]]; then
        purge_queues
    fi

    show_queues
}

run_feedback_demo() {
    run_setup

    write_section "Starting feedback-loop workers"

    start_worker "collect_worker_real" uv run python src/workers/collect_worker_real.py
    sleep 2

    start_worker "oracle_annotation_worker" uv run python src/workers/oracle_annotation_worker.py
    sleep 2

    publish_fake_low_confidence_infer_result

    echo ""
    echo "Waiting for Collect Worker and Oracle Annotation Worker..."
    sleep 8

    show_queues

    echo ""
    echo "Expected result before Data Worker:"
    echo "  q.infer.result  -> 0"
    echo "  q.label.task    -> 0"
    echo "  q.data.build    -> 1"
    echo ""
    echo "This means:"
    echo "  Collect Worker selected the low-confidence case."
    echo "  Oracle Annotation Worker injected image + label into data/raw."
    echo "  Oracle Annotation Worker published q.data.build automatically."

    write_section "Starting Data Worker to consume feedback q.data.build"

    start_worker "data_worker_real" uv run python src/workers/data_worker_real.py

    echo ""
    echo "Waiting for Data Worker to create a new dataset..."
    sleep 12

    show_queues

    echo ""
    echo "Expected result after Data Worker:"
    echo "  q.data.build -> 0"
    echo "  q.train.run  -> 1"
    echo ""
    echo "This proves the feedback build was consumed and a new train event was published."
}

run_full_demo() {
    run_setup

    write_section "Starting all real workers"

    start_worker "data_worker_real" uv run python src/workers/data_worker_real.py
    sleep 1

    start_worker "train_worker_real" uv run python src/workers/train_worker_real.py
    sleep 1

    start_worker "infer_worker_real" uv run python src/workers/infer_worker_real.py
    sleep 1

    start_worker "collect_worker_real" uv run python src/workers/collect_worker_real.py
    sleep 1

    start_worker "oracle_annotation_worker" uv run python src/workers/oracle_annotation_worker.py
    sleep 2

    publish_data_build

    echo ""
    echo "Full demo started."
    echo ""
    echo "What happens now:"
    echo "  1. Data Worker consumes q.data.build."
    echo "  2. Data Worker publishes q.train.run."
    echo "  3. Train Worker trains/evaluates/promotes if the model passes the gate."
    echo ""
    echo "Note: training can take some time."
    echo "Worker logs are available in:"
    echo "  $LOG_DIR"

    show_queues
}

write_section "MLOps Pipeline Demo Script"
echo "Mode: $MODE"

case "$MODE" in
    setup)
        run_setup
        ;;

    feedback)
        run_feedback_demo
        ;;

    build)
        require_command docker
        require_command uv
        assert_repo_root
        start_rabbitmq
        declare_queues
        publish_data_build
        show_queues
        ;;

    infer)
        require_command docker
        require_command uv
        assert_repo_root
        start_rabbitmq
        declare_queues
        publish_infer_request
        show_queues
        ;;

    full)
        run_full_demo
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage:"
        echo "  bash scripts/demo_pipeline.sh setup"
        echo "  bash scripts/demo_pipeline.sh feedback"
        echo "  bash scripts/demo_pipeline.sh build"
        echo "  bash scripts/demo_pipeline.sh infer"
        echo "  bash scripts/demo_pipeline.sh full"
        echo ""
        echo "Optional:"
        echo "  --keep-queues"
        exit 1
        ;;
esac