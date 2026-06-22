param(
    [ValidateSet("setup", "feedback", "build", "infer", "full")]
    [string]$Mode = "feedback",

    [switch]$KeepQueues
)

$ErrorActionPreference = "Stop"

$Queues = @(
    "q.data.build",
    "q.train.run",
    "q.model.promoted",
    "q.infer.request",
    "q.infer.result",
    "q.label.task",
    "q.example.dataset"
)

function Write-Section {
    param([string]$Text)

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Text
    Write-Host "============================================================"
}

function Require-Command {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue

    if (-not $command) {
        throw "Command not found: $Name. Please install it or add it to PATH."
    }
}

function Assert-RepoRoot {
    if (-not (Test-Path "docker-compose.yml")) {
        throw "docker-compose.yml not found. Run this script from the project root."
    }

    if (-not (Test-Path "src\messaging\rabbitmq.py")) {
        throw "src\messaging\rabbitmq.py not found. Run this script from the project root."
    }
}

function Invoke-DemoPython {
    param(
        [string]$Code,
        [string]$ErrorMessage
    )

    $tempFile = Join-Path ([System.IO.Path]::GetTempPath()) ("mlops_demo_" + [guid]::NewGuid().ToString("N") + ".py")
    $repoPath = (Get-Location).Path

    $pythonPrefix = @"
import sys
from pathlib import Path

sys.path.insert(0, r'''$repoPath''')

"@

    try {
        Set-Content -Path $tempFile -Value ($pythonPrefix + $Code) -Encoding UTF8

        & uv run python $tempFile

        if ($LASTEXITCODE -ne 0) {
            throw $ErrorMessage
        }
    }
    finally {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }
}

function Start-RabbitMq {
    Write-Section "Starting RabbitMQ with Docker Compose"

    & docker compose up -d

    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up -d failed."
    }

    Write-Host "Waiting for RabbitMQ to become ready..."

    for ($i = 1; $i -le 30; $i++) {
        & docker exec mlops-rabbitmq rabbitmq-diagnostics -q ping *> $null

        if ($LASTEXITCODE -eq 0) {
            Write-Host "RabbitMQ is ready."
            return
        }

        Start-Sleep -Seconds 2
    }

    throw "RabbitMQ did not become ready in time."
}

function Declare-Queues {
    Write-Section "Declaring RabbitMQ queues"

    $queueListPython = ($Queues | ForEach-Object { "'$_'" }) -join ", "

    $pythonCode = @"
from src.messaging.rabbitmq import create_connection, declare_queues

queues = [$queueListPython]

connection = create_connection()
channel = connection.channel()

declare_queues(channel, queues)

connection.close()

print("Queues declared:")
for queue in queues:
    print(f"  {queue}")
"@

    Invoke-DemoPython -Code $pythonCode -ErrorMessage "Queue declaration failed."
}

function Purge-Queues {
    Write-Section "Purging RabbitMQ queues"

    foreach ($queue in $Queues) {
        Write-Host "Purging $queue"
        & docker exec mlops-rabbitmq rabbitmqctl purge_queue $queue | Out-Host
    }
}

function Show-Queues {
    Write-Section "Current RabbitMQ queue state"

    & docker exec mlops-rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers
}

function Start-WorkerWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $repoPath = (Get-Location).Path

    $powerShellCommand = @"
cd "$repoPath"
Write-Host "Running $Title"
Write-Host "$Command"
$Command
"@

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $powerShellCommand
    ) | Out-Null

    Write-Host "Started worker window: $Title"
}

function Publish-DataBuild {
    Write-Section "Publishing q.data.build"

    $queueListPython = ($Queues | ForEach-Object { "'$_'" }) -join ", "

    $pythonCode = @"
from src.messaging.rabbitmq import create_connection, declare_queues, publish_json

queues = [$queueListPython]

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
"@

    Invoke-DemoPython -Code $pythonCode -ErrorMessage "Failed to publish q.data.build."
}

function Publish-InferRequest {
    Write-Section "Publishing q.infer.request"

    $queueListPython = ($Queues | ForEach-Object { "'$_'" }) -join ", "

    $pythonCode = @"
from src.messaging.rabbitmq import create_connection, declare_queues, publish_json

queues = [$queueListPython]

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
"@

    Invoke-DemoPython -Code $pythonCode -ErrorMessage "Failed to publish q.infer.request."
}

function Publish-FakeLowConfidenceInferResult {
    Write-Section "Publishing fake low-confidence q.infer.result"

    $queueListPython = ($Queues | ForEach-Object { "'$_'" }) -join ", "

    $pythonCode = @"
from datetime import datetime, timezone
from uuid import uuid4

from src.messaging.rabbitmq import create_connection, declare_queues, publish_json

queues = [$queueListPython]

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
"@

    Invoke-DemoPython -Code $pythonCode -ErrorMessage "Failed to publish fake q.infer.result."
}

function Start-FeedbackWorkers {
    Write-Section "Starting feedback-loop workers"

    Start-WorkerWindow -Title "Collect Worker Real" -Command "uv run python src\workers\collect_worker_real.py"
    Start-Sleep -Seconds 2

    Start-WorkerWindow -Title "Oracle Annotation Worker" -Command "uv run python src\workers\oracle_annotation_worker.py"
    Start-Sleep -Seconds 2
}

function Start-AllWorkers {
    Write-Section "Starting all real workers"

    Start-WorkerWindow -Title "Data Worker Real" -Command "uv run python src\workers\data_worker_real.py"
    Start-Sleep -Seconds 1

    Start-WorkerWindow -Title "Train Worker Real" -Command "uv run python src\workers\train_worker_real.py"
    Start-Sleep -Seconds 1

    Start-WorkerWindow -Title "Inference Worker Real" -Command "uv run python src\workers\infer_worker_real.py"
    Start-Sleep -Seconds 1

    Start-WorkerWindow -Title "Collect Worker Real" -Command "uv run python src\workers\collect_worker_real.py"
    Start-Sleep -Seconds 1

    Start-WorkerWindow -Title "Oracle Annotation Worker" -Command "uv run python src\workers\oracle_annotation_worker.py"
    Start-Sleep -Seconds 2
}

function Run-Setup {
    Require-Command "docker"
    Require-Command "uv"

    Assert-RepoRoot
    Start-RabbitMq
    Declare-Queues

    if (-not $KeepQueues) {
        Purge-Queues
    }

    Show-Queues
}

function Run-FeedbackDemo {
    Run-Setup

    Start-FeedbackWorkers

    Publish-FakeLowConfidenceInferResult

    Write-Host ""
    Write-Host "Waiting a few seconds for Collect Worker and Oracle Annotation Worker..."
    Start-Sleep -Seconds 8

    Show-Queues

    Write-Host ""
    Write-Host "Expected result at this point:"
    Write-Host "  q.infer.result  -> 0"
    Write-Host "  q.label.task    -> 0"
    Write-Host "  q.data.build    -> 1"
    Write-Host ""
    Write-Host "This means:"
    Write-Host "  Collect Worker selected the low-confidence case."
    Write-Host "  Oracle Annotation Worker injected image + label into data/raw."
    Write-Host "  Oracle Annotation Worker published q.data.build automatically."

    Write-Host ""
    $answer = Read-Host "Do you want to start Data Worker now to consume q.data.build? (y/n)"

    if ($answer -match "^[YySs]") {
        Start-WorkerWindow -Title "Data Worker Real" -Command "uv run python src\workers\data_worker_real.py"

        Write-Host ""
        Write-Host "Waiting for Data Worker to create the dataset..."
        Start-Sleep -Seconds 12

        Show-Queues

        Write-Host ""
        Write-Host "Expected result after Data Worker:"
        Write-Host "  q.data.build -> 0"
        Write-Host "  q.train.run  -> 1"
        Write-Host ""
        Write-Host "This proves the feedback build was consumed and a new train event was published."
    }
}

function Run-FullDemo {
    Run-Setup

    Start-AllWorkers

    Publish-DataBuild

    Write-Host ""
    Write-Host "Full demo started."
    Write-Host ""
    Write-Host "What happens now:"
    Write-Host "  1. Data Worker consumes q.data.build."
    Write-Host "  2. Data Worker publishes q.train.run."
    Write-Host "  3. Train Worker trains/evaluates/promotes if the model passes the gate."
    Write-Host "  4. After training finishes, run this command to publish an inference request:"
    Write-Host ""
    Write-Host "     powershell -ExecutionPolicy Bypass -File scripts\demo_pipeline.ps1 -Mode infer -KeepQueues"
    Write-Host ""
    Write-Host "Note: training can take some time."
    Write-Host ""

    Show-Queues
}

Write-Section "MLOps Pipeline Demo Script"
Write-Host "Mode: $Mode"

switch ($Mode) {
    "setup" {
        Run-Setup
    }

    "feedback" {
        Run-FeedbackDemo
    }

    "build" {
        Require-Command "docker"
        Require-Command "uv"
        Assert-RepoRoot
        Start-RabbitMq
        Declare-Queues
        Publish-DataBuild
        Show-Queues
    }

    "infer" {
        Require-Command "docker"
        Require-Command "uv"
        Assert-RepoRoot
        Start-RabbitMq
        Declare-Queues
        Publish-InferRequest
        Show-Queues
    }

    "full" {
        Run-FullDemo
    }
}