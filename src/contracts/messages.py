from typing import Any, Literal

from pydantic import BaseModel, Field


class DataBuildParams(BaseModel):
    val_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    test_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    seed: int = 42


class DataBuildMessage(BaseModel):
    event: Literal["data.build"]
    trigger: Literal["manual", "feedback"] = "manual"
    raw_uri: str
    params: DataBuildParams = Field(default_factory=DataBuildParams)


class DatasetCounts(BaseModel):
    train: int = 0
    val: int = 0
    test: int = 0


class TrainRunEvent(BaseModel):
    event: Literal["train.run"] = "train.run"
    run_request_id: str
    dataset_version: str
    dataset_uri: str
    classes: list[str]
    counts: DatasetCounts
    added_this_cycle: int = 0
    created_at: str
    source_event: dict[str, Any]

class ModelMetrics(BaseModel):
    mAP50: float = Field(ge=0.0, le=1.0)
    per_class: dict[str, float] = Field(default_factory=dict)


class ModelPromotedEvent(BaseModel):
    event: Literal["model.promoted"] = "model.promoted"
    model_version: str
    base_model: str
    model_uri: str
    dataset_version: str
    metrics: ModelMetrics
    baseline: float = Field(ge=0.0, le=1.0)
    promoted: Literal[True] = True
    created_at: str
    source_event: dict[str, Any]

class InferRequestEvent(BaseModel):
    event: Literal["infer.request"]
    image_uri: str
    model_version: str = "production"


class Detection(BaseModel):
    cls: str
    conf: float = Field(ge=0.0, le=1.0)
    box: list[float] = Field(min_length=4, max_length=4)


class InferResultEvent(BaseModel):
    event: Literal["infer.result"] = "infer.result"
    inference_id: str
    model_version: str
    status: Literal["success", "error"] = "success"
    image_uri: str
    latency_ms: float = Field(ge=0.0)
    min_conf: float = Field(ge=0.0, le=1.0)
    ts: str
    detections: list[Detection]
    source_event: dict[str, Any]

class LabelTaskEvent(BaseModel):
    event: Literal["label.task"] = "label.task"
    label_task_id: str
    reason: Literal["low_confidence"] = "low_confidence"
    inference_id: str
    image_uri: str
    model_version: str
    min_conf: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    status: Literal["pending_annotation"] = "pending_annotation"
    created_at: str
    source_event: dict[str, Any]