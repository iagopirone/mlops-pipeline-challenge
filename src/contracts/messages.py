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