import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
PRODUCTION_POINTER_FILE = ROOT_DIR / "storage" / "models" / "production.json"


def load_production_pointer() -> dict[str, Any] | None:
    if not PRODUCTION_POINTER_FILE.exists():
        return None

    return json.loads(PRODUCTION_POINTER_FILE.read_text(encoding="utf-8"))


def main() -> None:
    pointer = load_production_pointer()

    if pointer is None:
        status = {
            "status": "healthy",
            "model_version": "v0",
            "model_uri": "models/v0",
            "production_pointer_exists": False,
            "message": "No promoted model pointer found. Inference should fall back to v0.",
        }
        print(json.dumps(status, indent=2))
        return

    model_uri = pointer["model_uri"]
    model_path = ROOT_DIR / model_uri / "best.onnx"

    status = {
        "status": "healthy" if model_path.exists() else "model_file_missing",
        "model_version": pointer.get("model_version"),
        "model_uri": model_uri,
        "model_path": str(model_path),
        "model_file_exists": model_path.exists(),
        "dataset_version": pointer.get("dataset_version"),
        "metrics": pointer.get("metrics"),
        "baseline": pointer.get("baseline"),
        "base_model": pointer.get("base_model"),
        "updated_at": pointer.get("updated_at"),
        "production_pointer_exists": True,
    }

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()