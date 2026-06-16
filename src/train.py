#!/usr/bin/env python3
"""Treinamento (script base).

Fine-tune de um checkpoint sobre o dataset preparado, avaliação no split de TESTE
e export para ONNX. Treino offline (puxa torch via ultralytics).

Esta é a versão "solta" que roda à mão. No desafio, vire um Worker de Treino que
consome `q.train.run`, aplica o gate e publica `q.model.promoted`.

Uso:
    python src/train.py --data dataset/data.yaml --base models/v0/best.pt --epochs 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset/data.yaml"))
    ap.add_argument("--base", default="yolo11n.pt",
                    help="checkpoint de partida (fine-tune). Use models/v0/best.pt para continuar do v0.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="bccd")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device="cpu", project=str(ROOT / "runs"),
                name=args.name, exist_ok=True, patience=10)

    best = Path(model.trainer.save_dir) / "weights/best.pt"

    # avalia no split de TESTE (fonte de verdade do gate)
    metrics = YOLO(str(best)).val(data=args.data, split="test", device="cpu")
    print(f"TEST mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    # export leve p/ inferência
    YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    print(f"OK -> {best} (+ .onnx)")


if __name__ == "__main__":
    main()
