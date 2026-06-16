#!/usr/bin/env python3
"""Inferência (script base).

Carrega o modelo ONNX (só onnxruntime — leve, sem torch), roda sobre uma imagem e
devolve as detecções como lista de dicts {cls, conf, box}.

Esta é a versão "solta" que roda à mão. No desafio, vire um Worker de Inferência
que consome `q.infer.request`, GERA um `inference_id` único e publica
`q.infer.result` (que serve de resultado E de evento de coleta).

Uso:
    python src/infer.py --model models/v0/best.onnx --image data/stream/images/<nome>.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
CLASSES = (ROOT / "data/classes.txt").read_text().split()


class Detector:
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), r, px, py

    def __call__(self, frame):
        x, r, px, py = self._preprocess(frame)
        out = self.sess.run(None, {self.inp: x})[0][0].T   # (N, 4+nc)
        boxes, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf > self.conf
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if len(boxes) == 0:
            return []
        xy = np.empty_like(boxes)
        xy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - px) / r
        xy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - py) / r
        xy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - px) / r
        xy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - py) / r
        res = []
        for i in _nms(xy, conf, self.iou):
            x0, y0, x1, y1 = xy[i].astype(int)
            res.append({"cls": CLASSES[int(cls[i])], "conf": float(conf[i]),
                        "box": [int(x0), int(y0), int(x1), int(y1)]})
        return res


def _nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/v0/best.onnx"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    det = Detector(args.model, conf=args.conf)
    img = cv2.imread(args.image)   # BGR — o Detector espera BGR
    if img is None:
        raise SystemExit(f"não consegui ler {args.image}")
    for d in det(img):
        print(f"{d['cls']:10s} conf={d['conf']:.2f} box={d['box']}")


if __name__ == "__main__":
    main()
