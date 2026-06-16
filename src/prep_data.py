#!/usr/bin/env python3
"""Tratamento de dados (script base).

Lê a fonte raw (imagens + labels YOLO em data/raw/) e monta um dataset YOLO
particionado em train / val / test, com o data.yaml correspondente.

Esta é a versão "solta" que roda à mão. No desafio, vire um Worker de Dados que
consome `q.data.build` e publica `q.train.run`.

Uso:
    python src/prep_data.py --raw data/raw --out dataset --val 0.15 --test 0.15
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--out", default=str(ROOT / "dataset"))
    ap.add_argument("--classes", default=str(ROOT / "data/classes.txt"))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    classes = Path(args.classes).read_text().split()

    imgs = sorted((raw / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"nenhuma imagem em {raw/'images'}")

    # particiona de forma determinística (mesma seed -> mesmo split)
    random.seed(args.seed)
    random.shuffle(imgs)
    n = len(imgs)
    n_test = round(n * args.test)
    n_val = round(n * args.val)
    test = imgs[:n_test]
    val = imgs[n_test:n_test + n_val]
    train = imgs[n_test + n_val:]

    for split, items in (("train", train), ("val", val), ("test", test)):
        for sub in ("images", "labels"):
            d = out / sub / split
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        for img in items:
            shutil.copy(img, out / "images" / split / img.name)
            lbl = raw / "labels" / (img.stem + ".txt")
            dst = out / "labels" / split / (img.stem + ".txt")
            dst.write_text(lbl.read_text() if lbl.exists() else "")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\nnames: {classes}\n"
    )
    print(f"dataset: train={len(train)} val={len(val)} test={len(test)} "
          f"nc={len(classes)} -> {out/'data.yaml'}")


if __name__ == "__main__":
    main()
