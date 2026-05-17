"""Genera un video demo MP4 con vista original + vista anotada lado a lado.

Cumple el requisito § 3.5.3 de la convocatoria: "video de máximo 2 minutos
que muestre la vista original junto al resultado segmentado (lado a lado
o superpuesto)".

Uso:
    python scripts/make_side_by_side.py \\
        --original data/raw/drive_samples/video-977.mov \\
        --annotated data/processed/runs/video-977/annotated.mp4 \\
        --out reports/demo_video-977.mp4

Si las resoluciones difieren, se redimensiona la anotada al tamaño de
la original (la anotada se generó con el mismo size, pero por si acaso).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--annotated", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="cap a 2 min (requisito convocatoria)",
    )
    p.add_argument("--label-orig", default="ORIGINAL")
    p.add_argument("--label-anno", default="SAM 3.1 + OC-SORT + Kalman")
    return p.parse_args()


def put_label(frame, text, x=20, y=40):
    overlay = frame.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(overlay, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cap_o = cv2.VideoCapture(str(args.original))
    cap_a = cv2.VideoCapture(str(args.annotated))
    if not cap_o.isOpened():
        print(f"ERROR: no se pudo abrir {args.original}", file=sys.stderr)
        return 1
    if not cap_a.isOpened():
        print(f"ERROR: no se pudo abrir {args.annotated}", file=sys.stderr)
        return 1

    fps_o = cap_o.get(cv2.CAP_PROP_FPS)
    fps_a = cap_a.get(cv2.CAP_PROP_FPS)
    w_o = int(cap_o.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_o = int(cap_o.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # El video anotado típicamente tiene fps efectivo = fps_in/stride.
    # Para mantener sincronía: usar fps_a y leer frames de original con stride proporcional.
    stride_o = max(1, int(round(fps_o / fps_a))) if fps_a > 0 else 1
    fps_out = fps_a

    # Tamaño común: misma altura, anchos sumados
    target_h = max(h_o, h_a)
    scale_o = target_h / h_o
    scale_a = target_h / h_a
    new_w_o = int(round(w_o * scale_o))
    new_w_a = int(round(w_a * scale_a))
    out_w = new_w_o + new_w_a
    out_h = target_h

    print(f"Original: {w_o}x{h_o} @ {fps_o:.2f} fps")
    print(f"Anotada:  {w_a}x{h_a} @ {fps_a:.2f} fps")
    print(f"Stride original: {stride_o}")
    print(f"Output: {out_w}x{out_h} @ {fps_out:.2f} fps -> {args.out}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        print("ERROR: no se pudo crear writer", file=sys.stderr)
        return 1

    max_frames_out = int(args.max_seconds * fps_out)
    n_out = 0
    while n_out < max_frames_out:
        # Avanzar original stride_o frames
        ok_o = False
        for _ in range(stride_o):
            ok_o, frame_o = cap_o.read()
            if not ok_o:
                break
        if not ok_o:
            break
        ok_a, frame_a = cap_a.read()
        if not ok_a:
            break

        if (frame_o.shape[1], frame_o.shape[0]) != (new_w_o, target_h):
            frame_o = cv2.resize(frame_o, (new_w_o, target_h))
        if (frame_a.shape[1], frame_a.shape[0]) != (new_w_a, target_h):
            frame_a = cv2.resize(frame_a, (new_w_a, target_h))

        put_label(frame_o, args.label_orig)
        put_label(frame_a, args.label_anno)

        combined = np.hstack([frame_o, frame_a])
        writer.write(combined)
        n_out += 1

    writer.release()
    cap_o.release()
    cap_a.release()

    print(f"OK: {n_out} frames escritos = {n_out / fps_out:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
