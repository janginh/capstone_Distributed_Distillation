"""
KD 학습 체크포인트(state_dict 기반) → Ultralytics YOLO(path) 로딩 가능한 형식으로 변환.

사용법:
  python scripts/convert_kd_ckpt_to_yolo.py \
      --input  /app/checkpoints/student_e003.pt \
      --output /app/checkpoints/student_e003_yolo.pt

변환 후:
  from ultralytics import YOLOWorld
  m = YOLOWorld("student_e003_yolo.pt")
  m.set_classes(["person", "car", ...])   # 학습 시 사용한 vocab과 동일해야 함
  results = m.predict("test.jpg")
"""
import argparse
import os
import sys
from datetime import datetime

import torch

# scripts/kd가 import path에 있어야 ultralytics가 module 찾음
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def convert(kd_path, out_path, base_weights):
    from ultralytics import YOLOWorld

    print(f"📥 KD checkpoint 로드: {kd_path}")
    kd = torch.load(kd_path, map_location="cpu", weights_only=False)

    required = {"model_state_dict"}
    missing = required - set(kd.keys())
    if missing:
        raise KeyError(f"KD checkpoint에 키 누락: {missing}. keys={list(kd.keys())}")

    vocab = kd.get("vocab", None)
    print(f"   epoch={kd.get('epoch')} step={kd.get('global_step')} "
          f"shards_done={kd.get('shards_done')} vocab={len(vocab) if vocab else 'N/A'}")

    print(f"🏗️  base 모델 빌드: {base_weights}")
    wrapper = YOLOWorld(base_weights)
    if vocab:
        wrapper.set_classes(vocab)

    print(f"🔁 학습된 state_dict 주입 (strict=False)")
    missing_k, unexpected_k = wrapper.model.load_state_dict(
        kd["model_state_dict"], strict=False
    )
    if missing_k:
        print(f"   ⚠️  missing keys: {len(missing_k)}")
    if unexpected_k:
        print(f"   ⚠️  unexpected keys: {len(unexpected_k)}")
    if not missing_k and not unexpected_k:
        print(f"   ✅ 키 완벽 일치")

    # Ultralytics 표준 ckpt 포맷
    payload = {
        "model": wrapper.model,       # ★ DetectionModel 객체 자체
        "ema": None,
        "epoch": int(kd.get("epoch", 0)),
        "best_fitness": None,
        "updates": int(kd.get("global_step", 0)),
        "optimizer": None,            # inference만 할 거면 불필요
        "train_args": {},
        "train_metrics": {},
        "train_results": {},
        "date": datetime.now().isoformat(),
        "version": "kd-pipeline-converted",
        "vocab": vocab,               # 비표준 필드, 참고용
    }

    print(f"💾 Ultralytics 호환 형식으로 저장: {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✅ 완료 ({size_mb:.1f}MB)")
    if vocab:
        print()
        print(f"⚠️  사용 시 반드시 set_classes() 호출 필요:")
        print(f"    m = YOLOWorld('{out_path}')")
        print(f"    m.set_classes({vocab[:5]}... ({len(vocab)} classes))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="KD checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="출력 .pt 경로")
    parser.add_argument(
        "--base", default="yolov8s-worldv2.pt",
        help="베이스 모델 (Student 학습에 사용한 것과 동일해야 함)",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.base)
