"""
COCO val (80 class) → 학습 vocab (40 class) 라벨 리매핑.
- COCO 80 라벨 중 우리 40 vocab과 겹치는 32개 클래스만 추출, 인덱스 재할당
- 8개 (traffic sign, crosswalk, pothole, road marking, construction cone, barrier 등)
  는 COCO에 없으므로 GT=0으로 평가 (model.val()에서 NaN 처리)

기본값 (docker 안):
  --src_images   /data/coco/val2017/images
  --src_labels   /data/coco/val2017/labels
  --dst_root     /app/eval/val2017
"""
import argparse
import os
from pathlib import Path


# COCO 80 → 우리 40 vocab 인덱스 매핑 (32개 overlap)
COCO_80_TO_OUR_40 = {
    0:  0,    # person
    1:  1,    # bicycle
    2:  2,    # car
    3:  3,    # motorcycle
    4:  8,    # airplane
    5:  4,    # bus
    6:  6,    # train
    7:  5,    # truck
    8:  7,    # boat
    9:  9,    # traffic light
    10: 12,   # fire hydrant
    11: 11,   # stop sign
    12: 13,   # parking meter
    13: 14,   # bench
    14: 20,   # bird
    15: 21,   # cat
    16: 22,   # dog
    17: 23,   # horse
    18: 24,   # sheep
    19: 25,   # cow
    24: 26,   # backpack
    25: 28,   # umbrella
    26: 27,   # handbag
    28: 29,   # suitcase
    32: 30,   # sports ball → ball
    33: 31,   # kite
    36: 32,   # skateboard
    37: 33,   # surfboard
    39: 34,   # bottle
    41: 35,   # cup
    56: 36,   # chair
    63: 37,   # laptop
    67: 38,   # cell phone
    73: 39,   # book
}

# 우리 40 vocab 중 COCO에 없는 8개 (참고용)
NON_COCO_OUR_40 = {
    10: "traffic sign",
    15: "crosswalk",
    16: "pothole",
    17: "road marking",
    18: "construction cone",
    19: "barrier",
}


def remap_label_file(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """한 라벨 파일을 읽어 매핑된 줄만 출력. (kept, dropped) 반환."""
    kept = dropped = 0
    out_lines = []
    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                coco_id = int(parts[0])
            except ValueError:
                continue
            if coco_id not in COCO_80_TO_OUR_40:
                dropped += 1
                continue
            new_id = COCO_80_TO_OUR_40[coco_id]
            out_lines.append(" ".join([str(new_id)] + parts[1:]))
            kept += 1
    # 빈 파일이라도 생성 (해당 이미지에 우리 40 vocab 객체가 없다는 의미)
    with open(dst_path, "w") as f:
        if out_lines:
            f.write("\n".join(out_lines) + "\n")
    return kept, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_labels", default="/data/coco_orig_labels",
                        help="원본 COCO 80-class 라벨 디렉토리")
    parser.add_argument("--dst_labels", default="/app/eval/val2017/labels",
                        help="40-class로 remap된 라벨을 쓸 디렉토리")
    args = parser.parse_args()

    src_labels = Path(args.src_labels)
    dst_labels = Path(args.dst_labels)

    # 이미지는 docker-compose에서 /app/eval/val2017/images 로 직접 bind mount.
    # 심볼릭 안 만들어도 ultralytics가 그대로 찾음.

    # 라벨 변환
    dst_labels.mkdir(parents=True, exist_ok=True)
    label_files = sorted(src_labels.glob("*.txt"))
    print(f"📂 라벨 파일 {len(label_files):,}개 처리 중...")

    total_kept = total_dropped = files_empty = 0
    for i, src_f in enumerate(label_files):
        dst_f = dst_labels / src_f.name
        k, d = remap_label_file(src_f, dst_f)
        total_kept += k
        total_dropped += d
        if k == 0:
            files_empty += 1
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{len(label_files)}", end="\r", flush=True)
    print()

    # 3) labels.cache 삭제 (옛 버전이 남아있을 수 있음)
    for cache_path in [dst_labels.parent / "labels.cache",
                       Path("/data/coco_orig_labels.cache"),
                       Path("/data/coco/val2017/labels.cache")]:
        try:
            if cache_path.exists():
                cache_path.unlink()
        except Exception:
            pass

    # 4) 요약
    print(f"\n{'='*50}")
    print(f"🎉 리매핑 완료")
    print(f"   라벨 출력:     {dst_labels}")
    print(f"   라벨 줄 유지:   {total_kept:,}")
    print(f"   라벨 줄 버림:   {total_dropped:,}  (COCO-only 클래스)")
    print(f"   객체 없는 파일: {files_empty:,}")
    print(f"{'='*50}")
    print(f"\n⚠️  우리 40 vocab 중 COCO에 없는 8개 클래스 (GT=0, mAP NaN):")
    for idx, name in NON_COCO_OUR_40.items():
        print(f"   [{idx:2d}] {name}")


if __name__ == "__main__":
    main()
