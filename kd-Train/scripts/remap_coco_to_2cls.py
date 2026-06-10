"""
COCO val (80 class) → person(0), car(1) 2-class로 라벨 리매핑.
- COCO idx 0 (person) → 0
- COCO idx 2 (car)    → 1
- 나머지 78 클래스는 모두 버림
- 결과: /app/eval/coco2/val2017/labels/ + symlink로 이미지 연결

ultralytics 라벨 자동 검색 규칙: images → labels 문자열 치환
"""
import os
from pathlib import Path


COCO_TO_2CLS = {
    0: 0,   # person
    2: 1,   # car
}


def remap(src: Path, dst: Path) -> int:
    n = 0
    out = []
    try:
        with open(src) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    c = int(parts[0])
                except ValueError:
                    continue
                if c in COCO_TO_2CLS:
                    out.append(" ".join([str(COCO_TO_2CLS[c])] + parts[1:]))
                    n += 1
    except Exception:
        pass
    with open(dst, "w") as f:
        if out:
            f.write("\n".join(out) + "\n")
    return n


def main():
    SRC_LABELS = Path("/data/coco/val2017/labels")
    SRC_IMAGES = Path("/data/coco/val2017/images")
    DST_ROOT   = Path("/app/eval/coco2/val2017")
    DST_LABELS = DST_ROOT / "labels"
    DST_IMAGES = DST_ROOT / "images"

    DST_LABELS.mkdir(parents=True, exist_ok=True)

    # 이미지는 docker-compose 이미 mount되어 있어서 사용. 심볼릭으로 우회.
    if not DST_IMAGES.exists():
        # 직접 mount하는 게 더 안전 (이전 LVIS/COCO에서 본 symlink 이슈 회피)
        # 여기선 일단 symlink 시도, 안되면 호스트에서 bind mount 추가 권장
        try:
            os.symlink(SRC_IMAGES, DST_IMAGES)
            print(f"🔗 symlink: {DST_IMAGES} → {SRC_IMAGES}")
        except Exception as e:
            print(f"⚠️  symlink 실패 ({e}). 이미지 path를 절대경로로 data.yaml에 명시 권장")

    label_files = sorted(SRC_LABELS.glob("*.txt"))
    print(f"📂 라벨 파일 {len(label_files):,}개 처리 중...")

    total_kept = files_empty = 0
    for i, src in enumerate(label_files):
        dst = DST_LABELS / src.name
        n = remap(src, dst)
        total_kept += n
        if n == 0:
            files_empty += 1
        if (i + 1) % 500 == 0:
            print(f"   {i+1}/{len(label_files)}", end="\r", flush=True)
    print()

    # labels.cache 정리
    for p in [DST_LABELS.parent / "labels.cache",
              Path("/data/coco/val2017/labels.cache")]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    print(f"\n{'='*50}")
    print(f"🎉 person+car 리매핑 완료")
    print(f"   라벨 출력: {DST_LABELS}")
    print(f"   유지된 박스: {total_kept:,}개")
    print(f"   객체 없는 파일: {files_empty:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
