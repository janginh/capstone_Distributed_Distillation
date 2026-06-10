"""
영상 1개에 학습된 YOLO-World 가중치로 person, car 검출 → 박스 그려진 mp4 저장.

사용:
  # 기본
  python /app/scripts/infer_video.py \
      --model /app/weights/video_kd.pt \
      --video /app/videos/test/test.mp4

  # 저장 위치 변경
  python /app/scripts/infer_video.py \
      --model /app/weights/video_kd.pt \
      --video /app/videos/test/test.mp4 \
      --output /app/eval/my_run

  # conf 조정
  python /app/scripts/infer_video.py \
      --model yolov8s-worldv2.pt \
      --video /app/videos/test/test.mp4 \
      --conf 0.3
"""
import argparse
import time
from pathlib import Path

from ultralytics import YOLOWorld


CLASSES = ["person", "car"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="가중치 파일 (.pt)")
    p.add_argument("--video", required=True, help="입력 영상 파일 (.mp4 등)")
    p.add_argument("--output", default="/app/eval/infer",
                   help="결과 저장 디렉토리 (기본 /app/eval/infer)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou",  type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0",
                   help="cuda device (0,1,2,3 또는 cpu)")
    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"영상 없음: {video_path}")

    print(f"📥 모델 로드: {args.model}")
    model = YOLOWorld(args.model)
    model.set_classes(CLASSES)
    print(f"🎯 탐지 클래스: {CLASSES}")
    print(f"🎥 입력 영상:   {video_path}")
    print(f"💾 저장 위치:   {args.output}/{video_path.stem}/")
    print(f"⚙️  conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, device=cuda:{args.device}")
    print(f"\n🚀 추론 시작...\n")

    t0 = time.time()
    n_frames = 0
    total_det = 0
    cls_counts = {c: 0 for c in CLASSES}

    # stream=True: generator로 받음 → 긴 영상에서도 OOM 안 남
    # save=True: 박스 그려진 mp4를 {project}/{name}/{video.name} 위치에 저장
    results = model.predict(
        source=str(video_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=args.output,
        name=video_path.stem,
        verbose=False,
        stream=True,
    )

    for r in results:
        n_frames += 1
        n_box = len(r.boxes)
        total_det += n_box
        for cls in r.boxes.cls.int().tolist():
            cls_counts[CLASSES[cls]] += 1
        if n_frames % 50 == 0:
            el = time.time() - t0
            fps = n_frames / el if el > 0 else 0
            print(f"\r  📊 {n_frames:,}프레임 처리 | "
                  f"누적 검출 {total_det:,}개 | "
                  f"{fps:.1f} FPS",
                  end="", flush=True)

    elapsed = time.time() - t0
    avg_fps = n_frames / elapsed if elapsed > 0 else 0

    print(f"\n\n{'='*50}")
    print(f"✅ 완료")
    print(f"   프레임 수:   {n_frames:,}장")
    print(f"   처리 시간:   {elapsed:.1f}초")
    print(f"   평균 FPS:    {avg_fps:.1f}")
    print(f"   총 검출:     {total_det:,}개")
    for c, n in cls_counts.items():
        avg = n / n_frames if n_frames else 0
        print(f"   - {c:8s}: {n:6,}개 (프레임당 평균 {avg:.2f})")

    out_video = Path(args.output) / video_path.stem / video_path.name
    print(f"\n📁 결과 영상: {out_video}")
    if not out_video.exists():
        # 확장자가 다를 수 있음 (ultralytics가 mp4 → avi 등으로 바꾸는 경우)
        candidates = list((Path(args.output) / video_path.stem).glob("*"))
        if candidates:
            print(f"   (실제 저장된 파일: {candidates})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
