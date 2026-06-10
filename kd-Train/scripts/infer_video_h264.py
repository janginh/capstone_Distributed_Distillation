"""
영상 추론 + 박스 그려진 mp4 저장 (H.264 직접 인코딩).
- ultralytics 기본 cv2.VideoWriter는 압축률 매우 나빠 70GB+ 생성됨
- 이 스크립트는 ffmpeg subprocess로 직접 H.264 인코딩 → 원본과 비슷한 크기 (3-5GB)

사용:
  python /app/scripts/infer_video_h264.py \
      --model /app/weights/video_kd_v2.pt \
      --video /app/eval/test_h264/test.mp4

요구: 컨테이너에 ffmpeg 설치되어 있어야 함
  docker exec -u 0 kd-trainer apt-get install -y ffmpeg
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLOWorld


CLASSES = ["person", "car"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="/app/eval/infer")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--crf", type=int, default=23,
                   help="H.264 quality: 18=고화질,큼 / 23=균형 / 28=저화질,작음")
    p.add_argument("--preset", default="fast",
                   help="ffmpeg preset: ultrafast/superfast/fast/medium/slow")
    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"❌ 영상 없음: {video_path}")
        sys.exit(1)

    # ffmpeg 확인
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg 없음. 컨테이너에 설치:")
        print("   docker exec -u 0 kd-trainer apt-get install -y ffmpeg")
        sys.exit(1)

    print(f"📥 모델 로드: {args.model}")
    model = YOLOWorld(args.model)
    model.set_classes(CLASSES)
    print(f"🎯 클래스: {CLASSES}")
    print(f"🎥 입력:   {video_path}")

    # 영상 정보
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 영상 못 열음")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 경로
    out_dir = Path(args.output) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / video_path.name

    print(f"💾 출력:   {out_path}")
    print(f"   해상도: {w}x{h} @ {fps:.0f}fps, 총 {total:,}프레임")
    print(f"   CRF={args.crf}, preset={args.preset}, conf={args.conf}\n")

    # ffmpeg subprocess: BGR raw frames → H.264 mp4
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        # 입력 (stdin raw video)
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        # 출력 (H.264 mp4)
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # 추론 + 인코딩
    print(f"🚀 추론 시작...\n")
    t0 = time.time()
    n = 0
    total_det = 0
    cls_counts = {c: 0 for c in CLASSES}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 추론
            result = model.predict(
                frame, conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, device=args.device,
                verbose=False,
            )[0]

            # 박스 그리기 (ultralytics 기본)
            annotated = result.plot()

            # ffmpeg에 raw frame 전송
            try:
                proc.stdin.write(annotated.tobytes())
            except BrokenPipeError:
                print(f"\n❌ ffmpeg 파이프 끊김 (encoding 실패)")
                break

            # 통계
            total_det += len(result.boxes)
            for cls in result.boxes.cls.int().tolist():
                cls_counts[CLASSES[cls]] += 1

            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                fps_proc = n / el if el > 0 else 0
                pct = n / total * 100 if total else 0
                eta = (total - n) / fps_proc if fps_proc > 0 else 0
                print(f"\r  📊 {pct:5.1f}% ({n:,}/{total:,}) | "
                      f"{fps_proc:.1f} FPS | "
                      f"검출 {total_det:,} | "
                      f"ETA {eta/60:.0f}분",
                      end="", flush=True)

    finally:
        # ffmpeg 정리
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        cap.release()

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0

    print(f"\n\n{'='*55}")
    print(f"✅ 완료")
    print(f"   프레임:    {n:,}/{total:,}")
    print(f"   처리 시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    print(f"   평균 FPS:  {n/elapsed:.1f}")
    print(f"   총 검출:   {total_det:,}개")
    for c, cnt in cls_counts.items():
        avg = cnt / n if n else 0
        print(f"   - {c:8s}: {cnt:6,}개 ({avg:.2f}/frame)")
    print(f"\n📁 결과: {out_path}")
    print(f"   파일 크기: {size_mb:.0f}MB ({size_mb/1024:.2f}GB)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
