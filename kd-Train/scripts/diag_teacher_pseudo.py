"""
Teacher가 영상에서 person, car를 어떤 confidence로 잡는지 분포 확인.
- 영상 1개 샘플링 → 100 프레임에 대해 teacher 추론
- 클래스별 conf 히스토그램 출력
- 추천 conf threshold 제안

사용:
  python /app/scripts/diag_teacher_pseudo.py --video /app/eval/train_h264/videoplayback1.mp4
"""
import argparse
import cv2
from collections import Counter
from ultralytics import YOLOWorld


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--teacher", default="yolov8x-worldv2.pt")
    p.add_argument("--n_samples", type=int, default=100, help="샘플링 프레임 수")
    p.add_argument("--low_conf", type=float, default=0.05,
                   help="이 값 이상의 모든 detection 수집")
    args = p.parse_args()

    print(f"📥 Teacher 로드: {args.teacher}")
    m = YOLOWorld(args.teacher)
    m.set_classes(["person", "car"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 영상 못 열음: {args.video}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(total // args.n_samples, 1)
    print(f"🎥 {args.video}")
    print(f"   {total:,}프레임 @ {fps:.0f}fps → {args.n_samples}개 샘플 (매 {interval}프레임)")

    # 클래스별 conf 분포
    bins = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts = {0: Counter(), 1: Counter()}   # 0=person, 1=car
    total_boxes = {0: 0, 1: 0}
    no_det_frames = 0
    sampled = 0

    fr = 0
    while fr < total and sampled < args.n_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame = cap.read()
        if not ret:
            break
        sampled += 1

        r = m.predict(frame, conf=args.low_conf, verbose=False)[0]
        if len(r.boxes) == 0:
            no_det_frames += 1
        for cls, cf in zip(r.boxes.cls.int().tolist(),
                            r.boxes.conf.tolist()):
            total_boxes[cls] += 1
            # 해당하는 bin 찾기
            for i, b in enumerate(bins[:-1]):
                if b <= cf < bins[i+1]:
                    counts[cls][b] += 1
                    break
            else:
                if cf >= bins[-1]:
                    counts[cls][bins[-1]] += 1

        fr += interval

    cap.release()

    print(f"\n📊 결과 (n={sampled} 프레임 샘플링)")
    print(f"   detection 없는 프레임: {no_det_frames}/{sampled}\n")

    for cls, name in [(0, "person"), (1, "car")]:
        avg = total_boxes[cls] / sampled if sampled else 0
        print(f"=== {name} (총 {total_boxes[cls]} 박스, 프레임당 평균 {avg:.1f}) ===")
        cumulative = 0
        for b in reversed(bins):
            n = counts[cls].get(b, 0)
            cumulative += n
            bar = "█" * min(n // 2, 40) if n else ""
            print(f"   conf ≥ {b:.2f}: {cumulative:4d}개 누적  |{bar}")
        print()

    # 추천 threshold
    print("💡 추천 conf threshold")
    for cls, name in [(0, "person"), (1, "car")]:
        if total_boxes[cls] == 0:
            print(f"   {name}: detection 없음 → 영상에 해당 객체가 없거나 teacher 못 잡음")
            continue
        # 상위 60% 박스를 보존하는 threshold 찾기
        target = int(total_boxes[cls] * 0.6)
        cum = 0
        rec_thresh = 0.5
        for b in reversed(bins):
            cum += counts[cls].get(b, 0)
            if cum >= target:
                rec_thresh = b
                break
        print(f"   {name}: conf >= {rec_thresh:.2f} (상위 60% 박스 유지)")


if __name__ == "__main__":
    main()
