"""
yolov8s-worldv2 (student baseline) vs yolov8x-worldv2 (teacher) 검출 수 비교.
영상에서 100 프레임 샘플링하여 둘의 detection 개수 차이로 KD 효과 가능성 추정.

사용:
  python /app/scripts/compare_baseline_teacher.py --video /app/eval/test_h264/your.mp4
  python /app/scripts/compare_baseline_teacher.py --video ... --n_samples 50 --conf 0.25
"""
import argparse
import cv2
from ultralytics import YOLOWorld


CLASSES = ["person", "car"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--student", default="yolov8s-worldv2.pt")
    p.add_argument("--teacher", default="yolov8x-worldv2.pt")
    p.add_argument("--n_samples", type=int, default=100, help="샘플링 프레임 수")
    p.add_argument("--conf", type=float, default=0.25, help="검출 conf threshold")
    args = p.parse_args()

    print(f"📥 모델 로드 중...")
    s = YOLOWorld(args.student); s.set_classes(CLASSES)
    t = YOLOWorld(args.teacher); t.set_classes(CLASSES)
    print(f"   student: {args.student}")
    print(f"   teacher: {args.teacher}")
    print(f"   classes: {CLASSES}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 영상 못 열음: {args.video}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    interval = max(total // args.n_samples, 1)
    print(f"\n🎥 {args.video}")
    print(f"   {total:,}프레임 @ {fps_v:.0f}fps → {args.n_samples}개 샘플 (매 {interval}프레임)")
    print(f"   conf={args.conf}\n")
    print(f"🚀 비교 추론 시작...\n")

    # 클래스별 검출 수
    s_total = {0: 0, 1: 0}
    t_total = {0: 0, 1: 0}
    sampled = 0
    fr = 0

    while fr < total and sampled < args.n_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame = cap.read()
        if not ret:
            break
        sampled += 1

        sr = s.predict(frame, conf=args.conf, verbose=False)[0].boxes
        tr = t.predict(frame, conf=args.conf, verbose=False)[0].boxes

        for c in sr.cls.int().tolist():
            s_total[c] = s_total.get(c, 0) + 1
        for c in tr.cls.int().tolist():
            t_total[c] = t_total.get(c, 0) + 1

        if sampled % 10 == 0:
            print(f"\r  📊 {sampled}/{args.n_samples} 프레임 처리", end="", flush=True)

        fr += interval

    cap.release()

    s_sum = sum(s_total.values())
    t_sum = sum(t_total.values())
    s_avg = s_sum / sampled
    t_avg = t_sum / sampled
    ratio = t_sum / s_sum if s_sum else 0

    print(f"\n\n{'='*55}")
    print(f"📊 결과 (n={sampled} 프레임)")
    print(f"{'='*55}")
    print(f"{'클래스':10s}  {'student':>15s}  {'teacher':>15s}  {'gap':>8s}")
    print(f"{'-'*55}")
    for c, name in enumerate(CLASSES):
        s_n = s_total.get(c, 0)
        t_n = t_total.get(c, 0)
        r = t_n / s_n if s_n else 0
        gap_str = f"{r:.2f}x" if s_n else "N/A"
        print(f"{name:10s}  {s_n:>10d} ({s_n/sampled:.1f}/f)  "
              f"{t_n:>10d} ({t_n/sampled:.1f}/f)  {gap_str:>8s}")
    print(f"{'-'*55}")
    print(f"{'TOTAL':10s}  {s_sum:>10d} ({s_avg:.1f}/f)  "
          f"{t_sum:>10d} ({t_avg:.1f}/f)  {ratio:.2f}x")
    print(f"{'='*55}")

    # 해석
    print(f"\n💡 해석")
    if ratio < 1.05:
        print(f"   gap ratio {ratio:.2f}x → ⚠️  거의 차이 없음")
        print(f"   → KD 효과 매우 제한적, baseline yolov8s 그대로 쓰는 게 합리적")
        print(f"   → 다른 데이터/시나리오에서 격차 큰 곳 찾아 KD 적용 권장")
    elif ratio < 1.2:
        print(f"   gap ratio {ratio:.2f}x → 작은 차이")
        print(f"   → KD로 짤 여지 적음 (최대 1-2 mAP 정도)")
        print(f"   → 노력 대비 효과 미미할 가능성 ↑")
    elif ratio < 1.5:
        print(f"   gap ratio {ratio:.2f}x → 의미 있는 차이")
        print(f"   → KD 시도 가치 있음 (잘 되면 3-5 mAP 향상 가능)")
        print(f"   → 학습 영상이 충분히 다양한지 확인 필요")
    else:
        print(f"   gap ratio {ratio:.2f}x → 큰 차이")
        print(f"   → KD 효과 클 잠재력 (5+ mAP 향상 기대)")
        print(f"   → 이 도메인에 KD 적극 시도 권장")

    if t_sum > 0 and s_sum == 0:
        print(f"\n   ⚠️ student가 한 박스도 못 찾음 — 영상 도메인이 매우 어려움")
    if t_sum == 0 and s_sum == 0:
        print(f"\n   ⚠️ 둘 다 못 찾음 — 영상에 person/car 거의 없거나 너무 어려움")


if __name__ == "__main__":
    main()
