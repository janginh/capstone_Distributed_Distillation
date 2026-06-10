"""
3개 모델 (Student baseline / KD-trained / Teacher) 검출 수 비교.
영상 N 프레임 샘플링해서 person, car 검출 수와 gap ratio 측정.

사용:
  python /app/scripts/compare_three_models.py \
      --video /app/eval/test_h264/your.mp4 \
      --student yolov8s-worldv2.pt \
      --kd /app/weights/video_kd_dark.pt \
      --teacher yolov8x-worldv2.pt
"""
import argparse
import cv2
from ultralytics import YOLOWorld


CLASSES = ["person", "car"]


def run_model(model, frame, conf):
    r = model.predict(frame, conf=conf, verbose=False)[0]
    counts = {c: 0 for c in range(len(CLASSES))}
    for c in r.boxes.cls.int().tolist():
        counts[c] = counts.get(c, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--student", default="yolov8s-worldv2.pt",
                   help="원본 student baseline (KD 전)")
    p.add_argument("--kd", required=True,
                   help="KD 학습된 모델 (.pt)")
    p.add_argument("--teacher", default="yolov8x-worldv2.pt")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    print(f"📥 모델 3개 로드 중...")
    s = YOLOWorld(args.student); s.set_classes(CLASSES)
    k = YOLOWorld(args.kd);      k.set_classes(CLASSES)
    t = YOLOWorld(args.teacher); t.set_classes(CLASSES)
    print(f"   student (KD 전):  {args.student}")
    print(f"   kd      (학습됨): {args.kd}")
    print(f"   teacher (상한선): {args.teacher}")
    print(f"   classes: {CLASSES}, conf={args.conf}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 영상 못 열음: {args.video}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    interval = max(total // args.n_samples, 1)
    print(f"\n🎥 {args.video}")
    print(f"   {total:,}프레임 @ {fps_v:.0f}fps → {args.n_samples}개 샘플 (매 {interval}프레임)")
    print(f"\n🚀 비교 추론 시작...\n")

    sums = {
        "student": {0: 0, 1: 0},
        "kd":      {0: 0, 1: 0},
        "teacher": {0: 0, 1: 0},
    }
    sampled = 0
    fr = 0

    while fr < total and sampled < args.n_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame = cap.read()
        if not ret:
            break
        sampled += 1

        for name, model in [("student", s), ("kd", k), ("teacher", t)]:
            cnt = run_model(model, frame, args.conf)
            for c in cnt:
                sums[name][c] = sums[name].get(c, 0) + cnt[c]

        if sampled % 10 == 0:
            print(f"\r  📊 {sampled}/{args.n_samples} 프레임", end="", flush=True)

        fr += interval

    cap.release()

    def total_count(d):
        return sum(d.values())

    s_tot = total_count(sums["student"])
    k_tot = total_count(sums["kd"])
    t_tot = total_count(sums["teacher"])

    # ============ 출력 ============
    print(f"\n\n{'=' * 75}")
    print(f"📊 검출 수 비교 (n={sampled} 프레임)")
    print(f"{'=' * 75}")
    header = f"{'클래스':10s}  {'student':>12s}  {'kd':>12s}  {'teacher':>12s}"
    print(header)
    print(f"{'-' * 75}")
    for c, name in enumerate(CLASSES):
        ss = sums["student"].get(c, 0)
        kk = sums["kd"].get(c, 0)
        tt = sums["teacher"].get(c, 0)
        print(f"{name:10s}  {ss:>7d} ({ss/sampled:.1f}/f)  "
              f"{kk:>7d} ({kk/sampled:.1f}/f)  "
              f"{tt:>7d} ({tt/sampled:.1f}/f)")
    print(f"{'-' * 75}")
    print(f"{'TOTAL':10s}  {s_tot:>7d} ({s_tot/sampled:.1f}/f)  "
          f"{k_tot:>7d} ({k_tot/sampled:.1f}/f)  "
          f"{t_tot:>7d} ({t_tot/sampled:.1f}/f)")
    print(f"{'=' * 75}")

    # ============ 비율 분석 ============
    print(f"\n📈 비율 분석 (Teacher 대비)")
    print(f"{'-' * 75}")
    for c, name in enumerate(CLASSES):
        ss = sums["student"].get(c, 0)
        kk = sums["kd"].get(c, 0)
        tt = sums["teacher"].get(c, 0)
        s_r = ss / tt if tt else 0
        k_r = kk / tt if tt else 0
        s_gap = tt / ss if ss else 0
        k_gap = tt / kk if kk else 0
        print(f"  {name}:")
        print(f"    student: {ss}/{tt} = {s_r*100:5.1f}% of teacher (gap {s_gap:.2f}x)")
        print(f"    kd:      {kk}/{tt} = {k_r*100:5.1f}% of teacher (gap {k_gap:.2f}x)")
        if kk > ss:
            improvement = (kk - ss) / ss * 100 if ss else 0
            print(f"    → KD가 student 대비 +{improvement:.1f}% 더 검출 ✅")
        elif kk < ss:
            decrease = (ss - kk) / ss * 100 if ss else 0
            print(f"    → KD가 student 대비 -{decrease:.1f}% 적게 검출 ⚠️")
        else:
            print(f"    → KD가 student와 동일")
    print(f"{'-' * 75}")

    # 전체 요약
    s_r_total = s_tot / t_tot if t_tot else 0
    k_r_total = k_tot / t_tot if t_tot else 0
    s_gap_total = t_tot / s_tot if s_tot else 0
    k_gap_total = t_tot / k_tot if k_tot else 0
    print(f"\n📋 전체 요약:")
    print(f"   student: teacher 대비 {s_r_total*100:.1f}% 검출 (gap {s_gap_total:.2f}x)")
    print(f"   kd:      teacher 대비 {k_r_total*100:.1f}% 검출 (gap {k_gap_total:.2f}x)")

    # KD 효과 평가
    print(f"\n💡 KD 효과 평가:")
    gap_reduction = s_gap_total - k_gap_total
    if k_tot > s_tot:
        rel_imp = (k_tot - s_tot) / s_tot * 100
        print(f"   ✅ KD가 student 대비 +{rel_imp:.1f}% 향상 (gap {s_gap_total:.2f}x → {k_gap_total:.2f}x)")
        if k_gap_total < 1.1:
            print(f"   ✅ KD가 teacher의 90%+ 수준 도달")
        elif k_gap_total < s_gap_total - 0.1:
            print(f"   ✅ 의미 있는 향상 — KD 학습 성공")
        else:
            print(f"   ⚠️  향상은 있지만 미미함 — 추가 학습 또는 다른 시도 고려")
    elif k_tot == s_tot:
        print(f"   ⚠️  student와 동일 — KD 효과 없음")
    else:
        deg = (s_tot - k_tot) / s_tot * 100
        print(f"   ❌ KD가 student 대비 -{deg:.1f}% 저하 — KD 학습 실패 또는 부적합")
        print(f"   → lr, conf, 데이터 도메인 재검토 필요")


if __name__ == "__main__":
    main()
