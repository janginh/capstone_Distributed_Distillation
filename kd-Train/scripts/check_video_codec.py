"""OpenCV로 영상 파일들 열고 첫 프레임 읽기 시도 — H.264 변환 확인용."""
import cv2
import os
import sys


def main(target_dir):
    if not os.path.isdir(target_dir):
        print(f"❌ 디렉토리 없음: {target_dir}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(target_dir)
                   if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    if not files:
        print(f"⚠️  영상 파일 없음: {target_dir}")
        return

    ok = bad = 0
    for f in files:
        path = os.path.join(target_dir, f)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ {f}: 못 열음")
            bad += 1
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, _ = cap.read()
        if ret:
            print(f"✅ OK   {f}: {w}x{h} {fps:.0f}fps {n:,}frames")
            ok += 1
        else:
            print(f"❌ FAIL {f}: 프레임 읽기 실패 (코덱 미지원?)")
            bad += 1
        cap.release()

    print(f"\n총 {ok + bad}개 중 정상 {ok}개, 실패 {bad}개")
    sys.exit(0 if bad == 0 else 2)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/app/eval/train_h264"
    main(target)
