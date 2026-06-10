#!/bin/bash
# 영상 폴더의 모든 .mp4를 H.264로 일괄 변환.
# - 입력: /app/videos/{train,test} (read-only)
# - 출력: /app/eval/{train_h264, test_h264} (writable)
# - ffmpeg 없으면 자동 설치
#
# 사용:
#   docker exec -it kd-trainer bash /app/scripts/convert_to_h264.sh train
#   docker exec -it kd-trainer bash /app/scripts/convert_to_h264.sh test
#   docker exec -it kd-trainer bash /app/scripts/convert_to_h264.sh both

set -e

MODE="${1:-both}"

# ---------- ffmpeg 자동 설치 ----------
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 ffmpeg 설치 중..."
    apt-get update -qq && apt-get install -y -qq ffmpeg 2>&1 | tail -3
    if ! command -v ffmpeg &> /dev/null; then
        echo "❌ ffmpeg 설치 실패"
        exit 1
    fi
    echo "✅ ffmpeg 설치 완료"
fi

# AV1 디코더 확인
if ! ffmpeg -decoders 2>/dev/null | grep -qi "av1\|libdav1d"; then
    echo "⚠️  AV1 디코더 없음 (apt ffmpeg에 libdav1d 없을 때 발생)"
    echo "   AV1 영상이면 변환 실패 가능"
fi

# ---------- 변환 함수 ----------
convert_folder() {
    local SRC="$1"
    local DST="$2"
    local LABEL="$3"

    if [ ! -d "$SRC" ]; then
        echo "⚠️  $LABEL 폴더 없음: $SRC"
        return
    fi

    mkdir -p "$DST"

    echo ""
    echo "================================================"
    echo "🎬 $LABEL 변환: $SRC → $DST"
    echo "================================================"

    local n=0
    local total=$(ls "$SRC"/*.mp4 2>/dev/null | wc -l)
    if [ "$total" -eq 0 ]; then
        echo "   (mp4 없음, skip)"
        return
    fi

    for v in "$SRC"/*.mp4; do
        local name=$(basename "$v")
        n=$((n+1))
        echo ""
        echo "[$n/$total] >>> $name"
        local size_in=$(du -h "$v" | cut -f1)
        echo "    입력: $size_in"

        # 이미 변환됐는지 확인
        if [ -f "$DST/$name" ]; then
            local size_out=$(du -h "$DST/$name" | cut -f1)
            # 코덱 확인 (h264이면 skip)
            local codec=$(ffprobe -v error -select_streams v:0 \
                -show_entries stream=codec_name -of csv=p=0 "$DST/$name" 2>/dev/null)
            if [ "$codec" == "h264" ]; then
                echo "    이미 변환됨 ($codec, $size_out), skip"
                continue
            fi
        fi

        # 변환 실행 (진행률 표시)
        local t0=$(date +%s)
        ffmpeg -y -hide_banner -loglevel warning -stats \
            -i "$v" \
            -c:v libx264 -preset fast -crf 23 -c:a aac \
            "$DST/$name" 2>&1 | tail -3

        local t1=$(date +%s)
        local elapsed=$((t1 - t0))
        local size_out=$(du -h "$DST/$name" 2>/dev/null | cut -f1)
        echo "    완료: ${elapsed}초, 출력 크기 $size_out"
    done

    echo ""
    echo "✅ $LABEL 전체 완료 ($total 파일)"
    ls -lh "$DST"
}

# ---------- 모드별 실행 ----------
case "$MODE" in
    train)
        convert_folder /app/videos/train /app/eval/train_h264 "Train"
        ;;
    test)
        convert_folder /app/videos/test /app/eval/test_h264 "Test"
        ;;
    both)
        convert_folder /app/videos/train /app/eval/train_h264 "Train"
        convert_folder /app/videos/test  /app/eval/test_h264  "Test"
        ;;
    *)
        echo "Usage: $0 [train|test|both]"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "🎉 모든 변환 작업 완료"
echo "================================================"
