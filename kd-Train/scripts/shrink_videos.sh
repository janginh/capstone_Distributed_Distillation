#!/bin/bash
# 폴더 안의 모든 영상을 H.264로 재인코딩해서 크기 줄임.
# 원본 → 같은 폴더에 _small.mp4 접미사로 저장, 옵션으로 원본 교체 가능.
#
# 사용:
#   docker exec -it kd-trainer bash /app/scripts/shrink_videos.sh /app/eval/infer
#   docker exec -it kd-trainer bash /app/scripts/shrink_videos.sh /app/eval/infer 23 replace
#
# 인자:
#   $1: 검색 시작 폴더 (재귀)
#   $2: CRF (default 23, 작을수록 고화질 큰 파일)
#   $3: 'replace' 면 원본 삭제하고 압축본이 원본 이름 대체

# set -e 제거: 한 파일 실패해도 나머지 진행

ROOT="${1:?사용: $0 <폴더> [CRF=23] [replace]}"
CRF="${2:-23}"
MODE="${3:-keep}"   # keep 또는 replace

if [ ! -d "$ROOT" ]; then
    echo "❌ 폴더 없음: $ROOT"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "📦 ffmpeg 설치 중..."
    apt-get update -qq && apt-get install -y -qq ffmpeg 2>&1 | tail -3
fi

echo "🔍 검색: $ROOT (재귀)"
echo "   CRF: $CRF (작을수록 고화질, 23=균형)"
echo "   MODE: $MODE  (keep=원본유지, replace=원본교체)"
echo ""

# 영상 파일 리스트 (이미 _small 접미사 있는 건 제외)
mapfile -t VIDEOS < <(find "$ROOT" -type f \( -name "*.mp4" -o -name "*.avi" \) \
    ! -name "*_small.mp4" ! -name "*_small.avi" | sort)

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "⚠️  영상 없음"
    exit 0
fi

echo "📂 ${#VIDEOS[@]}개 영상 발견"
for v in "${VIDEOS[@]}"; do
    size=$(du -h "$v" | cut -f1)
    echo "   $size  $v"
done
echo ""

total_in=0
total_out=0
n=0

for v in "${VIDEOS[@]}"; do
    n=$((n+1))
    dir=$(dirname "$v")
    base=$(basename "$v")
    # .mp4 또는 .avi 확장자 모두 제거
    base="${base%.mp4}"
    base="${base%.avi}"
    out="$dir/${base}_small.mp4"

    size_in_bytes=$(stat -c %s "$v")
    size_in_h=$(du -h "$v" | cut -f1)

    echo "================================================"
    echo "[$n/${#VIDEOS[@]}] >>> $(basename "$v")"
    echo "    입력: $size_in_h"
    echo "    출력: $(basename "$out")"
    echo "================================================"

    t0=$(date +%s)

    # 이미 H.264 코덱이면 단순 -c:v copy (빠르고 크기도 충분히 작음)
    codec=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name -of csv=p=0 "$v" 2>/dev/null)
    if [ "$codec" == "h264" ]; then
        echo "    이미 H.264, copy 모드 (빠름)"
        ffmpeg -y -hide_banner -loglevel warning -stats \
            -i "$v" -c copy "$out" 2>&1 | tail -3
    else
        # 재인코딩
        ffmpeg -y -hide_banner -loglevel warning -stats \
            -i "$v" \
            -c:v libx264 -preset fast -crf "$CRF" \
            -pix_fmt yuv420p \
            -c:a aac -b:a 128k \
            -movflags +faststart \
            "$out" 2>&1 | tail -3
    fi

    if [ ! -f "$out" ]; then
        echo "    ❌ 변환 실패"
        continue
    fi

    size_out_bytes=$(stat -c %s "$out")
    size_out_h=$(du -h "$out" | cut -f1)
    elapsed=$(($(date +%s) - t0))
    # bc 없이 awk로 비율 계산
    ratio=$(awk "BEGIN {printf \"%.1f\", $size_out_bytes * 100 / $size_in_bytes}")

    echo "    ✅ ${elapsed}초, $size_in_h → $size_out_h (${ratio}%)"

    total_in=$((total_in + size_in_bytes))
    total_out=$((total_out + size_out_bytes))

    # replace 모드: 원본 삭제 후 압축본을 원본 이름으로
    if [ "$MODE" == "replace" ]; then
        rm -f "$v"
        mv "$out" "$v"
        echo "    🔄 원본 교체 완료"
    fi

    echo ""
done

# 전체 통계 (bc 없이 awk로)
in_gb=$(awk "BEGIN {printf \"%.2f\", $total_in / 1073741824}")
out_gb=$(awk "BEGIN {printf \"%.2f\", $total_out / 1073741824}")
saved_gb=$(awk "BEGIN {printf \"%.2f\", ($total_in - $total_out) / 1073741824}")

echo "================================================"
echo "🎉 전체 완료"
echo "   입력 합계: ${in_gb}GB"
echo "   출력 합계: ${out_gb}GB"
echo "   절약:      ${saved_gb}GB"
echo "================================================"
