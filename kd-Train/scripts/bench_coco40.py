"""
COCO val에 대한 KD-trained YOLO-World-s mAP 측정.
- 학습 vocab(40 class) 기준으로 평가
- 라벨은 미리 remap_coco_to_40.py로 변환된 것 사용
- pynvml로 GPU 전력/VRAM 측정 (원본 yolo_test_bench.py와 동일 포맷)

사용법 (docker 안):
  python /app/scripts/bench_coco40.py \
      --model /app/weights/1M.pt \
      --data /app/configs/coco40.yaml

체크포인트가 set_classes 없이 저장된 경우 자동으로 vocab.txt에서 읽어 set.
"""
import argparse
import math
import time
from pathlib import Path

import torch


def get_gpu_handle():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"⚠️  pynvml 비활성화 ({e})")
        return None, None


def get_gpu_power(pynvml, handle):
    if pynvml is None or handle is None:
        return float("nan")
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    # YOLO-World v2: backbone = model[:10], neck+head = model[10:]
    enc = sum(p.numel() for p in model.model[:10].parameters())
    dec = sum(p.numel() for p in model.model[10:].parameters())
    return total / 1e6, enc / 1e6, dec / 1e6


def load_vocab(vocab_path: Path) -> list[str]:
    return [ln.strip() for ln in vocab_path.read_text().splitlines() if ln.strip()]


def count_gt_instances(data_yaml: str, nc: int) -> list[int]:
    """label .txt 파일 직접 스캔해서 클래스별 GT 인스턴스 수 카운트."""
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg["path"])
    val_rel = cfg["val"]
    # ultralytics 규약: images → labels 문자열 치환
    labels_dir = root / val_rel.replace("images", "labels")
    counts = [0] * nc
    if not labels_dir.is_dir():
        print(f"⚠️  labels 디렉토리 없음: {labels_dir}")
        return counts
    for txt in labels_dir.glob("*.txt"):
        try:
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            c = int(parts[0])
                            if 0 <= c < nc:
                                counts[c] += 1
                        except ValueError:
                            pass
        except Exception:
            continue
    return counts


def benchmark(model_path: str, data_yaml: str, vocab_path: str,
              batch: int, imgsz: int, conf: float, iou: float):
    from ultralytics import YOLOWorld

    vocab = load_vocab(Path(vocab_path))
    print(f"📖 vocab: {len(vocab)} 클래스 ({vocab_path})")
    print(f"🏗️  모델 로드: {model_path}")
    model = YOLOWorld(model_path)
    model.set_classes(vocab)

    print("\n" + "=" * 50)
    print("🚀 KD Student COCO val (40-class) 평가 시작")
    print("=" * 50)

    inner = model.model.to("cuda")

    # ---- latency 측정 ----
    def measure_latency(img_size=imgsz, n_iter=100, warmup=20):
        dummy = torch.randn(1, 3, img_size, img_size, device="cuda")
        for _ in range(warmup):
            _ = inner(dummy)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(n_iter):
            _ = inner(dummy)
        torch.cuda.synchronize()
        return ((time.time() - t1) / n_iter) * 1000.0  # ms

    # ---- 검증 ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    pynvml, handle = get_gpu_handle()
    t0 = time.time()

    results = model.val(
        data=data_yaml,
        batch=batch,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save_json=False,
        verbose=False,   # 1203 클래스 라인 출력 끔 (LVIS 시 너무 김)
        plots=False,
    )

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    power = get_gpu_power(pynvml, handle)
    lat = measure_latency()

    # ---- 결과 집계 ----
    rd = results.results_dict
    map50 = rd.get("metrics/mAP50(B)", float("nan"))
    map5095 = rd.get("metrics/mAP50-95(B)", float("nan"))

    # 클래스별 결과 (평가 가능 32 vs NaN 8 분리)
    box = results.box
    per_class_map = box.ap50 if hasattr(box, "ap50") else None
    per_class_map95 = box.maps if hasattr(box, "maps") else None

    total, enc, dec = count_parameters(inner)
    print("\n=== Parameter Summary ===")
    print(f"Total parameters     : {total:.2f}M")
    print(f"Encoder (Backbone)   : {enc:.2f}M")
    print(f"Decoder (Neck+Head)  : {dec:.2f}M")
    print("=========================")

    print("\n📊 [Final Summary Statistics] " + "=" * 20)
    print(f"🎯 Overall mAP50    : {map50:.4f}")
    print(f"🎯 Overall mAP50-95 : {map5095:.4f}")
    print(f"🔥 Peak VRAM        : {peak_vram:.2f} GB")
    print(f"⚡ GPU Power        : {power:.2f} W")
    print(f"⏱️ Latency          : {lat:.2f} ms / img (bs=1, {imgsz}px)")
    print(f"🕐 Total val time   : {elapsed:.1f} s")
    print("=" * 50)

    # 클래스별 분석 (전체는 CSV로 저장, 터미널엔 Top/Bottom만)
    if per_class_map95 is not None:
        print("\n📊 GT 인스턴스 카운트 중...")
        gt_counts = count_gt_instances(data_yaml, len(vocab))

        all_classes = []                # CSV용 (전체)
        eval_ok, detect_fail, no_gt = [], [], []
        for i, name in enumerate(vocab):
            v = per_class_map95[i] if i < len(per_class_map95) else float("nan")
            ap50_v = per_class_map[i] if (per_class_map is not None and i < len(per_class_map)) else float("nan")
            n_gt = gt_counts[i]
            all_classes.append((i, name, n_gt, ap50_v, v))

            if n_gt == 0:
                no_gt.append((i, name))
            elif isinstance(v, float) and (math.isnan(v) or v == 0.0):
                detect_fail.append((i, name, n_gt))
            else:
                eval_ok.append((i, name, float(v), n_gt))

        # CSV로 전체 저장 (GT count 포함)
        import csv
        csv_path = Path(model_path).parent / f"{Path(model_path).stem}_per_class.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "name", "gt_instances", "mAP50", "mAP50-95"])
            for idx, name, n_gt, ap50, ap95 in all_classes:
                w.writerow([idx, name, n_gt, f"{ap50:.4f}", f"{ap95:.4f}"])
        print(f"📄 per-class 결과 저장: {csv_path}")

        total_with_gt = len(eval_ok) + len(detect_fail)
        print()
        print(f"📈 클래스 분포:")
        print(f"   ✅ 검출 성공 (GT > 0, mAP > 0): {len(eval_ok):4d}개")
        print(f"   ❌ 검출 실패 (GT > 0, mAP = 0): {len(detect_fail):4d}개  ← 실제 모델 약점")
        print(f"   ⚪ 평가 불가 (GT = 0):           {len(no_gt):4d}개  ← val 데이터에 없는 클래스")
        print(f"   ────────────────────────")
        print(f"   total: {len(vocab)}")

        if eval_ok:
            mean_ok = sum(v for _, _, v, _ in eval_ok) / len(eval_ok)
            # GT 있는 클래스 전체 평균 (실패=0 포함, 진짜 KD 효과 지표)
            mean_with_gt = mean_ok * len(eval_ok) / total_with_gt
            print(f"\n🎯 평가 가능 클래스(GT > 0) 평균 mAP50-95:")
            print(f"   검출 성공만:           {mean_ok:.4f}  ({len(eval_ok)}개)")
            print(f"   GT 있는 전체:          {mean_with_gt:.4f}  ({total_with_gt}개, 검출실패=0 포함)")
            print(f"   (Overall 위 수치는 GT 없는 클래스도 0으로 평균에 포함되어 deflated)")

            # Top/Bottom 5 (GT count 함께)
            sorted_ec = sorted(eval_ok, key=lambda x: -x[2])
            print("\n    Top 5 (mAP50-95):")
            for i, n, v, g in sorted_ec[:5]:
                print(f"      [{i:4d}] {n[:30]:30s}  mAP={v:.4f}  GT={g}")
            print("    Bottom 5 (mAP50-95):")
            for i, n, v, g in sorted_ec[-5:]:
                print(f"      [{i:4d}] {n[:30]:30s}  mAP={v:.4f}  GT={g}")

        # 검출 실패 클래스는 처음 10개만 (LVIS에선 수백 개 가능)
        if detect_fail:
            print(f"\n❌ 검출 실패 클래스 ({len(detect_fail)}개) 앞 10개 (GT 많은 순):")
            for i, n, g in sorted(detect_fail, key=lambda x: -x[2])[:10]:
                print(f"      [{i:4d}] {n[:30]:30s}  GT={g}")
            print(f"    (전체 CSV: {csv_path})")

    if pynvml:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/app/weights/1M.pt")
    parser.add_argument("--data",  default="/app/configs/coco40.yaml")
    parser.add_argument("--vocab", default="/app/configs/vocab.txt")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf",  type=float, default=0.01)
    parser.add_argument("--iou",   type=float, default=0.6)
    args = parser.parse_args()

    benchmark(args.model, args.data, args.vocab,
              args.batch, args.imgsz, args.conf, args.iou)
