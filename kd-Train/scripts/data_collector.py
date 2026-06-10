"""
=============================================================
[3060] Data Collector → Kafka Producer
- 웹 데이터 자동 다운로드 → A5000 Kafka로 이미지 전송
- 데이터 소스: HF streaming / CC3M URLs / 커스텀 URLs / 블랙박스 영상 / 폴더
=============================================================
사용법:
  # HuggingFace 데이터셋 streaming (디스크 캐싱 없음) — 추천
  python scripts/data_collector.py --mode hf \
      --hf_dataset pixparse/cc12m-wds --max_count 1000 --kafka <A5000_IP>:9092

  # CC3M URL TSV
  python scripts/data_collector.py --mode cc3m --cc3m_tsv cc3m.tsv --kafka <A5000_IP>:9092

  # 커스텀 URL 리스트
  python scripts/data_collector.py --mode urls --url_file urls.txt --kafka <A5000_IP>:9092

  # 블랙박스 영상 (최종 시연)
  python scripts/data_collector.py --mode video --video_path dashcam.mp4 --kafka <A5000_IP>:9092

  # 이미지 폴더
  python scripts/data_collector.py --mode folder --image_dir ./images --kafka <A5000_IP>:9092
"""

import os
import io
import csv
import json
import time
import base64
import argparse
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from kafka import KafkaProducer
from PIL import Image


class DataCollector:
    def __init__(self, kafka_bootstrap, topic="raw-images", workers=4):
        self.kafka_bootstrap = kafka_bootstrap
        self.topic = topic
        self.workers = workers

        self.stats_lock = Lock()
        self.stats = {"sent": 0, "failed": 0, "skipped": 0}

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            max_request_size=20 * 1024 * 1024,
            buffer_memory=64 * 1024 * 1024,
            batch_size=262144,
            linger_ms=50,
            compression_type="lz4",
            acks=1,
        )

    def _image_to_b64(self, img, max_size=640):
        """PIL Image → base64 (리사이즈 포함)"""
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _send_image(self, image_b64, filename, device_id, domain):
        """이미지 1장을 Kafka로 전송"""
        msg = {
            "device_id": device_id,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "image_b64": image_b64,
        }
        self.producer.send(self.topic, key=device_id, value=msg)

    def _download_and_send(self, url, idx, device_id, domain):
        """URL에서 이미지 다운로드 → Kafka 전송"""
        try:
            resp = requests.get(url, timeout=10, stream=True)
            if resp.status_code != 200:
                with self.stats_lock:
                    self.stats["failed"] += 1
                return

            img = Image.open(io.BytesIO(resp.content)).convert("RGB")

            # 너무 작은 이미지 스킵
            if img.width < 64 or img.height < 64:
                with self.stats_lock:
                    self.stats["skipped"] += 1
                return

            image_b64 = self._image_to_b64(img)
            filename = f"web_{idx:08d}.jpg"
            self._send_image(image_b64, filename, device_id, domain)

            with self.stats_lock:
                self.stats["sent"] += 1

        except Exception:
            with self.stats_lock:
                self.stats["failed"] += 1

    def _print_progress(self, total, start_time):
        """진행률 출력"""
        elapsed = time.time() - start_time
        done = self.stats["sent"] + self.stats["failed"] + self.stats["skipped"]
        rate = self.stats["sent"] / elapsed if elapsed > 0 else 0
        pct = done / total * 100 if total > 0 else 0
        print(
            f"\r  📊 {pct:5.1f}% | "
            f"전송:{self.stats['sent']:,} "
            f"실패:{self.stats['failed']:,} "
            f"스킵:{self.stats['skipped']:,} | "
            f"{rate:.0f} img/s",
            end="", flush=True,
        )

    # ==================== Mode: CC3M ====================

    def collect_cc3m(self, tsv_path, device_id="cc3m", domain="web", max_count=None):
        """CC3M TSV에서 URL 읽어서 다운로드 → Kafka"""
        print(f"📂 CC3M TSV 로딩: {tsv_path}")
        urls = []
        with open(tsv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    urls.append(row[1])  # CC3M: caption \t url
                if max_count and len(urls) >= max_count:
                    break

        total = len(urls)
        print(f"   URL: {total:,}개")
        self._batch_download(urls, total, device_id, domain)

    # ==================== Mode: Custom URLs ====================

    def collect_urls(self, url_file, device_id="custom", domain="web", max_count=None):
        """URL 리스트 파일 → 다운로드 → Kafka"""
        print(f"📂 URL 파일 로딩: {url_file}")
        with open(url_file, "r") as f:
            urls = [line.strip() for line in f if line.strip().startswith("http")]
        if max_count:
            urls = urls[:max_count]

        total = len(urls)
        print(f"   URL: {total:,}개")
        self._batch_download(urls, total, device_id, domain)

    def _batch_download(self, urls, total, device_id, domain):
        """URL 리스트를 멀티스레드로 다운로드"""
        start_time = time.time()
        print(f"🚀 다운로드 시작 (워커 {self.workers}개)\n")

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for idx, url in enumerate(urls):
                future = executor.submit(
                    self._download_and_send, url, idx, device_id, domain
                )
                futures.append(future)

            for future in as_completed(futures):
                future.result()
                self._print_progress(total, start_time)

        self.producer.flush()
        self._print_summary(start_time)

    # ==================== Mode: Video (블랙박스) ====================

    def collect_video(self, video_path, device_id="dashcam", domain="driving",
                      fps_sample=2, max_frames=None):
        """
        블랙박스 영상 → 프레임 추출 → Kafka
        fps_sample: 초당 추출할 프레임 수 (원본 30fps에서 2fps만 추출 등)
        """
        import cv2

        print(f"🎥 영상 로딩: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ 영상을 열 수 없습니다!")
            return

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / orig_fps if orig_fps > 0 else 0
        frame_interval = int(orig_fps / fps_sample) if fps_sample > 0 else 1

        print(f"   원본: {orig_fps:.0f}fps, {total_frames:,}프레임, {duration:.0f}초")
        print(f"   추출: {fps_sample}fps (매 {frame_interval}프레임마다)")

        start_time = time.time()
        frame_idx = 0
        extracted = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # OpenCV BGR → PIL RGB
                img = Image.fromarray(frame[:, :, ::-1])
                image_b64 = self._image_to_b64(img)
                filename = f"frame_{frame_idx:08d}.jpg"
                self._send_image(image_b64, filename, device_id, domain)

                extracted += 1
                self.stats["sent"] += 1

                if extracted % 50 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"\r  🎬 프레임 {extracted:,}장 추출 | "
                        f"{elapsed:.0f}s 경과",
                        end="", flush=True,
                    )

                if max_frames and extracted >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        self.producer.flush()
        print(f"\n  ✅ 총 {extracted:,}프레임 추출 → Kafka 전송 완료")
        self._print_summary(start_time)

    # ==================== Mode: HuggingFace streaming ====================

    def collect_hf_stream(self, dataset_name, split="train", device_id="hf",
                          domain="web", max_count=None, hf_token=None,
                          image_field="jpg",
                          start_idx=0, end_idx=None,
                          checkpoint_file=None, checkpoint_every=500):
        """
        HuggingFace 데이터셋을 streaming 모드로 받아 Kafka로 전송 (디스크 캐싱 없음).
        - webdataset 형식(pixparse/cc12m-wds 등)은 jpg 필드에 PIL Image 자동 디코딩
        - image_field가 None이거나 키가 없으면 ('jpg','image','img','picture','png') 순서로 탐색

        병렬/Resume:
        - start_idx, end_idx: 데이터셋의 절대 인덱스 범위로 슬라이싱 (병렬 분할용)
        - checkpoint_file: 매 checkpoint_every마다 현재 idx를 저장. 재시작 시 자동 복원
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ 'datasets' 라이브러리가 필요합니다: pip install datasets")
            return

        # checkpoint 복원 (있으면 start_idx 갱신)
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r") as f:
                    resumed = int(f.read().strip())
                if resumed > start_idx:
                    print(f"📁 checkpoint 복원: idx={resumed} (요청 start_idx={start_idx})")
                    start_idx = resumed
            except Exception as e:
                print(f"⚠️  checkpoint 읽기 실패 ({e}), 처음부터 시작")

        print(f"📦 HF streaming: {dataset_name} (split={split})")
        print(f"   range: [{start_idx}, {end_idx if end_idx is not None else '∞'})")
        print(f"   max_count: {max_count or '제한 없음'}")
        if checkpoint_file:
            print(f"   checkpoint: {checkpoint_file} (every {checkpoint_every})")

        kwargs = {"streaming": True, "split": split}
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset(dataset_name, **kwargs)

        if start_idx > 0:
            print(f"⏩ {start_idx:,}개 항목 skip 중...")
            ds = ds.skip(start_idx)

        print(f"🚀 시작\n")
        start_time = time.time()
        sent = 0
        cur_idx = start_idx
        candidate_fields = [image_field, "jpg", "image", "img", "picture", "png"]

        for item in ds:
            if end_idx is not None and cur_idx >= end_idx:
                break
            cur_idx += 1

            try:
                img = None
                for k in candidate_fields:
                    if k and k in item and item[k] is not None:
                        img = item[k]
                        break

                if not isinstance(img, Image.Image):
                    with self.stats_lock:
                        self.stats["failed"] += 1
                else:
                    img = img.convert("RGB")
                    if img.width < 64 or img.height < 64:
                        with self.stats_lock:
                            self.stats["skipped"] += 1
                    else:
                        image_b64 = self._image_to_b64(img)
                        filename = f"hf_{cur_idx:010d}.jpg"
                        self._send_image(image_b64, filename, device_id, domain)
                        with self.stats_lock:
                            self.stats["sent"] += 1
                        sent += 1
            except Exception:
                with self.stats_lock:
                    self.stats["failed"] += 1

            if sent and sent % 25 == 0:
                elapsed = time.time() - start_time
                rate = sent / elapsed if elapsed > 0 else 0
                print(
                    f"\r  📊 idx={cur_idx:,} sent={self.stats['sent']:,} "
                    f"fail={self.stats['failed']:,} skip={self.stats['skipped']:,} | "
                    f"{rate:.0f} img/s",
                    end="", flush=True,
                )

            # 주기적 checkpoint 저장 (atomic write)
            if checkpoint_file and sent and sent % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_file, cur_idx)

            if max_count and sent >= max_count:
                break

        # 종료 시 마지막 checkpoint
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, cur_idx)

        self.producer.flush()
        self._print_summary(start_time)

    @staticmethod
    def _save_checkpoint(path, idx):
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(str(idx))
            os.replace(tmp, path)
        except Exception:
            pass

    # ==================== Mode: Video folder (여러 영상) ====================

    def collect_video_folder(self, video_dir, device_id="videos", domain="video",
                             fps_sample=2, max_frames_per_video=None):
        """
        폴더 안의 모든 영상(.mp4/.avi/.mov/.mkv/.webm)을 순차 처리.
        Kafka producer는 단일 인스턴스 유지하면서 영상별로 frame 추출 → Kafka.
        영상마다 device_id에 영상명 접미사 붙여서 추적 가능.
        """
        import cv2

        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        videos = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if Path(f).suffix.lower() in exts
        ])
        if not videos:
            print(f"❌ 영상 파일이 없음: {video_dir}")
            return

        print(f"📂 영상 폴더: {video_dir}")
        print(f"   총 영상 수: {len(videos)}개")
        for v in videos:
            print(f"   - {os.path.basename(v)}")

        overall_start = time.time()
        total_frames = 0

        for vi, video_path in enumerate(videos):
            vname = os.path.basename(video_path)
            print(f"\n{'='*50}")
            print(f"🎥 [{vi+1}/{len(videos)}] {vname}")
            print(f"{'='*50}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"   ⚠️  열기 실패, 스킵")
                continue

            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = n_total / orig_fps if orig_fps > 0 else 0
            interval = max(int(round(orig_fps / fps_sample)) if fps_sample > 0 else 1, 1)
            print(f"   원본: {orig_fps:.0f}fps, {n_total:,}프레임, {duration:.0f}초")
            print(f"   추출: {fps_sample}fps (매 {interval}프레임)")

            vstart = time.time()
            frame_idx = 0
            extracted = 0
            video_did = f"{device_id}_{Path(vname).stem}"

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    img = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
                    image_b64 = self._image_to_b64(img)
                    filename = f"{Path(vname).stem}_f{frame_idx:08d}.jpg"
                    self._send_image(image_b64, filename, video_did, domain)
                    extracted += 1
                    with self.stats_lock:
                        self.stats["sent"] += 1

                    if extracted % 100 == 0:
                        el = time.time() - vstart
                        print(f"\r   📊 {extracted:,}장 추출 ({el:.0f}s)",
                              end="", flush=True)

                    if max_frames_per_video and extracted >= max_frames_per_video:
                        break

                frame_idx += 1

            cap.release()
            total_frames += extracted
            print(f"\n   ✅ {vname}: {extracted:,}장")

        self.producer.flush()
        elapsed = time.time() - overall_start
        print(f"\n{'='*50}")
        print(f"🎉 영상 폴더 처리 완료")
        print(f"   처리 영상: {len(videos)}개")
        print(f"   추출 프레임: {total_frames:,}장")
        print(f"   소요 시간: {elapsed:.0f}초")
        print(f"{'='*50}")
        self.producer.close()

    # ==================== Mode: Folder ====================

    def collect_folder(self, image_dir, device_id="local", domain="general"):
        """로컬 이미지 폴더 → Kafka"""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted([
            os.path.join(root, f)
            for root, _, fnames in os.walk(image_dir)
            for f in fnames if Path(f).suffix.lower() in extensions
        ])

        total = len(files)
        print(f"📂 이미지 폴더: {image_dir} ({total:,}장)")
        start_time = time.time()

        for idx, fpath in enumerate(files):
            try:
                img = Image.open(fpath).convert("RGB")
                image_b64 = self._image_to_b64(img)
                self._send_image(image_b64, os.path.basename(fpath), device_id, domain)
                self.stats["sent"] += 1
            except Exception:
                self.stats["failed"] += 1

            if (idx + 1) % 100 == 0:
                self._print_progress(total, start_time)

        self.producer.flush()
        self._print_summary(start_time)

    # ==================== Util ====================

    def _print_summary(self, start_time):
        elapsed = time.time() - start_time
        rate = self.stats["sent"] / elapsed if elapsed > 0 else 0
        print(f"\n\n{'=' * 50}")
        print(f"🎉 수집 완료!")
        print(f"   전송: {self.stats['sent']:,}장")
        print(f"   실패: {self.stats['failed']:,}장")
        print(f"   스킵: {self.stats['skipped']:,}장")
        print(f"   속도: {rate:.0f} img/s")
        print(f"   시간: {elapsed:.0f}초")
        print(f"{'=' * 50}")
        self.producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Collector → Kafka")
    parser.add_argument("--mode", required=True,
                        choices=["cc3m", "urls", "video", "video_folder", "folder", "hf"])
    parser.add_argument("--kafka", required=True, help="A5000 Kafka 주소 (예: 192.168.0.10:9092)")
    parser.add_argument("--topic", default="raw-images")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device_id", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--max_count", type=int, default=None)

    # CC3M
    parser.add_argument("--cc3m_tsv", default=None)
    # Custom URLs
    parser.add_argument("--url_file", default=None)
    # Video (single file)
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--fps_sample", type=int, default=2)
    parser.add_argument("--max_frames_per_video", type=int, default=None,
                        help="영상 1개당 최대 추출 프레임 수 (None=제한 없음)")
    # Video folder
    parser.add_argument("--video_dir", default=None,
                        help="여러 영상이 있는 폴더 (재귀 X, 단일 디렉토리만)")
    # Image folder
    parser.add_argument("--image_dir", default=None)
    # HuggingFace streaming
    parser.add_argument("--hf_dataset", default=None, help="예: pixparse/cc12m-wds")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_token", default=None, help="HF auth token (필요시)")
    parser.add_argument("--hf_image_field", default="jpg",
                        help="이미지 필드명 (webdataset류=jpg, 일반=image)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="HF 데이터셋에서 skip할 항목 수 (병렬 분할용)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="멈출 항목 인덱스 (배타적, 병렬 분할용)")
    parser.add_argument("--checkpoint_file", default=None,
                        help="진행 상태 저장/복원 파일 (예: ckpt_0.txt)")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="checkpoint 저장 주기 (이미지 수)")

    args = parser.parse_args()

    collector = DataCollector(args.kafka, args.topic, args.workers)

    if args.mode == "cc3m":
        collector.collect_cc3m(
            args.cc3m_tsv, args.device_id or "cc3m",
            args.domain or "web", args.max_count,
        )
    elif args.mode == "urls":
        collector.collect_urls(
            args.url_file, args.device_id or "custom",
            args.domain or "web", args.max_count,
        )
    elif args.mode == "video":
        collector.collect_video(
            args.video_path, args.device_id or "dashcam",
            args.domain or "driving", args.fps_sample, args.max_count,
        )
    elif args.mode == "video_folder":
        collector.collect_video_folder(
            video_dir=args.video_dir,
            device_id=args.device_id or "videos",
            domain=args.domain or "video",
            fps_sample=args.fps_sample,
            max_frames_per_video=args.max_frames_per_video,
        )
    elif args.mode == "folder":
        collector.collect_folder(
            args.image_dir, args.device_id or "local",
            args.domain or "general",
        )
    elif args.mode == "hf":
        collector.collect_hf_stream(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            device_id=args.device_id or "hf",
            domain=args.domain or "web",
            max_count=args.max_count,
            hf_token=args.hf_token,
            image_field=args.hf_image_field,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            checkpoint_file=args.checkpoint_file,
            checkpoint_every=args.checkpoint_every,
        )
