"""
=============================================================
[A5000] Shard Builder
- Kafka에서 이미지 수신 → 1GB tar shard로 패킹 → HDFS 저장
- 실행: python scripts/shard_builder.py --kafka <IP>:9092
=============================================================
"""

import os
import io
import json
import tarfile
import time
import base64
import signal
import argparse
from datetime import datetime

from kafka import KafkaConsumer
from hdfs import InsecureClient


class ShardBuilder:
    def __init__(self, kafka_bootstrap, hdfs_url, shard_size_gb=1.0, hdfs_path="/shards"):
        self.kafka_bootstrap = kafka_bootstrap
        self.hdfs_url = hdfs_url
        self.shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)
        self.hdfs_path = hdfs_path

        self.running = True
        self.shard_count = 0
        self.total_images = 0

        # 현재 shard 버퍼
        self.current_buffer = io.BytesIO()
        self.current_tar = None
        self.current_size = 0
        self.images_in_shard = 0

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        print("\n⏹️  종료 신호 수신...")
        self.running = False

    def connect(self):
        """Kafka + HDFS 연결"""
        print(f"🔌 Kafka 연결: {self.kafka_bootstrap}")
        self.consumer = KafkaConsumer(
            "raw-images",
            bootstrap_servers=self.kafka_bootstrap,
            group_id="shard-builder-group",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            max_partition_fetch_bytes=20 * 1024 * 1024,
            consumer_timeout_ms=10000,
        )
        print("✅ Kafka 연결 완료")

        print(f"🔌 HDFS 연결: {self.hdfs_url}")
        self.hdfs = InsecureClient(self.hdfs_url, user="root")
        self.hdfs.makedirs(self.hdfs_path)
        print("✅ HDFS 연결 완료")

    def _new_shard(self):
        """새 shard 시작"""
        self.current_buffer = io.BytesIO()
        self.current_tar = tarfile.open(fileobj=self.current_buffer, mode="w")
        self.current_size = 0
        self.images_in_shard = 0

    def _add_image(self, filename, image_bytes, metadata=None):
        """현재 shard에 이미지 추가"""
        # 이미지 파일 추가
        img_info = tarfile.TarInfo(name=filename)
        img_info.size = len(image_bytes)
        img_info.mtime = time.time()
        self.current_tar.addfile(img_info, io.BytesIO(image_bytes))
        self.current_size += len(image_bytes)
        self.images_in_shard += 1

        # 메타데이터 JSON도 같이 저장
        if metadata:
            meta_json = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
            meta_name = filename.rsplit(".", 1)[0] + ".json"
            meta_info = tarfile.TarInfo(name=meta_name)
            meta_info.size = len(meta_json)
            meta_info.mtime = time.time()
            self.current_tar.addfile(meta_info, io.BytesIO(meta_json))
            self.current_size += len(meta_json)

    def _flush_shard(self):
        """현재 shard를 HDFS에 저장"""
        if self.images_in_shard == 0:
            return

        self.current_tar.close()
        shard_data = self.current_buffer.getvalue()

        self.shard_count += 1
        shard_name = f"shard{self.shard_count:04d}.tar"
        hdfs_full_path = f"{self.hdfs_path}/{shard_name}"

        # HDFS에 업로드
        with self.hdfs.write(hdfs_full_path, overwrite=True) as writer:
            writer.write(shard_data)

        size_mb = len(shard_data) / (1024 * 1024)
        self.total_images += self.images_in_shard
        print(
            f"  💾 {shard_name} → HDFS ({size_mb:.0f}MB, "
            f"{self.images_in_shard}장, 누적 {self.total_images:,}장)"
        )

    def run(self):
        """메인 루프: Kafka → tar shard → HDFS"""
        self.connect()
        self._new_shard()

        print(f"\n{'=' * 60}")
        print(f"🚀 Shard Builder 시작")
        print(f"   shard 크기: {self.shard_size_bytes / (1024**3):.1f}GB")
        print(f"   HDFS 경로: {self.hdfs_path}")
        print(f"{'=' * 60}\n")

        while self.running:
            try:
                messages = self.consumer.poll(timeout_ms=3000)

                for tp, records in messages.items():
                    for record in records:
                        msg = record.value
                        filename = msg.get("filename", f"img_{self.total_images:08d}.jpg")
                        image_b64 = msg.get("image_b64", "")

                        if not image_b64:
                            continue

                        image_bytes = base64.b64decode(image_b64)

                        metadata = {
                            "device_id": msg.get("device_id"),
                            "domain": msg.get("domain"),
                            "timestamp": msg.get("timestamp"),
                            "original_size": len(image_bytes),
                        }

                        self._add_image(filename, image_bytes, metadata)

                        # shard 크기 초과 시 HDFS로 flush
                        if self.current_size >= self.shard_size_bytes:
                            self._flush_shard()
                            self._new_shard()

            except Exception as e:
                if self.running:
                    print(f"⚠️  오류: {e}")

        # 남은 데이터 flush
        self._flush_shard()
        self.consumer.close()

        print(f"\n{'=' * 60}")
        print(f"📊 Shard Builder 완료")
        print(f"   총 shard: {self.shard_count}개")
        print(f"   총 이미지: {self.total_images:,}장")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", default="localhost:9092")
    parser.add_argument("--hdfs", default="http://localhost:9870")
    parser.add_argument("--shard_size", type=float, default=1.0, help="shard 크기 (GB)")
    parser.add_argument("--hdfs_path", default="/shards")
    args = parser.parse_args()

    builder = ShardBuilder(args.kafka, args.hdfs, args.shard_size, args.hdfs_path)
    builder.run()
