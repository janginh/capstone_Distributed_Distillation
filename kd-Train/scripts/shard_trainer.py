"""
=============================================================
[A5000] Shard Trainer (Rotating Storage)
- HDFS에서 N(=5)개 shard만 로컬에 유지하며 순환 학습
- Teacher(YOLO-World-x) 추론 → soft pseudo-label → Student(YOLO-World-s) KD
- DDP 4-GPU 실행은 scripts/launch_kd.py(torchrun)로 진입
- 디버그용 단일 GPU 실행: python scripts/shard_trainer.py --config configs/kd.yaml
=============================================================
"""

import os
import io
import tarfile
import glob
import shutil
import argparse
import yaml
from pathlib import Path

from hdfs import InsecureClient

from kd.distill import KDState
from kd import ddp as ddp_utils
import torch


class ShardTrainer:
    def __init__(
        self,
        cfg: dict,
        hdfs_url="http://localhost:9870",
        hdfs_path="/shards",
        local_dir="./working_shards",
        max_local_shards=5,
        prefetch=True,
    ):
        self.cfg = cfg
        self.hdfs_url = hdfs_url
        self.hdfs_path = hdfs_path
        self.local_dir = local_dir
        self.max_local_shards = max_local_shards
        self.prefetch = prefetch
        self.kd_state = None   # lazy init after DDP setup

        os.makedirs(local_dir, exist_ok=True)

    def connect_hdfs(self):
        print(f"🔌 HDFS 연결: {self.hdfs_url}")
        self.hdfs = InsecureClient(self.hdfs_url, user="root")
        print("✅ 연결 완료")

    def list_shards(self):
        """HDFS에 있는 전체 shard 목록"""
        files = self.hdfs.list(self.hdfs_path)
        shards = sorted([f for f in files if f.endswith(".tar")])
        return shards

    def pull_shard(self, shard_name):
        """HDFS → 로컬로 shard 다운로드"""
        hdfs_file = f"{self.hdfs_path}/{shard_name}"
        local_file = os.path.join(self.local_dir, shard_name)

        if os.path.exists(local_file):
            return local_file

        print(f"  ⬇️  HDFS → 로컬: {shard_name}", end="", flush=True)
        with self.hdfs.read(hdfs_file) as reader:
            with open(local_file, "wb") as f:
                while True:
                    chunk = reader.read(8 * 1024 * 1024)  # 8MB chunks
                    if not chunk:
                        break
                    f.write(chunk)

        size_mb = os.path.getsize(local_file) / (1024 * 1024)
        print(f" ({size_mb:.0f}MB)")
        return local_file

    def delete_local_shard(self, shard_name):
        """로컬에서 shard 삭제 (디스크 확보)"""
        local_file = os.path.join(self.local_dir, shard_name)
        if os.path.exists(local_file):
            os.remove(local_file)
            print(f"  🗑️  로컬 삭제: {shard_name}")

        # 언팩된 디렉토리도 삭제
        unpack_dir = os.path.join(self.local_dir, shard_name.replace(".tar", ""))
        if os.path.exists(unpack_dir):
            shutil.rmtree(unpack_dir)

    def unpack_shard(self, shard_path):
        """tar shard를 로컬 디렉토리로 언팩"""
        shard_name = os.path.basename(shard_path).replace(".tar", "")
        unpack_dir = os.path.join(self.local_dir, shard_name)

        if os.path.exists(unpack_dir):
            return unpack_dir

        os.makedirs(unpack_dir, exist_ok=True)
        with tarfile.open(shard_path, "r") as tar:
            tar.extractall(unpack_dir)

        image_count = len([
            f for f in os.listdir(unpack_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"  📦 언팩: {shard_name} ({image_count}장)")
        return unpack_dir

    def _load_vocab(self):
        with open(self.cfg["paths"]["vocab"], "r") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _ensure_kd_state(self):
        if self.kd_state is not None:
            return
        local_rank, _, _ = ddp_utils.setup_ddp(
            backend=self.cfg["ddp"]["backend"],
            master_addr=self.cfg["ddp"]["master_addr"],
            master_port=self.cfg["ddp"]["master_port"],
        )
        device = torch.device(f"cuda:{local_rank}")
        vocab = self._load_vocab()
        if ddp_utils.is_main():
            print(f"📖 vocab: {len(vocab)}개 클래스 로드")
        self.kd_state = KDState(self.cfg, vocab, device)

    def train_on_shard(self, image_dir, shard_name):
        """단일 shard에 대해 Teacher → Student KD 학습 (soft response + conf 필터)."""
        self._ensure_kd_state()
        if ddp_utils.is_main():
            num = len([
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"\n  🧠 KD 학습 시작: {shard_name} ({num}장)")
            print(f"     teacher={self.cfg['teacher']['weights']} "
                  f"student={self.cfg['student']['weights']} "
                  f"conf>{self.cfg['kd']['conf_threshold']}")
        n = self.kd_state.train_on_shard(image_dir, shard_name)
        if ddp_utils.is_main():
            print(f"     ✅ 완료: {shard_name}")
        return n

    def run(self):
        """메인 학습 루프 (DDP-aware): rank0 = HDFS I/O, 모든 rank = 학습."""
        # KD state 먼저 초기화 (DDP setup도 여기서)
        self._ensure_kd_state()

        if ddp_utils.is_main():
            self.connect_hdfs()
            all_shards = self.list_shards()
        else:
            all_shards = None

        # rank0 → 다른 rank로 shard 목록 broadcast
        if ddp_utils.get_world_size() > 1:
            import torch.distributed as dist
            obj = [all_shards]
            dist.broadcast_object_list(obj, src=0)
            all_shards = obj[0]

        total_shards = len(all_shards) if all_shards else 0
        if total_shards == 0:
            if ddp_utils.is_main():
                print("❌ HDFS에 shard가 없습니다!")
            return

        if ddp_utils.is_main():
            print(f"\n{'=' * 60}")
            print(f"🚀 KD Shard Trainer (world_size={ddp_utils.get_world_size()})")
            print(f"   HDFS shard: {total_shards}개")
            print(f"   로컬 유지: {self.max_local_shards}개")
            print(f"   Prefetch: {'ON' if self.prefetch else 'OFF'}")
            print(f"{'=' * 60}\n")

            # 초기 로딩: rank0만
            initial_count = min(self.max_local_shards, total_shards)
            print(f"📥 초기 로딩: {initial_count}개 shard")
            for i in range(initial_count):
                self.pull_shard(all_shards[i])
        ddp_utils.barrier()

        total_images = 0
        for idx, shard_name in enumerate(all_shards):
            if ddp_utils.is_main():
                print(f"\n{'─' * 40}")
                print(f"📋 [{idx + 1}/{total_shards}] {shard_name}")
                shard_path = os.path.join(self.local_dir, shard_name)
                if not os.path.exists(shard_path):
                    self.pull_shard(shard_name)
                image_dir = self.unpack_shard(shard_path)

                if self.prefetch:
                    next_idx = idx + self.max_local_shards
                    if next_idx < total_shards:
                        print(f"  ⏩ Prefetch: {all_shards[next_idx]}")
                        self.pull_shard(all_shards[next_idx])
            ddp_utils.barrier()

            # 모든 rank가 같은 unpack_dir 읽음
            image_dir = os.path.join(self.local_dir, shard_name.replace(".tar", ""))
            images_trained = self.train_on_shard(image_dir, shard_name)
            total_images += images_trained
            ddp_utils.barrier()

            if ddp_utils.is_main():
                self.delete_local_shard(shard_name)
                # 1 epoch 가정: HDFS 원본도 즉시 삭제 (디스크 누적 방지)
                if self.cfg["hdfs"].get("delete_after_consume", False):
                    try:
                        self.hdfs.delete(f"{self.hdfs_path}/{shard_name}")
                        print(f"  🗑️  HDFS 삭제: {shard_name}")
                    except Exception as e:
                        print(f"  ⚠️  HDFS 삭제 실패 {shard_name}: {e}")
            ddp_utils.barrier()

        if ddp_utils.is_main():
            print(f"\n{'=' * 60}")
            print(f"🎉 1 Epoch 완료!")
            print(f"   처리 shard: {total_shards}개")
            print(f"   처리 이미지(rank별 합산): {total_images:,}장")
            print(f"{'=' * 60}")

        ddp_utils.cleanup_ddp()


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kd.yaml")
    parser.add_argument("--hdfs", default=None, help="override cfg.hdfs.url")
    parser.add_argument("--hdfs_path", default=None)
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--no_prefetch", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    trainer = ShardTrainer(
        cfg=cfg,
        hdfs_url=args.hdfs or cfg["hdfs"]["url"],
        hdfs_path=args.hdfs_path or cfg["hdfs"]["shard_path"],
        local_dir=args.local_dir or cfg["paths"]["local_shard_dir"],
        max_local_shards=args.max_shards or cfg["hdfs"]["max_local_shards"],
        prefetch=(not args.no_prefetch) and cfg["hdfs"]["prefetch"],
    )
    trainer.run()
