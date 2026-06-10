"""
DDP 진입점. torchrun으로 실행:

  torchrun --standalone --nproc_per_node=4 scripts/launch_kd.py \
      --config configs/kd.yaml

내부적으로 shard_trainer.ShardTrainer를 실행 (rank별 1 인스턴스).
"""
import argparse
import os
import sys
import yaml

# scripts/를 path에 추가 (절대 import 안정화)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from shard_trainer import ShardTrainer, load_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kd.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    trainer = ShardTrainer(
        cfg=cfg,
        hdfs_url=cfg["hdfs"]["url"],
        hdfs_path=cfg["hdfs"]["shard_path"],
        local_dir=cfg["paths"]["local_shard_dir"],
        max_local_shards=cfg["hdfs"]["max_local_shards"],
        prefetch=cfg["hdfs"]["prefetch"],
    )
    trainer.run()


if __name__ == "__main__":
    main()
