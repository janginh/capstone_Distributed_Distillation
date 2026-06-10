"""
shard 한 개에 대한 KD 학습 루프.
- DataLoader(DistributedSampler) 로 unlabeled 이미지 배치 공급
- 각 step:
    1) Teacher forward (no_grad) → soft pseudo-label
    2) build_batch_from_teacher → v8 loss용 batch
    3) Student forward (AMP) → SoftKDDetectionLoss
    4) [선택] Reference forward (no_grad) → consistency loss (forgetting 방지)
    5) backward + step
- 5 shard마다 checkpoint (main rank only)

Forgetting 방지 (configs/kd.yaml):
  freeze_backbone: backbone 동결 (visual feature 보존)
  use_reference:   원본 student를 reference로, 멀어지지 않게 제약
"""
import os
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from .dataset import ShardImageDataset, collate
from .teacher import TeacherWorld
from .student import StudentWorld
from .losses import SoftKDDetectionLoss, build_batch_from_teacher
from . import ddp as ddp_utils


# ============================================================
# 전역 상태 (1 epoch = HDFS 전체 1바퀴 동안 student/optim 유지)
# ============================================================
class KDState:
    """rank별 1개 인스턴스. shard 간 student/optimizer 상태 유지."""

    def __init__(self, cfg: dict, vocab: List[str], device: torch.device):
        self.cfg = cfg
        self.vocab = vocab
        self.device = device
        self.global_step = 0
        self.shards_done = 0

        # ----- Teacher / Student -----
        self.teacher = TeacherWorld(
            weights=cfg["teacher"]["weights"],
            vocab=vocab,
            device=device,
            fp16=cfg["teacher"].get("fp16", True),
            conf_threshold=cfg["kd"]["conf_threshold"],
            iou_threshold=cfg["kd"]["nms_iou_threshold"],
            max_dets=cfg["kd"]["max_dets_per_image"],
        )
        student = StudentWorld(
            weights=cfg["student"]["weights"], vocab=vocab, device=device,
            freeze_backbone=cfg["kd"].get("freeze_backbone", False),
            freeze_layers=cfg["kd"].get("freeze_layers", 10),
        )

        # ----- Reference model (B: forgetting 방지) -----
        # 원본 student를 frozen reference로 보관, student가 너무 멀어지지 않게 제약
        self.reference = None
        self.reference_weight = float(cfg["kd"].get("reference_weight", 0.0))
        self.reference_loss_type = cfg["kd"].get("reference_loss", "feature")
        if cfg["kd"].get("use_reference", False) and self.reference_weight > 0:
            if ddp_utils.is_main():
                print(f"📚 Reference 모델 로드 (forgetting 방지)")
                print(f"   weight λ = {self.reference_weight}")
                print(f"   loss type = {self.reference_loss_type}")
            from ultralytics import YOLOWorld
            ref_wrapper = YOLOWorld(cfg["student"]["weights"])
            ref_wrapper.set_classes(vocab)
            self.reference = ref_wrapper.model.to(device)
            # train 모드(raw feature 출력) but BN은 eval (running stat 고정)
            self.reference.train()
            for m in self.reference.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for p in self.reference.parameters():
                p.requires_grad = False

        # hyp 주입 (SoftKDDetectionLoss → v8 base init이 self.hyp 사용)
        from types import SimpleNamespace
        student.net.args = SimpleNamespace(
            box=cfg["kd"]["hyp"]["box"],
            cls=cfg["kd"]["hyp"]["cls"],
            dfl=cfg["kd"]["hyp"]["dfl"],
        )

        # DDP wrap
        # find_unused_parameters=True: YOLO-World의 CLIP 텍스트 인코더는 KD loss path에
        # 포함되지 않아 gradient를 못 받음 → DDP가 reduction을 못 끝내는 에러 방지.
        # 텍스트 인코더는 frozen 취급이라고 보면 됨 (성능 페널티 미미).
        if ddp_utils.get_world_size() > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.student_ddp = nn.parallel.DistributedDataParallel(
                student.net, device_ids=[local_rank], find_unused_parameters=True
            )
            base_for_loss = student.net   # loss는 unwrap된 모델 기준
        else:
            self.student_ddp = student.net
            base_for_loss = student.net

        # ----- Loss -----
        cls_loss_type = cfg["kd"].get("cls_loss_type", "bce")
        temperature = cfg["kd"].get("temperature", 1.0)
        self.criterion = SoftKDDetectionLoss(
            base_for_loss,
            cls_loss_type=cls_loss_type,
            temperature=temperature,
        )
        if ddp_utils.is_main():
            print(f"📐 cls loss: {cls_loss_type}" +
                  (f" (T={temperature})" if cls_loss_type == "kl" else ""))

        # ----- Optimizer -----
        opt_cfg = cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            student.net.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        self.scaler = GradScaler(enabled=cfg["student"].get("use_amp", True))
        self.warmup_steps = cfg["scheduler"]["warmup_steps"]
        self.base_lr = opt_cfg["lr"]
        self.grad_clip = cfg["train"].get("grad_clip", 10.0)

        # ----- ckpt dir -----
        self.ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
        if ddp_utils.is_main():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    def _lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        return self.base_lr

    def _apply_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # --------------------------------------------------------
    def train_on_shard(self, image_dir: str, shard_name: str) -> int:
        cfg = self.cfg
        ds = ShardImageDataset(image_dir, imgsz=cfg["data"]["imgsz"])

        if ddp_utils.get_world_size() > 1:
            sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        else:
            sampler = None

        loader = DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"].get("pin_memory", True),
            collate_fn=collate,
            drop_last=True,
            persistent_workers=cfg["data"]["num_workers"] > 0,
        )

        self.student_ddp.train()
        if sampler is not None:
            sampler.set_epoch(self.shards_done)

        log_every = cfg["train"]["log_every"]
        nc = len(self.vocab)
        imgsz = cfg["data"]["imgsz"]
        t0 = time.time()

        for step, batch in enumerate(loader):
            imgs = batch["image"].to(self.device, non_blocking=True)

            # 1) Teacher forward → soft pseudo-label
            teacher_outs = self.teacher(imgs)
            kd_batch = build_batch_from_teacher(teacher_outs, imgsz, self.device, nc)

            # 빈 배치 스킵 — DDP 동기화 필수.
            # 어느 한 rank라도 pseudo=0이면 모든 rank가 함께 skip해야 forward 호출 횟수가 일치.
            # (rank 간 desync 시 ALLREDUCE timeout 발생)
            local_no_data = kd_batch["bboxes"].shape[0] == 0
            if ddp_utils.get_world_size() > 1:
                flag = torch.tensor(
                    [1 if local_no_data else 0],
                    device=self.device, dtype=torch.int32,
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                must_skip = bool(flag.item())
            else:
                must_skip = local_no_data

            if must_skip:
                if ddp_utils.is_main() and (step + 1) % log_every == 0:
                    print(
                        f"  [shard {shard_name} step {step+1}/{len(loader)}] "
                        f"skip (some rank pseudo=0)"
                    )
                continue

            # 2) lr warmup
            lr = self._lr_at(self.global_step)
            self._apply_lr(lr)

            # 3) Student forward + KD loss
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["student"].get("use_amp", True)):
                preds = self.student_ddp(imgs)
                loss_kd, loss_items = self.criterion(preds, kd_batch)

                # 4) Reference consistency loss (forgetting 방지)
                loss_ref = torch.tensor(0.0, device=self.device)
                if self.reference is not None:
                    with torch.no_grad():
                        ref_preds = self.reference(imgs)
                    loss_ref = self._reference_loss(preds, ref_preds)
                    loss = loss_kd + self.reference_weight * loss_ref
                else:
                    loss = loss_kd

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student_ddp.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1

            if ddp_utils.is_main() and (step + 1) % log_every == 0:
                el = time.time() - t0
                lb, lc, ld = loss_items.tolist()
                num_pseudo = int(kd_batch["bboxes"].shape[0])
                ref_str = ""
                if self.reference is not None:
                    ref_str = f" ref={loss_ref.item():.3f}"
                print(
                    f"  [shard {shard_name} step {step+1}/{len(loader)}] "
                    f"loss={loss.item():.3f} (kd={loss_kd.item():.3f}{ref_str} "
                    f"box={lb:.3f} cls={lc:.3f} dfl={ld:.3f}) "
                    f"pseudo={num_pseudo} lr={lr:.2e} ({el:.0f}s)"
                )

        self.shards_done += 1
        ddp_utils.barrier()

        # 5 shard마다 checkpoint
        if (
            self.shards_done % self.cfg["train"]["ckpt_every_n_shards"] == 0
            and ddp_utils.is_main()
        ):
            self._save_ckpt(epoch=self.shards_done // self.cfg["train"]["ckpt_every_n_shards"])

        return len(ds)

    # --------------------------------------------------------
    def _reference_loss(self, student_preds, ref_preds) -> torch.Tensor:
        """
        원본 student와 현재 student의 출력 차이 측정.
        - 'feature': 각 scale의 feature map 간 MSE (가장 안전, backbone 잘하던 거 보존)
        - 'logit':   class logit (마지막 nc 채널) 간 KL divergence (소프트 출력 보존)
        """
        # 둘 다 list[Tensor] (multi-scale features). 길이/shape 일치해야 함.
        if not (isinstance(student_preds, (list, tuple))
                and isinstance(ref_preds, (list, tuple))):
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        n = 0
        for s, r in zip(student_preds, ref_preds):
            r = r.float().detach()
            s = s.float()
            if self.reference_loss_type == "logit":
                # 마지막 nc 채널이 class logit, 앞 4*reg_max는 box dfl
                # YOLOv8 channel layout: [reg*4, cls*nc]
                nc = len(self.vocab)
                s_cls = s[:, -nc:]
                r_cls = r[:, -nc:]
                # 두 분포에 대한 sigmoid → BCE
                loss = loss + F.binary_cross_entropy_with_logits(
                    s_cls, r_cls.sigmoid()
                )
            else:
                # default: feature MSE
                loss = loss + F.mse_loss(s, r)
            n += 1
        return loss / max(n, 1)

    # --------------------------------------------------------
    def _save_ckpt(self, epoch: int):
        path = self.ckpt_dir / f"student_e{epoch:03d}.pt"
        model = (
            self.student_ddp.module
            if isinstance(self.student_ddp, nn.parallel.DistributedDataParallel)
            else self.student_ddp
        )
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "shards_done": self.shards_done,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "vocab": self.vocab,
        }
        torch.save(payload, path)
        # latest 별칭 (항상 최신을 가리킴)
        latest = self.ckpt_dir / "student_latest.pt"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            torch.save(payload, latest)
        except Exception:
            pass
        print(f"  💾 checkpoint → {path}")

        # 오래된 checkpoint 정리 (최근 N개만 유지)
        self._prune_old_ckpts()

    def _prune_old_ckpts(self):
        keep = int(self.cfg["train"].get("ckpt_keep_last_n", 3))
        if keep <= 0:
            return
        ckpts = sorted(self.ckpt_dir.glob("student_e*.pt"))
        for p in ckpts[:-keep]:
            try:
                p.unlink()
                print(f"  🗑️  오래된 ckpt 삭제: {p.name}")
            except Exception:
                pass
