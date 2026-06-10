"""
TeacherWorld: YOLOv8x-worldv2를 frozen FP16 eval로 감싸고,
각 이미지마다 (boxes_xyxy, soft cls_probs, conf)를 반환.
- conf > τ 필터링
- NMS는 torchvision batched_nms
- DDP에서 각 rank가 동일 weights로 복제 (rank별 독립 추론)
"""
from typing import List, Dict

import torch
import torch.nn as nn
from torchvision.ops import batched_nms

from ultralytics import YOLOWorld
from ultralytics.utils import ops


class TeacherWorld(nn.Module):
    def __init__(
        self,
        weights: str,
        vocab: List[str],
        device: torch.device,
        fp16: bool = True,
        conf_threshold: float = 0.7,
        iou_threshold: float = 0.5,
        max_dets: int = 100,
    ):
        super().__init__()
        self.device = device
        self.fp16 = fp16
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_dets = max_dets
        self.vocab = vocab
        self.nc = len(vocab)

        wrapper = YOLOWorld(weights)
        wrapper.set_classes(vocab)             # text embedding 주입
        self.net: nn.Module = wrapper.model    # DetectionModel
        self.net.to(device).eval()
        if fp16:
            self.net.half()
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        imgs: (B,3,H,W) float [0,1] on self.device
        Returns: list of B dicts:
          - boxes_xyxy: (M,4) imgsz scale
          - cls_probs:  (M, nc) soft class targets (∈[0,1])
          - conf:       (M,)    max class prob
        """
        x = imgs.to(self.device, non_blocking=True)
        if self.fp16:
            x = x.half()

        preds = self.net(x)
        # YOLOv8 eval mode → (decoded_preds, feats_list) 튜플로 반환
        # decoded_preds shape: (B, 4+nc, num_anchors) — bbox(xywh)는 디코딩 완료,
        # cls 부분은 이미 sigmoid 적용되어 [0,1] 확률
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        return self._postprocess(preds.float())

    def _postprocess(self, preds: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        # (B, 4+nc, N) → (B, N, 4+nc)
        preds = preds.transpose(1, 2).contiguous()
        bs = preds.shape[0]
        out: List[Dict[str, torch.Tensor]] = []

        for b in range(bs):
            x = preds[b]
            xywh = x[:, :4]
            probs = x[:, 4:]                          # 이미 sigmoid 적용
            best_conf, best_cls = probs.max(dim=1)
            mask = best_conf > self.conf_threshold
            if not mask.any():
                out.append(self._empty())
                continue

            xywh = xywh[mask]
            probs = probs[mask]
            best_conf = best_conf[mask]
            best_cls = best_cls[mask]
            xyxy = ops.xywh2xyxy(xywh)

            keep = batched_nms(xyxy, best_conf, best_cls, self.iou_threshold)
            keep = keep[: self.max_dets]

            out.append({
                "boxes_xyxy": xyxy[keep],
                "cls_probs":  probs[keep],            # soft target
                "conf":       best_conf[keep],
            })
        return out

    def _empty(self):
        return {
            "boxes_xyxy": torch.zeros(0, 4, device=self.device),
            "cls_probs":  torch.zeros(0, self.nc, device=self.device),
            "conf":       torch.zeros(0, device=self.device),
        }
