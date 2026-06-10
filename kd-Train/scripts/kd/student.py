"""
StudentWorld: YOLOv8s-worldv2 (학습 대상).
- vocab text embedding 주입
- DDP wrap은 distill.py에서 처리
- model attribute (DetectionModel)을 외부로 노출해 v8DetectionLoss 호환
- 옵션: backbone freeze (forgetting 방지)
"""
from typing import List

import torch
import torch.nn as nn

from ultralytics import YOLOWorld


class StudentWorld(nn.Module):
    def __init__(self, weights: str, vocab: List[str], device: torch.device,
                 freeze_backbone: bool = False, freeze_layers: int = 10):
        super().__init__()
        self.device = device
        self.vocab = vocab
        self.nc = len(vocab)

        wrapper = YOLOWorld(weights)
        wrapper.set_classes(vocab)
        self.net: nn.Module = wrapper.model        # DetectionModel
        self.net.to(device).train()

        # hyp 주입 (v8DetectionLoss가 self.hyp.box/cls/dfl 읽음)
        if not hasattr(self.net, "args") or self.net.args is None:
            from types import SimpleNamespace
            self.net.args = SimpleNamespace(box=7.5, cls=1.0, dfl=1.5)

        # ── Backbone freeze (A) ──
        if freeze_backbone:
            self._freeze_backbone(freeze_layers)

    def _freeze_backbone(self, n_layers: int):
        """YOLOv8 backbone 레이어 (model.0 ~ model.{n_layers-1}) 동결."""
        frozen = 0
        total = 0
        for name, p in self.net.named_parameters():
            total += 1
            # name 형식: "model.0.conv.weight", "model.1.cv1.conv.weight", ...
            if any(name.startswith(f"model.{i}.") for i in range(n_layers)):
                p.requires_grad = False
                frozen += 1

        # BatchNorm running stats도 backbone 부분은 freeze (eval mode)
        for i in range(n_layers):
            try:
                layer = self.net.model[i]
                for m in layer.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            except (IndexError, AttributeError):
                pass

        print(f"   🧊 Backbone freeze: model.0 ~ model.{n_layers-1}")
        print(f"      파라미터 동결: {frozen}/{total} ({frozen/total*100:.1f}%)")

    def forward(self, imgs: torch.Tensor):
        """train mode → list of feature maps per scale."""
        return self.net(imgs)

    @property
    def detection_model(self):
        return self.net
