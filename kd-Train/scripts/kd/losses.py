"""
SoftKDDetectionLoss:
  - Ultralytics v8DetectionLoss를 subclass.
  - Teacher의 conf>τ 박스를 GT로 사용하되, 클래스 타겟을 one-hot이 아니라
    Teacher가 낸 per-class probability 벡터(soft)로 대체.
  - Box / DFL loss는 v8 기본 동작 유지.

batch 입력 규약(build_batch_from_teacher()가 만들어 줌):
  batch_idx : (M, 1)
  cls       : (M, 1)        ← assigner 호환용 argmax (loss target은 cls_probs로 덮어씀)
  bboxes    : (M, 4)        ← xywhn (0~1 정규화)
  cls_probs : (M, nc)       ← soft target (Teacher의 sigmoid'd class probs)
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class SoftKDDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss의 cls target을 soft prob로 교체.

    cls loss 종류 선택 가능 (cls_loss_type):
      - 'bce'  : 각 클래스 독립 binary CE (기본, multi-label 자연스러움)
      - 'kl'   : Softmax + KL divergence with temperature (Hinton 원조 KD)
    """

    def __init__(self, model, cls_loss_type: str = "bce", temperature: float = 1.0):
        super().__init__(model)
        self.cls_loss_type = cls_loss_type
        self.temperature = float(temperature)

    def __call__(self, preds, batch: Dict[str, torch.Tensor]):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   # (B, A, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(
            feats[0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # ---------- GT 준비 ----------
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            1,
        ).to(self.device)
        targets = self.preprocess(
            targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)             # (B, T, 1) , (B, T, 4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # ---------- pred bbox decode ----------
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, A, 4)

        # ---------- assigner ----------
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        # target_scores: (B, A, nc) — one-hot * task-alignment metric
        # target_gt_idx: (B, A)    — 각 anchor가 매칭된 GT의 batch-내 index

        # ---------- SOFT REPLACEMENT ----------
        # cls_probs 패딩: (B, max_gt, nc)
        cls_probs_padded = self._pad_cls_probs(
            batch["cls_probs"].to(self.device),
            batch["batch_idx"].view(-1).to(self.device),
            batch_size,
            gt_labels.shape[1],
        )
        # alignment metric을 가중치로 보존
        alignment_metric = target_scores.amax(dim=-1, keepdim=True)  # (B, A, 1)
        # target_gt_idx로 soft prob gather
        gather_idx = target_gt_idx.unsqueeze(-1).expand(-1, -1, self.nc)  # (B, A, nc)
        soft_target = cls_probs_padded.gather(1, gather_idx) * alignment_metric
        soft_target = soft_target * fg_mask.unsqueeze(-1).type(soft_target.dtype)
        target_scores = soft_target
        # ---------- END SOFT REPLACEMENT ----------

        target_scores_sum = max(target_scores.sum(), 1)

        # ---------- cls loss (BCE 또는 KL) ----------
        if self.cls_loss_type == "kl":
            # KL divergence: foreground anchor에만 적용
            # softmax 기반이라 각 anchor가 클래스에 대한 분포로 해석됨
            fg = fg_mask.bool()  # (B, A)
            if fg.any():
                # foreground만 추출: (M_fg, nc)
                s_logits = pred_scores[fg]               # (M_fg, nc)
                t_target = target_scores[fg].to(dtype)   # (M_fg, nc)

                # Teacher target을 확률분포로 정규화 (합=1)
                t_prob = t_target / t_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                T = self.temperature
                s_log_prob = F.log_softmax(s_logits / T, dim=-1)
                # KL(t || s_log) = Σ t · (log t - log s)
                # PyTorch: kl_div(input=log_prob, target=prob) = Σ target · (log target - input)
                loss[1] = F.kl_div(
                    s_log_prob, t_prob, reduction="batchmean"
                ) * (T * T)
            else:
                loss[1] = torch.tensor(0.0, device=self.device)
        else:
            # 기본: BCE (multi-label binary, YOLO 표준)
            loss[1] = (
                self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
            )

        # box + dfl
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    def _pad_cls_probs(self, cls_probs_flat, batch_idx_flat, bs, max_gt):
        """(M, nc) flat → (B, max_gt, nc) padded."""
        out = torch.zeros(
            bs, max_gt, self.nc, device=self.device, dtype=cls_probs_flat.dtype
        )
        if cls_probs_flat.numel() == 0:
            return out
        batch_idx_flat = batch_idx_flat.long()
        for b in range(bs):
            mask = batch_idx_flat == b
            n = int(mask.sum().item())
            if n == 0:
                continue
            n = min(n, max_gt)
            out[b, :n] = cls_probs_flat[mask][:n]
        return out


# ============================================================
# Teacher 출력 → v8 loss용 batch dict 변환
# ============================================================

def build_batch_from_teacher(
    teacher_outs: List[Dict[str, torch.Tensor]],
    imgsz: int,
    device: torch.device,
    nc: int,
):
    """
    teacher_outs[i] = {"boxes_xyxy": (M,4), "cls_probs": (M,nc), "conf": (M,)}
    (xyxy는 imgsz pixel 좌표)

    Returns batch dict for SoftKDDetectionLoss:
      batch_idx: (M_total, 1)
      cls:       (M_total, 1)   ← argmax(cls_probs) (assigner용)
      bboxes:    (M_total, 4)   ← xywhn 정규화
      cls_probs: (M_total, nc)
    """
    batch_idx_list, cls_list, bbox_list, prob_list = [], [], [], []
    for i, out in enumerate(teacher_outs):
        boxes = out["boxes_xyxy"]
        probs = out["cls_probs"]
        if boxes.numel() == 0:
            continue
        # xyxy → xywhn
        xy1, xy2 = boxes[:, :2], boxes[:, 2:]
        wh = (xy2 - xy1).clamp(min=1e-6)
        cxcy = (xy1 + xy2) / 2
        xywhn = torch.cat([cxcy / imgsz, wh / imgsz], dim=1).clamp(0, 1)

        cls_arg = probs.argmax(dim=1, keepdim=True).float()  # (M,1)
        bi = torch.full((boxes.shape[0], 1), float(i), device=device)

        batch_idx_list.append(bi)
        cls_list.append(cls_arg)
        bbox_list.append(xywhn)
        prob_list.append(probs)

    if not batch_idx_list:
        return {
            "batch_idx": torch.zeros(0, 1, device=device),
            "cls":       torch.zeros(0, 1, device=device),
            "bboxes":    torch.zeros(0, 4, device=device),
            "cls_probs": torch.zeros(0, nc, device=device),
        }

    return {
        "batch_idx": torch.cat(batch_idx_list, 0),
        "cls":       torch.cat(cls_list, 0),
        "bboxes":    torch.cat(bbox_list, 0),
        "cls_probs": torch.cat(prob_list, 0),
    }
