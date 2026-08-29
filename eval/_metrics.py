import numpy as np
from sklearn.metrics import roc_auc_score


def per_image_iou_f1(pred_bin, gt_bin):
    """IoU and pixel F1 of one binary prediction against one binary mask.

    Accepts torch tensors or numpy arrays.
    """
    gt_empty = float(gt_bin.sum()) == 0
    pred_empty = float(pred_bin.sum()) == 0
    if gt_empty and pred_empty:
        return 1.0, 1.0
    if gt_empty or pred_empty:
        return 0.0, 0.0
    tp = float((pred_bin * gt_bin).sum())
    fp = float((pred_bin * (1 - gt_bin)).sum())
    fn = float(((1 - pred_bin) * gt_bin).sum())
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return iou, f1


class _Acc:
    """Accumulates detection scores and per-image localization scores."""

    def __init__(self):
        self.img_scores: list[float] = []
        self.img_labels: list[int] = []
        self.iou_per_image: list[float] = []
        self.f1_per_image: list[float] = []

    def update(self, probs_bhw, gt_bhw, is_tampered: bool = False):
        for prob, gt in zip(probs_bhw.cpu().numpy(), gt_bhw.cpu().numpy()):
            self.img_scores.append(float(prob.mean()))
            self.img_labels.append(int(is_tampered))
            iou, f1 = per_image_iou_f1((prob >= 0.5).astype(float), gt)
            self.iou_per_image.append(iou)
            self.f1_per_image.append(f1)

    def image_acc(self) -> float:
        if not self.img_scores:
            return float("nan")
        preds = [int(s >= 0.5) for s in self.img_scores]
        return sum(p == l for p, l in zip(preds, self.img_labels)) / len(preds)

    def image_auc(self) -> float:
        if len(set(self.img_labels)) < 2:
            return float("nan")
        return float(roc_auc_score(self.img_labels, self.img_scores))

    def iou_per_image_mean(self) -> float:
        return float(np.mean(self.iou_per_image)) if self.iou_per_image else float("nan")

    def f1_per_image_mean(self) -> float:
        return float(np.mean(self.f1_per_image)) if self.f1_per_image else float("nan")

    def summary(self) -> dict:
        return {
            "img_acc": self.image_acc(),
            "img_auc": self.image_auc(),
            "iou_per_image": self.iou_per_image_mean(),
            "f1_per_image": self.f1_per_image_mean(),
            "n": len(self.img_scores),
        }
