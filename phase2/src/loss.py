"""
Risk-Aware Loss Functions
==========================
TIS 점수를 고려한 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RiskAwareBCELoss(nn.Module):
    """
    TIS 기반 Soft Label BCE Loss
    
    Positive 엣지의 타겟이 TIS에 따라 조정됨:
    - TIS 낮음 (안전) → target = 1.0
    - TIS 높음 (위험) → target = 1.0 - alpha * TIS
    
    Parameters
    ----------
    alpha : float
        TIS 페널티 강도 (0~1)
        0: TIS 무시, 1: TIS 완전 반영
    """
    
    def __init__(self, alpha: float = 0.3):
        super(RiskAwareBCELoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tis_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor, shape (E,)
            예측 점수 (0~1)
        target : torch.Tensor, shape (E,)
            타겟 레이블 (0 or 1)
        tis_scores : torch.Tensor, shape (E,), optional
            TIS 점수 (positive 엣지만)
        
        Returns
        -------
        loss : torch.Tensor, scalar
        """
        # TIS 기반 Soft Label
        if tis_scores is not None:
            # Positive 엣지만 조정
            pos_mask = (target == 1)
            adjusted_target = target.clone()
            
            # Soft Label: 1.0 - alpha * TIS
            adjusted_target[pos_mask] = 1.0 - self.alpha * tis_scores[pos_mask]
            adjusted_target = torch.clamp(adjusted_target, 0.0, 1.0)
        else:
            adjusted_target = target
        
        # BCE Loss
        loss = self.bce(pred, adjusted_target)
        
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    대조 학습 손실 (선택적)
    Positive는 가깝게, Negative는 멀리
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor, shape (N, D)
            노드 임베딩
        edge_index : torch.Tensor, shape (2, E)
            엣지 인덱스
        labels : torch.Tensor, shape (E,)
            0 (negative) or 1 (positive)
        """
        src_emb = embeddings[edge_index[0]]
        dst_emb = embeddings[edge_index[1]]
        
        # Euclidean Distance
        dist = torch.norm(src_emb - dst_emb, p=2, dim=1)
        
        # Contrastive Loss
        pos_loss = labels * dist ** 2
        neg_loss = (1 - labels) * torch.clamp(self.margin - dist, min=0) ** 2
        
        loss = (pos_loss + neg_loss).mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss (Hard Example에 집중)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor, shape (E,)
            예측 확률 (0~1)
        target : torch.Tensor, shape (E,)
            타겟 레이블 (0 or 1)
        """
        # BCE Loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss.mean()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    평가 지표 계산
    
    Returns
    -------
    metrics : dict
        - accuracy: 정확도
        - precision: 정밀도
        - recall: 재현율
        - f1: F1 Score
    """
    pred_labels = (pred > 0.5).float()
    
    tp = ((pred_labels == 1) & (target == 1)).sum().item()
    fp = ((pred_labels == 1) & (target == 0)).sum().item()
    tn = ((pred_labels == 0) & (target == 0)).sum().item()
    fn = ((pred_labels == 0) & (target == 1)).sum().item()
    
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Loss Functions 테스트")
    print("=" * 70)
    
    # 더미 데이터
    batch_size = 1000
    pred = torch.rand(batch_size)
    target = torch.randint(0, 2, (batch_size,)).float()
    tis_scores = torch.rand(batch_size)
    
    # Risk-Aware BCE Loss
    print("\n[1] RiskAwareBCELoss")
    loss_fn = RiskAwareBCELoss(alpha=0.3)
    
    loss_without_tis = loss_fn(pred, target, tis_scores=None)
    loss_with_tis = loss_fn(pred, target, tis_scores=tis_scores)
    
    print(f"   - Loss (without TIS): {loss_without_tis:.4f}")
    print(f"   - Loss (with TIS): {loss_with_tis:.4f}")
    
    # Metrics
    print("\n[2] Metrics")
    metrics = compute_metrics(pred, target)
    for key, val in metrics.items():
        print(f"   - {key}: {val:.4f}")
