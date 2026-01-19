"""
Phase 3: Hybrid Loss Function
==============================
TIS 기반 Soft Label BCE + Ranking Loss

손실 함수:
    1. TIS-aware BCE: Positive 엣지는 1.0 - TIS*alpha, Negative는 0.0 (또는 soft 0.05)
    2. Ranking Loss: Positive 점수가 Negative 점수보다 margin만큼 높아야 함
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HybridLoss(nn.Module):
    """
    Hybrid Loss = TIS-aware BCE + Ranking Loss
    
    Parameters
    ----------
    alpha : float
        TIS 페널티 강도 (default: 0.3)
    soft_negative : float
        Negative 엣지의 Soft Label (default: 0.0)
    ranking_margin : float
        Ranking Loss의 margin (default: 0.5)
    ranking_weight : float
        Ranking Loss의 가중치 (default: 0.1)
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        soft_negative: float = 0.0,
        ranking_margin: float = 0.5,
        ranking_weight: float = 0.1
    ):
        super().__init__()
        
        self.alpha = alpha
        self.soft_negative = soft_negative
        self.ranking_margin = ranking_margin
        self.ranking_weight = ranking_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ranking_loss = nn.MarginRankingLoss(margin=ranking_margin, reduction='mean')
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tis_scores: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Parameters
        ----------
        logits : torch.Tensor [N]
            모델 예측 (logit)
        labels : torch.Tensor [N]
            실제 레이블 (1.0 or 0.0)
        tis_scores : torch.Tensor [num_nodes], optional
            각 노드의 TIS 점수
        edge_index : torch.Tensor [2, N], optional
            엣지 인덱스 (tis_scores 사용 시 필요)
        
        Returns
        -------
        total_loss : torch.Tensor
        bce_loss : torch.Tensor
        ranking_loss : torch.Tensor
        """
        
        # ============================================================
        # 1. TIS 기반 Soft Label 생성
        # ============================================================
        
        if tis_scores is not None and edge_index is not None:
            # edge_index: [2, batch_size]
            # tis_scores: [num_nodes] 전체 노드의 TIS
            
            # Positive 엣지의 TIS 점수 (dst 노드 기준)
            dst_nodes = edge_index[1]  # [batch_size]
            
            # tis_scores가 1차원인지 확인
            if tis_scores.dim() == 1:
                dst_tis = tis_scores[dst_nodes]  # [batch_size]
            else:
                # 2차원인 경우 squeeze
                dst_tis = tis_scores.squeeze()[dst_nodes]  # [batch_size]
            
            # Soft Label 계산 (명시적으로 1D 텐서 생성)
            positive_labels = torch.ones_like(labels) - self.alpha * dst_tis
            negative_labels = torch.full_like(labels, self.soft_negative)
            
            soft_labels = torch.where(
                labels > 0.5,  # Positive인 경우
                positive_labels,
                negative_labels
            )
        else:
            # TIS 없으면 일반 레이블
            soft_labels = torch.where(
                labels > 0.5,
                torch.ones_like(labels),
                torch.full_like(labels, self.soft_negative)
            )
        
        # ============================================================
        # 2. BCE Loss (Soft Label)
        # ============================================================
        
        bce = self.bce_loss(logits, soft_labels)
        bce_mean = bce.mean()
        
        # ============================================================
        # 3. Ranking Loss (Positive vs Negative)
        # ============================================================
        
        # Positive/Negative 분리
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        
        # Ranking Loss 계산 (Positive가 Negative보다 커야 함)
        if len(pos_logits) > 0 and len(neg_logits) > 0:
            # 짝 맞추기 (같은 개수만큼 샘플링)
            min_size = min(len(pos_logits), len(neg_logits))
            pos_sample = pos_logits[:min_size]
            neg_sample = neg_logits[:min_size]
            
            # target = 1 (pos가 neg보다 커야 함)
            target = torch.ones(min_size, device=logits.device)
            ranking = self.ranking_loss(pos_sample, neg_sample, target)
        else:
            ranking = torch.tensor(0.0, device=logits.device)
        
        # ============================================================
        # 4. Total Loss
        # ============================================================
        
        total_loss = bce_mean + self.ranking_weight * ranking
        
        return total_loss, bce_mean, ranking


class TISAwareBCELoss(nn.Module):
    """
    TIS-aware BCE Loss (Ranking Loss 없는 버전)
    
    Parameters
    ----------
    alpha : float
        TIS 페널티 강도
    soft_negative : float
        Negative 엣지의 Soft Label
    """
    
    def __init__(self, alpha: float = 0.3, soft_negative: float = 0.0):
        super().__init__()
        
        self.alpha = alpha
        self.soft_negative = soft_negative
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tis_scores: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns
        -------
        loss : torch.Tensor
        """
        
        # TIS 기반 Soft Label
        if tis_scores is not None and edge_index is not None:
            dst_nodes = edge_index[1]
            dst_tis = tis_scores[dst_nodes]
            
            soft_labels = torch.where(
                labels > 0.5,
                1.0 - self.alpha * dst_tis,
                torch.full_like(labels, self.soft_negative)
            )
        else:
            soft_labels = torch.where(
                labels > 0.5,
                torch.ones_like(labels),
                torch.full_like(labels, self.soft_negative)
            )
        
        return self.bce_loss(logits, soft_labels)


# ============================================================
# 유틸리티 함수
# ============================================================

def compute_soft_labels(
    labels: torch.Tensor,
    tis_scores: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = 0.3,
    soft_negative: float = 0.0
) -> torch.Tensor:
    """
    TIS 기반 Soft Label 계산
    
    Parameters
    ----------
    labels : torch.Tensor [N]
        원본 레이블 (1.0 or 0.0)
    tis_scores : torch.Tensor [num_nodes]
        각 노드의 TIS 점수
    edge_index : torch.Tensor [2, N]
        엣지 인덱스
    alpha : float
        TIS 페널티 강도
    soft_negative : float
        Negative 엣지의 Soft Label
    
    Returns
    -------
    soft_labels : torch.Tensor [N]
    """
    
    dst_nodes = edge_index[1]
    dst_tis = tis_scores[dst_nodes]
    
    soft_labels = torch.where(
        labels > 0.5,
        1.0 - alpha * dst_tis,  # Positive: 1.0 - TIS*alpha
        torch.full_like(labels, soft_negative)  # Negative: 0.0 or 0.05
    )
    
    return soft_labels


def visualize_soft_labels(
    labels: torch.Tensor,
    soft_labels: torch.Tensor,
    tis_scores: torch.Tensor,
    edge_index: torch.Tensor
):
    """
    Soft Label 분포 시각화 (디버깅용)
    """
    import matplotlib.pyplot as plt
    
    pos_mask = labels > 0.5
    pos_soft = soft_labels[pos_mask].cpu().numpy()
    
    dst_nodes = edge_index[1][pos_mask]
    pos_tis = tis_scores[dst_nodes].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Soft Label 분포
    axes[0].hist(pos_soft, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Soft Label (Positive)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Positive Edge Soft Label Distribution')
    axes[0].axvline(pos_soft.mean(), color='red', linestyle='--', label=f'Mean: {pos_soft.mean():.3f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # TIS vs Soft Label
    axes[1].scatter(pos_tis, pos_soft, alpha=0.5, s=10)
    axes[1].set_xlabel('TIS Score (Destination)')
    axes[1].set_ylabel('Soft Label')
    axes[1].set_title('TIS Score vs Soft Label')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/soft_label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Soft Label 분포 저장: results/soft_label_distribution.png")
    print(f"   - Positive Soft Label 평균: {pos_soft.mean():.4f}")
    print(f"   - Positive Soft Label 범위: [{pos_soft.min():.4f}, {pos_soft.max():.4f}]")
