"""
Evaluation Metrics for Link Prediction
=======================================
링크 예측 평가 메트릭:
    - Accuracy, Precision, Recall, F1, AUC-ROC, AP
    - Recall@K (Top-K 추천)
    - MRR (Mean Reciprocal Rank)
    - RMSE (Risk-aware 예측 정확도)
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    mean_squared_error
)
from typing import Dict, Tuple, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    평가 메트릭 계산
    
    Parameters
    ----------
    y_true : np.ndarray [N]
        실제 레이블 (0 or 1)
    y_pred : np.ndarray [N]
        예측 레이블 (0 or 1)
    y_score : np.ndarray [N], optional
        예측 확률 (0~1)
    threshold : float
        분류 임계값
    
    Returns
    -------
    metrics : dict
        평가 메트릭 딕셔너리
    """
    metrics = {}
    
    # 기본 메트릭
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # 확률 기반 메트릭
    if y_score is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_score)
        except:
            metrics['auc_roc'] = 0.0
        
        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_score)
        except:
            metrics['auc_pr'] = 0.0
    
    return metrics


def compute_topk_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_list: list = [10, 50, 100]
) -> Dict[str, float]:
    """
    Top-K Precision 계산
    
    Parameters
    ----------
    y_true : np.ndarray [N]
        실제 레이블
    y_score : np.ndarray [N]
        예측 확률
    k_list : list of int
        K 값 리스트
    
    Returns
    -------
    topk_metrics : dict
        Top-K Precision 딕셔너리
    """
    topk_metrics = {}
    
    # 점수 내림차순 정렬
    sorted_indices = np.argsort(y_score)[::-1]
    
    for k in k_list:
        if k > len(y_true):
            continue
        
        # Top-K 인덱스
        topk_indices = sorted_indices[:k]
        topk_true = y_true[topk_indices]
        
        # Precision@K
        precision_at_k = topk_true.sum() / k
        topk_metrics[f'precision@{k}'] = precision_at_k
    
    return topk_metrics


def compute_recall_at_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float = 0.9
) -> Tuple[float, float]:
    """
    특정 Precision에서의 Recall 계산
    
    Parameters
    ----------
    y_true : np.ndarray [N]
    y_score : np.ndarray [N]
    target_precision : float
        목표 Precision
    
    Returns
    -------
    recall : float
    threshold : float
        해당 Recall을 달성하는 임계값
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # target_precision 이상인 인덱스 찾기
    valid_indices = np.where(precision >= target_precision)[0]
    
    if len(valid_indices) == 0:
        return 0.0, 1.0
    
    # 가장 높은 Recall
    best_idx = valid_indices[np.argmax(recall[valid_indices])]
    best_recall = recall[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    
    return best_recall, best_threshold


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Confusion Matrix 기반 메트릭
    
    Returns
    -------
    metrics : dict
        TP, FP, TN, FN, TPR, FPR, TNR, FNR
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics = {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'TPR': TP / (TP + FN) if (TP + FN) > 0 else 0.0,  # Recall
        'FPR': FP / (FP + TN) if (FP + TN) > 0 else 0.0,
        'TNR': TN / (TN + FP) if (TN + FP) > 0 else 0.0,  # Specificity
        'FNR': FN / (FN + TP) if (FN + TP) > 0 else 0.0,
    }
    
    return metrics


def compute_mrr(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    MRR (Mean Reciprocal Rank) 계산
    
    실제 Positive 엣지들이 예측 랭킹에서 얼마나 상위에 위치하는가?
    
    Parameters
    ----------
    y_true : np.ndarray [N]
        실제 레이블 (1: positive, 0: negative)
    y_score : np.ndarray [N]
        예측 점수 (높을수록 positive일 확률 높음)
    
    Returns
    -------
    mrr : float
        Mean Reciprocal Rank (0~1, 높을수록 좋음)
    
    Example
    -------
    >>> y_true = [1, 0, 1, 0, 1]  # Positive at 0, 2, 4
    >>> y_score = [0.9, 0.8, 0.7, 0.6, 0.5]  # Ranking: 0, 1, 2, 3, 4
    >>> # Positive ranks: 1st, 3rd, 5th
    >>> # Reciprocal ranks: 1/1, 1/3, 1/5
    >>> # MRR = (1 + 0.333 + 0.2) / 3 = 0.511
    """
    # 점수 기준 내림차순 정렬
    ranking = np.argsort(y_score)[::-1]
    
    # Positive 엣지 인덱스
    pos_indices = np.where(y_true == 1)[0]
    
    if len(pos_indices) == 0:
        return 0.0
    
    # 각 Positive의 rank (1-indexed)
    rank_dict = {idx: rank + 1 for rank, idx in enumerate(ranking)}
    pos_ranks = [rank_dict[idx] for idx in pos_indices]
    
    # Reciprocal Rank
    reciprocal_ranks = [1.0 / rank for rank in pos_ranks]
    
    # MRR
    mrr = np.mean(reciprocal_ranks)
    
    return mrr


def compute_rmse_with_confidence(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_scores: Optional[np.ndarray] = None,
    tis_scores: Optional[np.ndarray] = None,
    alpha: float = 0.3
) -> Dict[str, float]:
    """
    RMSE 계산 (Risk-aware)
    
    TIS 기반 Soft Label과 예측값 간의 오차 측정
    
    Parameters
    ----------
    y_true : np.ndarray [N]
        실제 레이블 (1 or 0)
    y_score : np.ndarray [N]
        예측 점수 (0~1)
    confidence_scores : np.ndarray [N], optional
        UKGE 신뢰도 점수 (0~1)
    tis_scores : np.ndarray [N], optional
        각 엣지의 TIS 점수 (0~1)
    alpha : float
        TIS 페널티 강도
    
    Returns
    -------
    rmse_metrics : Dict[str, float]
        {
            'rmse': float,  # 전체 RMSE
            'rmse_pos': float,  # Positive 엣지 RMSE
            'rmse_neg': float,  # Negative 엣지 RMSE
            'rmse_weighted': float  # Confidence-weighted RMSE (if available)
        }
    """
    metrics = {}
    
    # 1. 기본 RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_score))
    metrics['rmse'] = rmse
    
    # 2. Positive/Negative 분리 RMSE
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    if pos_mask.sum() > 0:
        rmse_pos = np.sqrt(mean_squared_error(y_true[pos_mask], y_score[pos_mask]))
        metrics['rmse_pos'] = rmse_pos
    
    if neg_mask.sum() > 0:
        rmse_neg = np.sqrt(mean_squared_error(y_true[neg_mask], y_score[neg_mask]))
        metrics['rmse_neg'] = rmse_neg
    
    # 3. TIS-aware RMSE (Soft Label 기반)
    if tis_scores is not None:
        # Soft Label: Positive는 1.0 - alpha * TIS, Negative는 0.0
        soft_labels = np.where(
            y_true == 1,
            1.0 - alpha * tis_scores,
            0.0
        )
        rmse_tis = np.sqrt(mean_squared_error(soft_labels, y_score))
        metrics['rmse_tis_aware'] = rmse_tis
    
    # 4. Confidence-weighted RMSE (UKGE)
    if confidence_scores is not None:
        # 신뢰도가 높은 예측에 더 큰 가중치
        weights = confidence_scores
        weighted_errors = weights * (y_true - y_score) ** 2
        rmse_weighted = np.sqrt(weighted_errors.mean())
        metrics['rmse_weighted'] = rmse_weighted
    
    return metrics


def compute_recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_list: list = [10, 50, 100]
) -> Dict[str, float]:
    """
    Recall@K 계산
    
    상위 K개 예측 중 실제 Positive가 몇 개 포함되는가?
    
    Parameters
    ----------
    y_true : np.ndarray [N]
        실제 레이블 (1 or 0)
    y_score : np.ndarray [N]
        예측 점수
    k_list : list
        K 값 리스트
    
    Returns
    -------
    recall_metrics : Dict[str, float]
        {'recall@10': 0.3, 'recall@50': 0.5, ...}
    """
    metrics = {}
    
    # 점수 기준 내림차순 정렬
    ranking = np.argsort(y_score)[::-1]
    
    # 전체 Positive 개수
    num_pos = y_true.sum()
    
    if num_pos == 0:
        return {f'recall@{k}': 0.0 for k in k_list}
    
    for k in k_list:
        if k > len(y_true):
            k = len(y_true)
        
        # Top-K 인덱스
        topk_indices = ranking[:k]
        topk_labels = y_true[topk_indices]
        
        # Recall@K = Top-K 중 Positive 개수 / 전체 Positive 개수
        recall_k = topk_labels.sum() / num_pos
        metrics[f'recall@{k}'] = recall_k
    
    return metrics


# ============================================================
# 통합 평가 함수
# ============================================================

def evaluate_link_prediction_comprehensive(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_scores: Optional[np.ndarray] = None,
    tis_scores: Optional[np.ndarray] = None,
    k_list: list = [10, 50, 100, 500],
    threshold: float = 0.5,
    alpha: float = 0.3
) -> Dict[str, float]:
    """
    링크 예측 종합 평가
    
    Parameters
    ----------
    y_true : np.ndarray [N]
    y_score : np.ndarray [N]
    confidence_scores : np.ndarray [N], optional
    tis_scores : np.ndarray [N], optional
    k_list : list
    threshold : float
    alpha : float
    
    Returns
    -------
    all_metrics : Dict[str, float]
        모든 평가 메트릭
    """
    all_metrics = {}
    
    # 1. 기본 메트릭
    y_pred = (y_score >= threshold).astype(int)
    basic_metrics = compute_metrics(y_true, y_pred, y_score, threshold)
    all_metrics.update(basic_metrics)
    
    # 2. Recall@K
    recall_metrics = compute_recall_at_k(y_true, y_score, k_list)
    all_metrics.update(recall_metrics)
    
    # 3. MRR
    mrr = compute_mrr(y_true, y_score)
    all_metrics['MRR'] = mrr
    
    # 4. RMSE (Risk-aware)
    rmse_metrics = compute_rmse_with_confidence(
        y_true, y_score, confidence_scores, tis_scores, alpha
    )
    all_metrics.update(rmse_metrics)
    
    return all_metrics


def evaluate_link_prediction(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    k_list: list = [10, 50, 100],
    verbose: bool = True
) -> Dict[str, float]:
    """
    링크 예측 종합 평가
    
    Parameters
    ----------
    y_true : np.ndarray [N]
        실제 레이블
    y_score : np.ndarray [N]
        예측 확률
    threshold : float
        분류 임계값
    k_list : list of int
        Top-K 평가용 K 리스트
    verbose : bool
        결과 출력 여부
    
    Returns
    -------
    all_metrics : dict
        모든 평가 메트릭
    """
    # 예측 레이블
    y_pred = (y_score >= threshold).astype(int)
    
    # 1. 기본 메트릭
    metrics = compute_metrics(y_true, y_pred, y_score, threshold)
    
    # 2. Top-K Precision
    topk_metrics = compute_topk_precision(y_true, y_score, k_list)
    
    # 3. Confusion Matrix
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    
    # 4. Recall @ High Precision
    recall_at_90, threshold_90 = compute_recall_at_precision(y_true, y_score, 0.9)
    
    # 통합
    all_metrics = {
        **metrics,
        **topk_metrics,
        **cm_metrics,
        'recall@precision_0.9': recall_at_90,
        'threshold@precision_0.9': threshold_90
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("📊 링크 예측 평가 결과")
        print("=" * 70)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics.get('auc_roc', 0):.4f}")
        print(f"AUC-PR:    {metrics.get('auc_pr', 0):.4f}")
        print("-" * 70)
        print(f"TP: {cm_metrics['TP']:,} | FP: {cm_metrics['FP']:,} | "
              f"TN: {cm_metrics['TN']:,} | FN: {cm_metrics['FN']:,}")
        print(f"TPR: {cm_metrics['TPR']:.4f} | FPR: {cm_metrics['FPR']:.4f}")
        print("-" * 70)
        for k in k_list:
            key = f'precision@{k}'
            if key in topk_metrics:
                print(f"Precision@{k}: {topk_metrics[key]:.4f}")
        print("-" * 70)
        print(f"Recall @ Precision 0.9: {recall_at_90:.4f} (Threshold: {threshold_90:.4f})")
        print("=" * 70)
    
    return all_metrics


def convert_torch_to_numpy(
    *tensors: torch.Tensor
) -> Tuple[np.ndarray, ...]:
    """
    PyTorch Tensor를 NumPy로 변환
    """
    result = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            result.append(tensor.detach().cpu().numpy())
        else:
            result.append(tensor)
    
    if len(result) == 1:
        return result[0]
    return tuple(result)
