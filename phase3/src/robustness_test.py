"""
Phase 3: Robustness Test (Optional)
====================================
견고성 테스트: Negative 비율을 1:1 → 1:4로 증가시키며 모델 성능 평가

이 테스트는 파이프라인의 필수 요소는 아니며, 
모델 견고성 검증이 필요할 때 선택적으로 사용
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def run_robustness_test(
    model: nn.Module,
    events: List,
    node_features: torch.Tensor,
    node_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    tis_scores: torch.Tensor,
    test_mask: np.ndarray,
    num_nodes: int,
    device: str = 'cpu',
    neg_ratios: List[float] = [1.0, 2.0, 3.0, 4.0]
) -> Dict[float, Dict[str, float]]:
    """
    견고성 테스트 실행
    
    Negative 비율을 단계적으로 증가시키며 성능 측정
    
    Parameters
    ----------
    model : nn.Module
        학습된 Hybrid 모델
    events : List
        전체 이벤트 리스트
    node_features : torch.Tensor
    node_embeddings : torch.Tensor
    edge_index : torch.Tensor
    tis_scores : torch.Tensor
    test_mask : np.ndarray
        Test 이벤트 마스크
    num_nodes : int
    device : str
    neg_ratios : List[float]
        테스트할 Negative 비율 리스트
    
    Returns
    -------
    results : Dict[float, Dict[str, float]]
        {
            1.0: {'recall@10': 0.3, 'recall@50': 0.5, 'mrr': 0.4},
            2.0: {'recall@10': 0.25, ...},
            ...
        }
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔬 견고성 테스트 (Robustness Test)")
    logger.info("=" * 70)
    logger.info(f"Negative 비율: {neg_ratios}")
    logger.info("=" * 70)
    
    from src.negative_sampler_phase3 import prepare_events_with_negatives
    from src.metrics import compute_recall_at_k, compute_mrr
    
    model.eval()
    results = {}
    
    for neg_ratio in neg_ratios:
        logger.info(f"\n📊 Negative Ratio: 1:{neg_ratio}")
        
        # 네거티브 샘플링
        test_events = prepare_events_with_negatives(
            events=events,
            mask=test_mask,
            num_nodes=num_nodes,
            current_edges=edge_index,
            data_dir="data",
            historical_ratio=0.5,
            neg_ratio=neg_ratio,
            seed=44
        )
        
        # 평가
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            batch_size = 2048
            for i in range(0, len(test_events), batch_size):
                batch_events = test_events[i:i+batch_size]
                
                timestamps = torch.tensor([e[0] for e in batch_events], dtype=torch.long)
                src_nodes = torch.tensor([e[1] for e in batch_events], dtype=torch.long)
                dst_nodes = torch.tensor([e[2] for e in batch_events], dtype=torch.long)
                labels = torch.tensor([e[4] for e in batch_events], dtype=torch.float32)
                
                # GPU 이동
                timestamps = timestamps.to(device)
                src_nodes = src_nodes.to(device)
                dst_nodes = dst_nodes.to(device)
                
                src_features = node_features[src_nodes].to(device)
                dst_features = node_features[dst_nodes].to(device)
                
                # Forward
                logits, _ = model(
                    src_nodes=src_nodes,
                    dst_nodes=dst_nodes,
                    src_features=src_features,
                    dst_features=dst_features,
                    node_embeddings=node_embeddings.to(device),
                    edge_index=edge_index.to(device),
                    timestamps=timestamps,
                    tis_scores=None
                )
                
                scores = torch.sigmoid(logits)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        # 메트릭 계산
        recall_metrics = compute_recall_at_k(all_labels, all_scores, k_list=[10, 50, 100])
        mrr = compute_mrr(all_labels, all_scores)
        
        ratio_results = {**recall_metrics, 'MRR': mrr}
        results[neg_ratio] = ratio_results
        
        logger.info(f"   ✓ Recall@10: {ratio_results['recall@10']:.4f}")
        logger.info(f"   ✓ Recall@50: {ratio_results['recall@50']:.4f}")
        logger.info(f"   ✓ MRR: {ratio_results['MRR']:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ 견고성 테스트 완료")
    logger.info("=" * 70)
    
    # 결과 요약
    logger.info("\n📊 견고성 테스트 요약:")
    logger.info("-" * 70)
    logger.info(f"{'Neg Ratio':<12} {'Recall@10':<12} {'Recall@50':<12} {'MRR':<12}")
    logger.info("-" * 70)
    for neg_ratio, metrics in results.items():
        logger.info(
            f"1:{neg_ratio:<10.1f} "
            f"{metrics['recall@10']:<12.4f} "
            f"{metrics['recall@50']:<12.4f} "
            f"{metrics['MRR']:<12.4f}"
        )
    logger.info("-" * 70)
    
    return results


def visualize_robustness_results(
    results: Dict[float, Dict[str, float]],
    save_path: str = "results/robustness_test.png"
):
    """
    견고성 테스트 결과 시각화
    
    Parameters
    ----------
    results : Dict[float, Dict[str, float]]
        run_robustness_test의 출력
    save_path : str
    """
    import matplotlib.pyplot as plt
    
    neg_ratios = sorted(results.keys())
    recall10 = [results[r]['recall@10'] for r in neg_ratios]
    recall50 = [results[r]['recall@50'] for r in neg_ratios]
    mrr = [results[r]['MRR'] for r in neg_ratios]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recall@10
    axes[0].plot(neg_ratios, recall10, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Negative Ratio (1:X)', fontsize=12)
    axes[0].set_ylabel('Recall@10', fontsize=12)
    axes[0].set_title('Recall@10 vs Negative Ratio', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(neg_ratios)
    axes[0].set_xticklabels([f'1:{int(r)}' for r in neg_ratios])
    
    # Recall@50
    axes[1].plot(neg_ratios, recall50, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Negative Ratio (1:X)', fontsize=12)
    axes[1].set_ylabel('Recall@50', fontsize=12)
    axes[1].set_title('Recall@50 vs Negative Ratio', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(neg_ratios)
    axes[1].set_xticklabels([f'1:{int(r)}' for r in neg_ratios])
    
    # MRR
    axes[2].plot(neg_ratios, mrr, marker='^', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Negative Ratio (1:X)', fontsize=12)
    axes[2].set_ylabel('MRR', fontsize=12)
    axes[2].set_title('MRR vs Negative Ratio', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(neg_ratios)
    axes[2].set_xticklabels([f'1:{int(r)}' for r in neg_ratios])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 견고성 테스트 그래프 저장: {save_path}")
