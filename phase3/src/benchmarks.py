"""
Phase 3: Benchmark Heuristic Algorithms
========================================
고전적 링크 예측 휴리스틱 알고리즘:
    1. PA (Preferential Attachment)
    2. RA (Resource Allocation)
    3. JC (Jaccard Coefficient)
"""

import numpy as np
import torch
from typing import Tuple, Dict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class LinkPredictionBenchmarks:
    """
    링크 예측 벤치마크 휴리스틱 알고리즘
    
    Parameters
    ----------
    edge_index : torch.Tensor [2, E]
        학습 그래프의 엣지 인덱스
    num_nodes : int
        전체 노드 수
    """
    
    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        self.edge_index = edge_index.numpy()
        self.num_nodes = num_nodes
        
        # 인접 리스트 구축
        self.adjacency = defaultdict(set)
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            self.adjacency[src].add(dst)
            # 무방향 그래프로 취급
            self.adjacency[dst].add(src)
        
        # Degree 계산
        self.degrees = {node: len(neighbors) for node, neighbors in self.adjacency.items()}
        
        logger.info(f"✅ Benchmarks 초기화")
        logger.info(f"   - 노드 수: {num_nodes}")
        logger.info(f"   - 엣지 수: {self.edge_index.shape[1]:,}")
        logger.info(f"   - 평균 Degree: {np.mean(list(self.degrees.values())):.2f}")
    
    def preferential_attachment(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor
    ) -> np.ndarray:
        """
        PA (Preferential Attachment)
        
        Score(u, v) = degree(u) * degree(v)
        
        Parameters
        ----------
        src_nodes : torch.Tensor [N]
        dst_nodes : torch.Tensor [N]
        
        Returns
        -------
        scores : np.ndarray [N]
        """
        src_nodes = src_nodes.numpy()
        dst_nodes = dst_nodes.numpy()
        
        scores = []
        for src, dst in zip(src_nodes, dst_nodes):
            deg_u = self.degrees.get(src, 0)
            deg_v = self.degrees.get(dst, 0)
            scores.append(deg_u * deg_v)
        
        return np.array(scores, dtype=np.float32)
    
    def resource_allocation(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor
    ) -> np.ndarray:
        """
        RA (Resource Allocation)
        
        Score(u, v) = Σ_{z ∈ common_neighbors} 1 / degree(z)
        
        공통 이웃이 많을수록, 그 이웃의 degree가 작을수록 높은 점수
        
        Parameters
        ----------
        src_nodes : torch.Tensor [N]
        dst_nodes : torch.Tensor [N]
        
        Returns
        -------
        scores : np.ndarray [N]
        """
        src_nodes = src_nodes.numpy()
        dst_nodes = dst_nodes.numpy()
        
        scores = []
        for src, dst in zip(src_nodes, dst_nodes):
            neighbors_u = self.adjacency.get(src, set())
            neighbors_v = self.adjacency.get(dst, set())
            
            # 공통 이웃
            common = neighbors_u & neighbors_v
            
            if len(common) == 0:
                scores.append(0.0)
            else:
                # Resource Allocation: 1 / degree(z) 합산
                score = sum(1.0 / self.degrees.get(z, 1) for z in common)
                scores.append(score)
        
        return np.array(scores, dtype=np.float32)
    
    def jaccard_coefficient(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor
    ) -> np.ndarray:
        """
        JC (Jaccard Coefficient)
        
        Score(u, v) = |common_neighbors| / |union_neighbors|
        
        두 노드의 이웃 집합이 얼마나 겹치는가 (유사도)
        
        Parameters
        ----------
        src_nodes : torch.Tensor [N]
        dst_nodes : torch.Tensor [N]
        
        Returns
        -------
        scores : np.ndarray [N]
        """
        src_nodes = src_nodes.numpy()
        dst_nodes = dst_nodes.numpy()
        
        scores = []
        for src, dst in zip(src_nodes, dst_nodes):
            neighbors_u = self.adjacency.get(src, set())
            neighbors_v = self.adjacency.get(dst, set())
            
            # 공통 이웃 & 합집합
            common = neighbors_u & neighbors_v
            union = neighbors_u | neighbors_v
            
            if len(union) == 0:
                scores.append(0.0)
            else:
                # Jaccard: 교집합 / 합집합
                score = len(common) / len(union)
                scores.append(score)
        
        return np.array(scores, dtype=np.float32)
    
    def compute_all_benchmarks(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        모든 벤치마크 알고리즘 점수 계산
        
        Returns
        -------
        scores : Dict[str, np.ndarray]
            {
                'PA': scores_pa,
                'RA': scores_ra,
                'JC': scores_jc
            }
        """
        logger.info(f"🔍 벤치마크 점수 계산 중... ({len(src_nodes):,}개 엣지)")
        
        scores = {
            'PA': self.preferential_attachment(src_nodes, dst_nodes),
            'RA': self.resource_allocation(src_nodes, dst_nodes),
            'JC': self.jaccard_coefficient(src_nodes, dst_nodes)
        }
        
        logger.info(f"✅ 벤치마크 점수 계산 완료")
        
        return scores


# ============================================================
# 유틸리티 함수
# ============================================================

def evaluate_benchmarks(
    edge_index: torch.Tensor,
    num_nodes: int,
    test_pos_edges: torch.Tensor,
    test_neg_edges: torch.Tensor,
    k_list: list = [10, 50, 100]
) -> Dict[str, Dict[str, float]]:
    """
    벤치마크 알고리즘 평가
    
    Parameters
    ----------
    edge_index : torch.Tensor [2, E]
        학습 그래프
    num_nodes : int
    test_pos_edges : torch.Tensor [2, E_pos]
    test_neg_edges : torch.Tensor [2, E_neg]
    k_list : list
    
    Returns
    -------
    results : Dict[str, Dict[str, float]]
        {
            'PA': {'recall@10': 0.3, 'recall@50': 0.5, ...},
            'RA': {...},
            'JC': {...}
        }
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔍 벤치마크 알고리즘 평가")
    logger.info("=" * 70)
    
    # 벤치마크 초기화
    benchmarks = LinkPredictionBenchmarks(edge_index, num_nodes)
    
    # Test 엣지
    src_test = torch.cat([test_pos_edges[0], test_neg_edges[0]])
    dst_test = torch.cat([test_pos_edges[1], test_neg_edges[1]])
    labels = torch.cat([
        torch.ones(test_pos_edges.shape[1]),
        torch.zeros(test_neg_edges.shape[1])
    ]).numpy()
    
    # 모든 벤치마크 점수
    all_scores = benchmarks.compute_all_benchmarks(src_test, dst_test)
    
    # 각 알고리즘별 평가
    results = {}
    
    for method_name, scores in all_scores.items():
        method_results = {}
        
        # Recall@K
        for k in k_list:
            # 상위 K개 인덱스
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_labels = labels[top_k_indices]
            
            # Recall@K
            recall_k = top_k_labels.sum() / labels.sum()
            method_results[f'recall@{k}'] = recall_k
        
        # MRR (Mean Reciprocal Rank)
        # Positive 엣지들의 랭킹 위치
        pos_indices = np.where(labels == 1)[0]
        
        # 점수 기준 랭킹 (내림차순)
        ranking = np.argsort(scores)[::-1]
        rank_dict = {idx: rank + 1 for rank, idx in enumerate(ranking)}
        
        # Positive 엣지들의 rank
        pos_ranks = [rank_dict[idx] for idx in pos_indices]
        
        # Reciprocal Rank
        reciprocal_ranks = [1.0 / rank for rank in pos_ranks]
        mrr = np.mean(reciprocal_ranks)
        method_results['MRR'] = mrr
        
        results[method_name] = method_results
        
        logger.info(f"\n📊 {method_name} 결과:")
        for metric, value in method_results.items():
            logger.info(f"   - {metric}: {value:.4f}")
    
    logger.info("=" * 70)
    
    return results
