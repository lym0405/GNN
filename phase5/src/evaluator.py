"""
Phase 5: 성능 평가 (Evaluator)
===============================
재배선 예측 결과를 실제 데이터와 비교 평가
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class Phase5Evaluator:
    """
    Historical Back-testing 평가
    
    Parameters
    ----------
    predicted_edges : List[Tuple[int, int, float]]
        예측된 엣지 리스트 [(src, dst, score), ...]
    actual_new_edges : Set[Tuple[int, int]]
        실제로 새로 형성된 엣지 (2020년 - 2018년)
    actual_removed_edges : Set[Tuple[int, int]]
        실제로 삭제된 엣지 (2018년 - 2020년)
    """
    
    def __init__(
        self,
        predicted_edges: List[Tuple[int, int, float]],
        actual_new_edges: Set[Tuple[int, int]],
        actual_removed_edges: Set[Tuple[int, int]]
    ):
        self.predicted_edges = predicted_edges
        self.actual_new_edges = actual_new_edges
        self.actual_removed_edges = actual_removed_edges
        
        # 예측 점수로 정렬
        self.predicted_edges = sorted(
            predicted_edges,
            key=lambda x: x[2],
            reverse=True
        )
        
        logger.info(f"✅ Phase5Evaluator 초기화")
        logger.info(f"   - 예측 엣지: {len(predicted_edges):,}개")
        logger.info(f"   - 실제 신규 엣지: {len(actual_new_edges):,}개")
        logger.info(f"   - 실제 삭제 엣지: {len(actual_removed_edges):,}개")
    
    def compute_hit_rate_at_k(
        self,
        k_list: List[int] = [10, 50, 100, 500, 1000]
    ) -> Dict[str, float]:
        """
        Hit Rate@K 계산
        
        Top-K 예측 중에서 실제 신규 엣지가 몇 개 포함되었는가?
        
        Returns
        -------
        hit_rates : Dict[str, float]
            {f'hit_rate@{k}': value}
        """
        hit_rates = {}
        
        for k in k_list:
            # Top-K 예측
            top_k_predictions = set(
                (src, dst) for src, dst, score in self.predicted_edges[:k]
            )
            
            # 실제 신규 엣지와 교집합
            hits = top_k_predictions & self.actual_new_edges
            
            hit_rate = len(hits) / min(k, len(self.actual_new_edges))
            hit_rates[f'hit_rate@{k}'] = hit_rate
            
            logger.info(f"   ✓ Hit Rate@{k}: {hit_rate:.4f} ({len(hits)}/{min(k, len(self.actual_new_edges))})")
        
        return hit_rates
    
    def compute_recall_at_k(
        self,
        k_list: List[int] = [10, 50, 100, 500, 1000]
    ) -> Dict[str, float]:
        """
        Recall@K 계산
        
        실제 신규 엣지 중에서 Top-K에 포함된 비율
        
        Returns
        -------
        recalls : Dict[str, float]
            {f'recall@{k}': value}
        """
        recalls = {}
        
        for k in k_list:
            # Top-K 예측
            top_k_predictions = set(
                (src, dst) for src, dst, score in self.predicted_edges[:k]
            )
            
            # Recall = (실제 & 예측) / 실제
            hits = top_k_predictions & self.actual_new_edges
            recall = len(hits) / len(self.actual_new_edges) if self.actual_new_edges else 0.0
            
            recalls[f'recall@{k}'] = recall
            
            logger.info(f"   ✓ Recall@{k}: {recall:.4f} ({len(hits)}/{len(self.actual_new_edges)})")
        
        return recalls
    
    def compute_precision_at_k(
        self,
        k_list: List[int] = [10, 50, 100, 500, 1000]
    ) -> Dict[str, float]:
        """
        Precision@K 계산
        
        Top-K 예측 중에서 실제로 맞춘 비율
        
        Returns
        -------
        precisions : Dict[str, float]
            {f'precision@{k}': value}
        """
        precisions = {}
        
        for k in k_list:
            # Top-K 예측
            top_k_predictions = set(
                (src, dst) for src, dst, score in self.predicted_edges[:k]
            )
            
            # Precision = (실제 & 예측) / 예측
            hits = top_k_predictions & self.actual_new_edges
            precision = len(hits) / k if k > 0 else 0.0
            
            precisions[f'precision@{k}'] = precision
            
            logger.info(f"   ✓ Precision@{k}: {precision:.4f} ({len(hits)}/{k})")
        
        return precisions
    
    def compute_all_metrics(
        self,
        k_list: List[int] = [10, 50, 100, 500, 1000]
    ) -> Dict[str, float]:
        """
        모든 메트릭 계산
        
        Returns
        -------
        metrics : Dict[str, float]
            모든 메트릭을 담은 dict
        """
        logger.info(f"\n📊 성능 평가 시작")
        
        metrics = {}
        
        # 1. Hit Rate@K
        logger.info(f"\n[1] Hit Rate@K")
        hit_rates = self.compute_hit_rate_at_k(k_list)
        metrics.update(hit_rates)
        
        # 2. Recall@K
        logger.info(f"\n[2] Recall@K")
        recalls = self.compute_recall_at_k(k_list)
        metrics.update(recalls)
        
        # 3. Precision@K
        logger.info(f"\n[3] Precision@K")
        precisions = self.compute_precision_at_k(k_list)
        metrics.update(precisions)
        
        # 4. 기타 통계
        metrics['total_predictions'] = len(self.predicted_edges)
        metrics['total_actual_new'] = len(self.actual_new_edges)
        metrics['total_actual_removed'] = len(self.actual_removed_edges)
        
        logger.info(f"\n✅ 평가 완료")
        
        return metrics


class ResilienceEvaluator:
    """
    공급망 안정성 (Resilience) 평가
    
    모델이 추천한 재배선이 실제로 더 나은 결과를 가져왔는지 시뮬레이션
    """
    
    def __init__(
        self,
        original_network: pd.DataFrame,
        actual_network: pd.DataFrame,
        predicted_network: pd.DataFrame
    ):
        self.original_network = original_network
        self.actual_network = actual_network
        self.predicted_network = predicted_network
        
        logger.info(f"✅ ResilienceEvaluator 초기화")
    
    def compute_total_sales_change(self) -> Dict[str, float]:
        """
        총 매출 변화 비교
        
        Returns
        -------
        results : Dict[str, float]
            {
                'original_sales': ...,
                'actual_sales': ...,
                'predicted_sales': ...,
                'actual_reduction_ratio': ...,
                'predicted_reduction_ratio': ...,
                'improvement': ...
            }
        """
        # 매출 컬럼 확인
        sales_col = None
        for col in ['총공급금액', 'total_sales', 'sales', '거래금액']:
            if col in self.original_network.columns:
                sales_col = col
                break
        
        if sales_col is None:
            logger.warning("⚠️  매출 컬럼 없음")
            return {}
        
        original_sales = self.original_network[sales_col].sum()
        actual_sales = self.actual_network[sales_col].sum() if sales_col in self.actual_network.columns else 0
        predicted_sales = self.predicted_network[sales_col].sum() if sales_col in self.predicted_network.columns else 0
        
        actual_reduction = (original_sales - actual_sales) / original_sales if original_sales > 0 else 0
        predicted_reduction = (original_sales - predicted_sales) / original_sales if original_sales > 0 else 0
        
        improvement = actual_reduction - predicted_reduction
        
        results = {
            'original_sales': original_sales,
            'actual_sales': actual_sales,
            'predicted_sales': predicted_sales,
            'actual_reduction_ratio': actual_reduction,
            'predicted_reduction_ratio': predicted_reduction,
            'improvement': improvement,
        }
        
        logger.info(f"   ✓ 원본 매출: {original_sales:,.0f}")
        logger.info(f"   ✓ 실제 매출: {actual_sales:,.0f} (감소: {actual_reduction:.2%})")
        logger.info(f"   ✓ 예측 매출: {predicted_sales:,.0f} (감소: {predicted_reduction:.2%})")
        logger.info(f"   ✓ 개선도: {improvement:.2%}")
        
        return results
    
    def compute_network_connectivity(self) -> Dict[str, int]:
        """
        네트워크 연결성 비교
        
        Returns
        -------
        results : Dict[str, int]
            {
                'original_edges': ...,
                'actual_edges': ...,
                'predicted_edges': ...
            }
        """
        results = {
            'original_edges': len(self.original_network),
            'actual_edges': len(self.actual_network),
            'predicted_edges': len(self.predicted_network),
        }
        
        logger.info(f"   ✓ 원본 엣지: {results['original_edges']:,}개")
        logger.info(f"   ✓ 실제 엣지: {results['actual_edges']:,}개")
        logger.info(f"   ✓ 예측 엣지: {results['predicted_edges']:,}개")
        
        return results


# ============================================================
# 유틸리티 함수
# ============================================================

def compare_networks(
    network_2018: pd.DataFrame,
    network_2020: pd.DataFrame
) -> Tuple[Set[Tuple], Set[Tuple]]:
    """
    두 시점의 네트워크를 비교하여 신규/삭제 엣지 추출
    
    Parameters
    ----------
    network_2018 : pd.DataFrame
        2018년 네트워크
    network_2020 : pd.DataFrame
        2020년 네트워크
    
    Returns
    -------
    new_edges : Set[Tuple[int, int]]
        2020년에 새로 생긴 엣지
    removed_edges : Set[Tuple[int, int]]
        2018년에서 사라진 엣지
    """
    # 엣지 컬럼 확인
    src_col_2018, dst_col_2018 = _get_edge_columns(network_2018)
    src_col_2020, dst_col_2020 = _get_edge_columns(network_2020)
    
    # 엣지 set 생성
    edges_2018 = set(zip(network_2018[src_col_2018], network_2018[dst_col_2018]))
    edges_2020 = set(zip(network_2020[src_col_2020], network_2020[dst_col_2020]))
    
    # 차집합
    new_edges = edges_2020 - edges_2018
    removed_edges = edges_2018 - edges_2020
    
    logger.info(f"   ✓ 신규 엣지: {len(new_edges):,}개")
    logger.info(f"   ✓ 삭제 엣지: {len(removed_edges):,}개")
    
    return new_edges, removed_edges


def _get_edge_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """엣지 컬럼명 찾기"""
    if '사업자등록번호' in df.columns and '거래처사업자등록번호' in df.columns:
        return '사업자등록번호', '거래처사업자등록번호'
    elif 'source' in df.columns and 'target' in df.columns:
        return 'source', 'target'
    elif 'src' in df.columns and 'dst' in df.columns:
        return 'src', 'dst'
    else:
        return df.columns[0], df.columns[1]
