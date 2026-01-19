"""
Buffer Calculator: 충격완충력 계산 모듈
======================================

Buffer(v) = f(z_v) × 1/(TIS_v + ε)

충격완충력은 기업의 기초 체력과 관세 노출도의 역수를 곱하여 계산합니다.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BufferCalculator:
    """
    충격완충력 계산기
    
    Buffer(v) = f(z_v) × 1/(TIS_v + ε)
    
    Parameters
    ----------
    edge_index : np.ndarray [2, num_edges]
        엣지 인덱스
    shock_threshold : float
        충격 임계값 (기본값: 0.3)
    epsilon : float
        분모 0 방지 (기본값: 1e-6)
    """
    
    def __init__(
        self,
        edge_index: np.ndarray,
        shock_threshold: float = 0.3,
        epsilon: float = 1e-6
    ):
        self.edge_index = edge_index
        self.shock_threshold = shock_threshold
        self.epsilon = epsilon
        
        # 노드 수 추출
        self.num_nodes = int(edge_index.max() + 1)
        
        # Degree 계산 (캐싱)
        self.in_degree = np.bincount(edge_index[1], minlength=self.num_nodes)
        self.out_degree = np.bincount(edge_index[0], minlength=self.num_nodes)
        self.total_degree = self.in_degree + self.out_degree
        
        logger.info("BufferCalculator 초기화")
        logger.info(f"  - 노드 수: {self.num_nodes:,}")
        logger.info(f"  - 엣지 수: {edge_index.shape[1]:,}")
        logger.info(f"  - Shock threshold: {shock_threshold}")
        logger.info(f"  - Epsilon: {epsilon}")
    
    def compute_buffer(
        self,
        buyer_node: Optional[int] = None,
        supplier_node: Optional[int] = None,
        tis_scores: Optional[np.ndarray] = None,
        node_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        충격완충력 계산
        
        Buffer(v) = f(z_v) × 1/(TIS_v + ε)
        
        Parameters
        ----------
        buyer_node : int, optional
            특정 buyer 노드 (None이면 모든 노드)
        supplier_node : int, optional
            특정 supplier 노드
        tis_scores : np.ndarray, optional
            TIS 점수 [N]
        node_df : pd.DataFrame, optional
            노드 특성 데이터프레임
        
        Returns
        -------
        buffer : np.ndarray or float
            충격완충력 점수
        """
        # 모든 노드의 Buffer 계산
        if buyer_node is None and supplier_node is None:
            if tis_scores is None or node_df is None:
                raise ValueError("tis_scores와 node_df가 필요합니다")
            
            # 기초 체력 계산
            f_z = self._compute_fundamental_strength(node_df)
            
            # TIS 페널티
            tis_penalty = 1.0 / (tis_scores + self.epsilon)
            
            # 충격완충력
            buffer = f_z * tis_penalty
            
            return buffer
        
        # 특정 노드 쌍의 Buffer 계산
        if buyer_node is not None and supplier_node is not None:
            if tis_scores is None or node_df is None:
                raise ValueError("tis_scores와 node_df가 필요합니다")
            
            # Supplier의 기초 체력
            f_z_supplier = self._compute_fundamental_strength(
                node_df.iloc[[supplier_node]]
            )[0]
            
            # Supplier의 TIS 페널티
            tis_penalty_supplier = 1.0 / (tis_scores[supplier_node] + self.epsilon)
            
            # Buffer
            buffer = f_z_supplier * tis_penalty_supplier
            
            return buffer
        
        raise ValueError("buyer_node와 supplier_node를 모두 제공하거나 모두 None이어야 합니다")
    
    def _compute_fundamental_strength(
        self,
        node_df: pd.DataFrame
    ) -> np.ndarray:
        """
        기초 체력 계산
        
        f(z_v) = w_revenue × revenue + w_assets × assets + w_profit × profit
        
        Parameters
        ----------
        node_df : pd.DataFrame
            노드 특성 데이터프레임
        
        Returns
        -------
        f_z : np.ndarray
            정규화된 기초 체력
        """
        # 가중치
        w_revenue = 0.4
        w_assets = 0.3
        w_profit = 0.3
        
        # 재무 컬럼 찾기 (유연한 매칭)
        revenue = self._extract_financial_column(
            node_df, ['revenue', '매출', 'sales', 'tg_2024_final']
        )
        assets = self._extract_financial_column(
            node_df, ['total_assets', 'assets', '자산', 'asset']
        )
        profit = self._extract_financial_column(
            node_df, ['operating_profit', 'profit', '영업이익', 'income']
        )
        
        # 정규화 (0-1 범위)
        revenue_norm = self._normalize(revenue)
        assets_norm = self._normalize(assets)
        profit_norm = self._normalize(profit)
        
        # 가중 합
        f_z = (
            w_revenue * revenue_norm +
            w_assets * assets_norm +
            w_profit * profit_norm
        )
        
        return f_z
    
    def _extract_financial_column(
        self,
        node_df: pd.DataFrame,
        candidates: list
    ) -> np.ndarray:
        """
        재무 컬럼 추출 (다중 후보 지원)
        
        Parameters
        ----------
        node_df : pd.DataFrame
        candidates : list
            후보 컬럼명 리스트
        
        Returns
        -------
        values : np.ndarray
            추출된 값
        """
        for col_name in candidates:
            # 정확한 매칭
            if col_name in node_df.columns:
                values = node_df[col_name].values
                return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 부분 매칭
            for col in node_df.columns:
                if col_name.lower() in col.lower():
                    values = node_df[col].values
                    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 컬럼을 찾지 못하면 더미 값
        logger.warning(f"컬럼을 찾지 못함: {candidates}, 더미 값(1.0) 사용")
        return np.ones(len(node_df))
    
    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Min-Max 정규화 (0-1 범위)
        
        Parameters
        ----------
        values : np.ndarray
        
        Returns
        -------
        normalized : np.ndarray
        """
        min_val = values.min()
        max_val = values.max()
        
        if max_val - min_val < 1e-10:
            # 모든 값이 동일하면 0.5 반환
            return np.full_like(values, 0.5, dtype=float)
        
        normalized = (values - min_val) / (max_val - min_val)
        return normalized
    
    def compute_shock_propagation(
        self,
        source_nodes: np.ndarray,
        shock_magnitude: float = 1.0
    ) -> np.ndarray:
        """
        충격 전파 시뮬레이션 (선택적 기능)
        
        Parameters
        ----------
        source_nodes : np.ndarray
            충격 시작 노드들
        shock_magnitude : float
            충격 크기
        
        Returns
        -------
        propagated_shock : np.ndarray
            각 노드의 충격 크기
        """
        logger.info(f"충격 전파 시뮬레이션: {len(source_nodes)}개 시작 노드")
        
        # 초기 충격
        shock = np.zeros(self.num_nodes)
        shock[source_nodes] = shock_magnitude
        
        # 단순 전파 (1-hop)
        # 실제로는 더 복잡한 전파 모델 사용 가능
        for src in source_nodes:
            # src에서 나가는 엣지
            out_neighbors = self.edge_index[1, self.edge_index[0] == src]
            
            if len(out_neighbors) > 0:
                # 충격을 이웃에 분배
                shock[out_neighbors] += shock_magnitude / len(out_neighbors)
        
        logger.info(f"  - 충격 받은 노드: {np.sum(shock > 0):,}")
        logger.info(f"  - 평균 충격: {shock.mean():.4f}")
        
        return shock


if __name__ == "__main__":
    # 간단한 테스트
    import pandas as pd
    
    # 더미 데이터
    edge_index = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 0]
    ])
    
    node_df = pd.DataFrame({
        'revenue': [1e6, 2e6, 3e6, 4e6],
        'total_assets': [5e6, 6e6, 7e6, 8e6],
        'operating_profit': [1e5, 2e5, 3e5, 4e5]
    })
    
    tis_scores = np.array([0.8, 0.6, 0.4, 0.2])
    
    calc = BufferCalculator(edge_index)
    buffer = calc.compute_buffer(tis_scores=tis_scores, node_df=node_df)
    
    print("Buffer scores:", buffer)
    print("Module loaded successfully!")
