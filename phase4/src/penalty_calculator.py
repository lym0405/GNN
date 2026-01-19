"""
Penalty Calculator: 재고/용량 패널티 계산 모듈
==============================================

Penalty_inv = α × RecipeMismatch + β × CapacityShortage

레시피 불일치와 용량 부족에 대한 패널티를 계산합니다.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PenaltyCalculator:
    """
    재고/용량 패널티 계산기
    
    Penalty_inv = α × RecipeMismatch + β × CapacityShortage
    
    Parameters
    ----------
    node_features : np.ndarray [N, D]
        노드 특성 행렬
    node_df : pd.DataFrame
        노드 데이터프레임 (recipe 컬럼 포함)
    alpha : float
        레시피 불일치 가중치 (기본값: 0.3)
    beta : float
        용량 부족 가중치 (기본값: 0.2)
    """
    
    def __init__(
        self,
        node_features: np.ndarray,
        node_df: pd.DataFrame,
        alpha: float = 0.3,
        beta: float = 0.2
    ):
        self.node_features = node_features
        self.node_df = node_df
        self.alpha = alpha
        self.beta = beta
        
        self.num_nodes = node_features.shape[0]
        
        # Recipe 벡터 추출
        self.recipe_vectors = self._extract_recipe_vectors(node_df)
        
        # 용량 데이터 추출 (매출 기준)
        self.capacities = self._extract_capacities(node_df)
        
        logger.info("PenaltyCalculator 초기화")
        logger.info(f"  - 노드 수: {self.num_nodes:,}")
        logger.info(f"  - Recipe 차원: {self.recipe_vectors.shape[1]}")
        logger.info(f"  - Alpha (레시피): {alpha}")
        logger.info(f"  - Beta (용량): {beta}")
    
    def compute_penalty(
        self,
        supplier: int,
        buyer: int
    ) -> float:
        """
        전체 패널티 계산
        
        Penalty = α × RecipeMismatch + β × CapacityShortage
        
        Parameters
        ----------
        supplier : int
            공급자 노드 인덱스
        buyer : int
            구매자 노드 인덱스
        
        Returns
        -------
        penalty : float
            총 패널티
        """
        # 레시피 불일치
        recipe_mismatch = self.compute_recipe_mismatch(supplier, buyer)
        
        # 용량 부족
        capacity_shortage = self.compute_capacity_shortage(supplier, buyer)
        
        # 총 패널티
        penalty = self.alpha * recipe_mismatch + self.beta * capacity_shortage
        
        return penalty
    
    def compute_recipe_mismatch(
        self,
        supplier: int,
        buyer: int
    ) -> float:
        """
        레시피 불일치 계산 (Cosine Distance)
        
        RecipeMismatch = 1 - CosineSimilarity
        
        Parameters
        ----------
        supplier : int
        buyer : int
        
        Returns
        -------
        mismatch : float
            레시피 불일치 (0~1, 높을수록 불일치)
        """
        recipe_s = self.recipe_vectors[supplier]
        recipe_b = self.recipe_vectors[buyer]
        
        # Cosine Similarity
        similarity = self._cosine_similarity(recipe_s, recipe_b)
        
        # Distance (1 - similarity)
        mismatch = 1.0 - similarity
        
        return mismatch
    
    def compute_capacity_shortage(
        self,
        supplier: int,
        buyer: int
    ) -> float:
        """
        용량 부족 계산
        
        CapacityShortage = max(0, required - available) / required
        
        Parameters
        ----------
        supplier : int
        buyer : int
        
        Returns
        -------
        shortage : float
            용량 부족 비율 (0~1)
        """
        # Buyer가 필요한 용량 (매출 기준)
        required = self.capacities[buyer]
        
        # Supplier가 제공 가능한 용량 (매출 기준)
        available = self.capacities[supplier]
        
        if required < 1e-6:
            # 필요량이 거의 없으면 부족 없음
            return 0.0
        
        # 부족 비율
        shortage = max(0.0, required - available) / required
        
        return shortage
    
    def _extract_recipe_vectors(
        self,
        node_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Recipe 벡터 추출
        
        Parameters
        ----------
        node_df : pd.DataFrame
        
        Returns
        -------
        recipe_vectors : np.ndarray [N, R]
            레시피 벡터 (R: recipe 차원)
        """
        # Recipe 컬럼 찾기
        recipe_cols = [col for col in node_df.columns if 'recipe' in col.lower()]
        
        if len(recipe_cols) == 0:
            logger.warning("Recipe 컬럼을 찾지 못함, 더미 벡터 사용")
            # 더미 벡터 (모두 동일)
            return np.ones((len(node_df), 10))
        
        logger.info(f"  - Recipe 컬럼 {len(recipe_cols)}개 발견")
        
        # Recipe 값 추출
        recipe_values = node_df[recipe_cols].values
        
        # NaN 처리
        recipe_values = np.nan_to_num(recipe_values, nan=0.0)
        
        # 정규화 (L2 norm)
        recipe_vectors = self._normalize_l2(recipe_values)
        
        return recipe_vectors
    
    def _extract_capacities(
        self,
        node_df: pd.DataFrame
    ) -> np.ndarray:
        """
        용량 데이터 추출 (매출 기준)
        
        Parameters
        ----------
        node_df : pd.DataFrame
        
        Returns
        -------
        capacities : np.ndarray [N]
            노드별 용량
        """
        # 매출 컬럼 찾기
        revenue_candidates = ['revenue', '매출', 'sales', 'tg_2024_final']
        
        for col_name in revenue_candidates:
            # 정확한 매칭
            if col_name in node_df.columns:
                capacities = node_df[col_name].values
                capacities = np.nan_to_num(capacities, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"  - 용량 데이터 컬럼: {col_name}")
                return capacities
            
            # 부분 매칭
            for col in node_df.columns:
                if col_name.lower() in col.lower():
                    capacities = node_df[col].values
                    capacities = np.nan_to_num(capacities, nan=0.0, posinf=0.0, neginf=0.0)
                    logger.info(f"  - 용량 데이터 컬럼: {col}")
                    return capacities
        
        # 찾지 못하면 더미 값
        logger.warning("용량 데이터를 찾지 못함, 더미 값 사용")
        return np.ones(len(node_df)) * 1e6
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Cosine Similarity 계산
        
        Parameters
        ----------
        vec1 : np.ndarray
        vec2 : np.ndarray
        
        Returns
        -------
        similarity : float
            코사인 유사도 (-1~1)
        """
        # L2 norm
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            # Zero vector
            return 0.0
        
        # Dot product
        dot_product = np.dot(vec1, vec2)
        
        # Cosine similarity
        similarity = dot_product / (norm1 * norm2)
        
        return similarity
    
    def _normalize_l2(
        self,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        L2 정규화 (row-wise)
        
        Parameters
        ----------
        vectors : np.ndarray [N, D]
        
        Returns
        -------
        normalized : np.ndarray [N, D]
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Zero vector 방지
        norms = np.where(norms < 1e-10, 1.0, norms)
        
        normalized = vectors / norms
        
        return normalized
    
    def compute_penalty_matrix(
        self,
        node_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        패널티 행렬 계산 (선택적 기능)
        
        Parameters
        ----------
        node_indices : np.ndarray, optional
            계산할 노드 인덱스 (None이면 모든 노드)
        
        Returns
        -------
        penalty_matrix : np.ndarray [N, N] or [K, K]
            패널티 행렬
        """
        if node_indices is None:
            node_indices = np.arange(self.num_nodes)
        
        K = len(node_indices)
        penalty_matrix = np.zeros((K, K))
        
        logger.info(f"패널티 행렬 계산: {K}×{K}")
        
        for i, supplier in enumerate(node_indices):
            for j, buyer in enumerate(node_indices):
                if supplier != buyer:
                    penalty_matrix[i, j] = self.compute_penalty(supplier, buyer)
        
        logger.info(f"  - 평균 패널티: {penalty_matrix.mean():.4f}")
        logger.info(f"  - 최대 패널티: {penalty_matrix.max():.4f}")
        
        return penalty_matrix
    
    def get_compatible_suppliers(
        self,
        buyer: int,
        max_recipe_mismatch: float = 0.5,
        min_capacity_ratio: float = 0.5
    ) -> np.ndarray:
        """
        호환 가능한 공급자 찾기
        
        Parameters
        ----------
        buyer : int
            구매자 노드
        max_recipe_mismatch : float
            최대 레시피 불일치
        min_capacity_ratio : float
            최소 용량 비율
        
        Returns
        -------
        compatible_suppliers : np.ndarray
            호환 가능한 공급자 인덱스
        """
        compatible = []
        buyer_capacity = self.capacities[buyer]
        
        for supplier in range(self.num_nodes):
            if supplier == buyer:
                continue
            
            # 레시피 체크
            mismatch = self.compute_recipe_mismatch(supplier, buyer)
            if mismatch > max_recipe_mismatch:
                continue
            
            # 용량 체크
            supplier_capacity = self.capacities[supplier]
            if buyer_capacity > 1e-6:
                capacity_ratio = supplier_capacity / buyer_capacity
                if capacity_ratio < min_capacity_ratio:
                    continue
            
            compatible.append(supplier)
        
        return np.array(compatible)


if __name__ == "__main__":
    # 간단한 테스트
    import pandas as pd
    
    # 더미 데이터
    node_features = np.random.randn(4, 10)
    
    node_df = pd.DataFrame({
        'node_id': [0, 1, 2, 3],
        'revenue': [1e6, 2e6, 3e6, 4e6],
        'recipe_0': [1, 0, 1, 0],
        'recipe_1': [0, 1, 0, 1],
        'recipe_2': [1, 1, 0, 0]
    })
    
    calc = PenaltyCalculator(node_features, node_df)
    
    # 패널티 계산
    penalty = calc.compute_penalty(0, 1)
    print(f"Penalty(0→1): {penalty:.4f}")
    
    # 레시피 불일치
    mismatch = calc.compute_recipe_mismatch(0, 1)
    print(f"Recipe Mismatch(0→1): {mismatch:.4f}")
    
    # 호환 공급자
    compatible = calc.get_compatible_suppliers(1)
    print(f"Compatible suppliers for node 1: {compatible}")
    
    print("Module loaded successfully!")
