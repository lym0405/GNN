"""
Constraint Checker: 제약 조건 검증 모듈
======================================

재배선 시 만족해야 하는 제약 조건을 검증합니다:
1. Self-loop 방지
2. Degree 제한 (outdegree, indegree)
3. Recipe 유사도 임계값
4. 용량 비율 제한
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """
    제약 조건 검증기
    
    Parameters
    ----------
    edge_index : np.ndarray [2, num_edges]
        기존 엣지 인덱스
    node_features : np.ndarray [N, D]
        노드 특성 행렬
    node_df : pd.DataFrame
        노드 데이터프레임
    max_supplier_outdegree : int
        공급자 최대 outdegree (기본값: 10)
    max_buyer_indegree : int
        구매자 최대 indegree (기본값: 10)
    recipe_similarity_threshold : float
        레시피 유사도 최소 임계값 (기본값: 0.7)
    capacity_ratio_min : float
        용량 비율 최소값 (기본값: 0.5)
    capacity_ratio_max : float
        용량 비율 최대값 (기본값: 2.0)
    """
    
    def __init__(
        self,
        edge_index: np.ndarray,
        node_features: np.ndarray,
        node_df: pd.DataFrame,
        max_supplier_outdegree: int = 10,
        max_buyer_indegree: int = 10,
        recipe_similarity_threshold: float = 0.7,
        capacity_ratio_min: float = 0.5,
        capacity_ratio_max: float = 2.0
    ):
        self.edge_index = edge_index
        self.node_features = node_features
        self.node_df = node_df
        
        self.max_supplier_outdegree = max_supplier_outdegree
        self.max_buyer_indegree = max_buyer_indegree
        self.recipe_similarity_threshold = recipe_similarity_threshold
        self.capacity_ratio_min = capacity_ratio_min
        self.capacity_ratio_max = capacity_ratio_max
        
        self.num_nodes = node_features.shape[0]
        
        # Degree 카운터 초기화
        self.current_in_degree = np.bincount(edge_index[1], minlength=self.num_nodes)
        self.current_out_degree = np.bincount(edge_index[0], minlength=self.num_nodes)
        
        # Recipe 벡터 추출
        self.recipe_vectors = self._extract_recipe_vectors(node_df)
        
        # 용량 데이터 추출
        self.capacities = self._extract_capacities(node_df)
        
        logger.info("ConstraintChecker 초기화")
        logger.info(f"  - 노드 수: {self.num_nodes:,}")
        logger.info(f"  - 최대 supplier outdegree: {max_supplier_outdegree}")
        logger.info(f"  - 최대 buyer indegree: {max_buyer_indegree}")
        logger.info(f"  - Recipe 유사도 임계값: {recipe_similarity_threshold}")
        logger.info(f"  - 용량 비율 범위: [{capacity_ratio_min}, {capacity_ratio_max}]")
    
    def check_constraints(
        self,
        supplier: int,
        buyer: int,
        verbose: bool = False
    ) -> bool:
        """
        모든 제약 조건 검증
        
        Parameters
        ----------
        supplier : int
            공급자 노드
        buyer : int
            구매자 노드
        verbose : bool
            상세 로그 출력 여부
        
        Returns
        -------
        is_valid : bool
            제약 조건 만족 여부
        """
        # 1. Self-loop 체크
        if not self.check_no_self_loop(supplier, buyer):
            if verbose:
                logger.debug(f"  ✗ Self-loop: ({supplier}, {buyer})")
            return False
        
        # 2. Degree 제한 체크
        if not self.check_degree_constraints(supplier, buyer):
            if verbose:
                logger.debug(f"  ✗ Degree 제한: ({supplier}, {buyer})")
            return False
        
        # 3. Recipe 유사도 체크
        if not self.check_recipe_similarity(supplier, buyer):
            if verbose:
                logger.debug(f"  ✗ Recipe 유사도: ({supplier}, {buyer})")
            return False
        
        # 4. 용량 비율 체크
        if not self.check_capacity_ratio(supplier, buyer):
            if verbose:
                logger.debug(f"  ✗ 용량 비율: ({supplier}, {buyer})")
            return False
        
        # 모든 제약 조건 만족
        return True
    
    def check_no_self_loop(
        self,
        supplier: int,
        buyer: int
    ) -> bool:
        """
        Self-loop 방지
        
        Parameters
        ----------
        supplier : int
        buyer : int
        
        Returns
        -------
        is_valid : bool
        """
        return supplier != buyer
    
    def check_degree_constraints(
        self,
        supplier: int,
        buyer: int
    ) -> bool:
        """
        Degree 제한 체크
        
        Parameters
        ----------
        supplier : int
        buyer : int
        
        Returns
        -------
        is_valid : bool
        """
        # Supplier의 outdegree 체크
        if self.current_out_degree[supplier] >= self.max_supplier_outdegree:
            return False
        
        # Buyer의 indegree 체크
        if self.current_in_degree[buyer] >= self.max_buyer_indegree:
            return False
        
        return True
    
    def check_recipe_similarity(
        self,
        supplier: int,
        buyer: int
    ) -> bool:
        """
        Recipe 유사도 체크
        
        Parameters
        ----------
        supplier : int
        buyer : int
        
        Returns
        -------
        is_valid : bool
        """
        recipe_s = self.recipe_vectors[supplier]
        recipe_b = self.recipe_vectors[buyer]
        
        # Cosine similarity
        similarity = self._cosine_similarity(recipe_s, recipe_b)
        
        # 임계값 이상이어야 함
        return similarity >= self.recipe_similarity_threshold
    
    def check_capacity_ratio(
        self,
        supplier: int,
        buyer: int
    ) -> bool:
        """
        용량 비율 체크
        
        Parameters
        ----------
        supplier : int
        buyer : int
        
        Returns
        -------
        is_valid : bool
        """
        supplier_capacity = self.capacities[supplier]
        buyer_capacity = self.capacities[buyer]
        
        if buyer_capacity < 1e-6:
            # Buyer 용량이 거의 없으면 통과
            return True
        
        # 용량 비율
        ratio = supplier_capacity / buyer_capacity
        
        # 범위 체크
        return self.capacity_ratio_min <= ratio <= self.capacity_ratio_max
    
    def add_edge(
        self,
        supplier: int,
        buyer: int
    ):
        """
        엣지 추가 (Degree 카운터 업데이트)
        
        Parameters
        ----------
        supplier : int
        buyer : int
        """
        self.current_out_degree[supplier] += 1
        self.current_in_degree[buyer] += 1
    
    def reset_degrees(self):
        """
        Degree 카운터 초기화
        """
        self.current_in_degree = np.bincount(
            self.edge_index[1], minlength=self.num_nodes
        )
        self.current_out_degree = np.bincount(
            self.edge_index[0], minlength=self.num_nodes
        )
        logger.info("Degree 카운터 초기화 완료")
    
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
        """
        # Recipe 컬럼 찾기
        recipe_cols = [col for col in node_df.columns if 'recipe' in col.lower()]
        
        if len(recipe_cols) == 0:
            logger.warning("Recipe 컬럼을 찾지 못함, 더미 벡터 사용")
            return np.ones((len(node_df), 10))
        
        # Recipe 값 추출
        recipe_values = node_df[recipe_cols].values
        recipe_values = np.nan_to_num(recipe_values, nan=0.0)
        
        # L2 정규화
        recipe_vectors = self._normalize_l2(recipe_values)
        
        return recipe_vectors
    
    def _extract_capacities(
        self,
        node_df: pd.DataFrame
    ) -> np.ndarray:
        """
        용량 데이터 추출
        
        Parameters
        ----------
        node_df : pd.DataFrame
        
        Returns
        -------
        capacities : np.ndarray [N]
        """
        revenue_candidates = ['revenue', '매출', 'sales', 'tg_2024_final']
        
        for col_name in revenue_candidates:
            if col_name in node_df.columns:
                capacities = node_df[col_name].values
                return np.nan_to_num(capacities, nan=0.0, posinf=0.0, neginf=0.0)
            
            for col in node_df.columns:
                if col_name.lower() in col.lower():
                    capacities = node_df[col].values
                    return np.nan_to_num(capacities, nan=0.0, posinf=0.0, neginf=0.0)
        
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
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)
        
        return similarity
    
    def _normalize_l2(
        self,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        L2 정규화
        
        Parameters
        ----------
        vectors : np.ndarray [N, D]
        
        Returns
        -------
        normalized : np.ndarray [N, D]
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normalized = vectors / norms
        return normalized
    
    def get_constraint_violations(
        self,
        supplier: int,
        buyer: int
    ) -> dict:
        """
        제약 조건 위반 상세 정보
        
        Parameters
        ----------
        supplier : int
        buyer : int
        
        Returns
        -------
        violations : dict
            위반 정보
        """
        violations = {}
        
        # Self-loop
        violations['self_loop'] = (supplier == buyer)
        
        # Degree
        violations['supplier_outdegree_exceeded'] = (
            self.current_out_degree[supplier] >= self.max_supplier_outdegree
        )
        violations['buyer_indegree_exceeded'] = (
            self.current_in_degree[buyer] >= self.max_buyer_indegree
        )
        
        # Recipe similarity
        recipe_sim = self._cosine_similarity(
            self.recipe_vectors[supplier],
            self.recipe_vectors[buyer]
        )
        violations['recipe_similarity'] = recipe_sim
        violations['recipe_similarity_ok'] = (
            recipe_sim >= self.recipe_similarity_threshold
        )
        
        # Capacity ratio
        if self.capacities[buyer] > 1e-6:
            capacity_ratio = self.capacities[supplier] / self.capacities[buyer]
            violations['capacity_ratio'] = capacity_ratio
            violations['capacity_ratio_ok'] = (
                self.capacity_ratio_min <= capacity_ratio <= self.capacity_ratio_max
            )
        else:
            violations['capacity_ratio'] = None
            violations['capacity_ratio_ok'] = True
        
        return violations


if __name__ == "__main__":
    # 간단한 테스트
    import pandas as pd
    
    # 더미 데이터
    edge_index = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 0]
    ])
    
    node_features = np.random.randn(4, 10)
    
    node_df = pd.DataFrame({
        'node_id': [0, 1, 2, 3],
        'revenue': [1e6, 2e6, 3e6, 4e6],
        'recipe_0': [1, 0, 1, 0],
        'recipe_1': [0, 1, 0, 1]
    })
    
    checker = ConstraintChecker(
        edge_index=edge_index,
        node_features=node_features,
        node_df=node_df,
        max_supplier_outdegree=5,
        max_buyer_indegree=5,
        recipe_similarity_threshold=0.5
    )
    
    # 제약 조건 체크
    print("Testing constraints:")
    print(f"  (0, 1) valid: {checker.check_constraints(0, 1)}")
    print(f"  (0, 0) valid (self-loop): {checker.check_constraints(0, 0)}")
    
    # 위반 상세 정보
    violations = checker.get_constraint_violations(0, 1)
    print(f"\nViolations for (0, 1): {violations}")
    
    print("\nModule loaded successfully!")
