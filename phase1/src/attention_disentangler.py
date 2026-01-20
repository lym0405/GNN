"""
Attention-based Recipe Disentangler
====================================
Query-Key Attention으로 기업별 다중 상품 레시피 분리

핵심 아이디어:
- Query: 구매자의 업종/상품 정보
- Key: 공급자의 RAS 벡터 (기술계수)
- Value: 공급자가 생산하는 상품별 레시피
- Attention: Query와 Key의 유사도로 Value를 가중합

예시:
- 구매자 A: 자동차 제조 (Query)
- 공급자 B: 철강 60%, 화학 40% 생산 (Key/Value)
- 구매자 A는 주로 철강을 필요로 하므로 → 철강 레시피 가중치 ↑
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionRecipeDisentangler:
    """
    Attention 기반 레시피 분리 모듈
    
    Parameters
    ----------
    B_matrix : np.ndarray, shape (N, 33)
        기업별 기술계수 행렬 (RAS 벡터)
    firm_products : Dict[str, List[Tuple[str, float]]]
        사업자번호 → [(IO코드, 가중치), ...] 매핑
    firm_ids : List[str]
        기업 ID 리스트 (B 행렬 행 순서와 동일)
    """
    
    def __init__(
        self,
        B_matrix: np.ndarray,
        firm_products: Dict[str, List[Tuple[str, float]]],
        firm_ids: List[str]
    ):
        self.B = B_matrix
        self.firm_products = firm_products
        self.firm_ids = firm_ids
        self.N = len(firm_ids)
        
        # 기업별 IO 코드 → 인덱스 매핑
        self.firm_product_indices = {}
        for biz_id, products in firm_products.items():
            # IO 코드를 0-based 인덱스로 변환
            indices = []
            weights = []
            for code, weight in products:
                try:
                    idx = int(code) - 1  # '01' → 0, '02' → 1, ...
                    if 0 <= idx < 33:
                        indices.append(idx)
                        weights.append(weight)
                except:
                    continue
            
            if indices:
                self.firm_product_indices[biz_id] = (indices, weights)
        
        logger.info(f"✅ AttentionRecipeDisentangler 초기화")
        logger.info(f"   - 기업 수: {self.N}")
        logger.info(f"   - 다중 상품 기업 수: {len(self.firm_product_indices)}")
    
    def disentangle_recipes(
        self,
        H_matrix,
        temperature: float = 1.0,
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Attention으로 기업별 레시피 분리
        
        Parameters
        ----------
        H_matrix : sparse matrix, shape (N, N)
            거래 네트워크 (H[i,j] = 기업 i가 j로부터 구매)
        temperature : float
            Attention softmax temperature (작을수록 sharp)
        alpha : float
            Prior (B 행렬) vs Attention 가중치 (0.7이면 70% Attention, 30% Prior)
        
        Returns
        -------
        recipes : np.ndarray, shape (N, 33)
            분리된 레시피 행렬
        """
        logger.info(f"🔍 Attention 기반 레시피 분리 시작...")
        logger.info(f"   - Temperature: {temperature}")
        logger.info(f"   - Alpha (Attention weight): {alpha}")
        
        recipes = np.zeros((self.N, 33), dtype=np.float32)
        
        # 배치 처리
        batch_size = 5000
        for start in range(0, self.N, batch_size):
            end = min(start + batch_size, self.N)
            
            for i in range(start, end):
                biz_id = self.firm_ids[i]
                
                # 1) Query: 구매자의 상품 분포
                query = self._get_query(biz_id)
                
                # 2) 공급자 찾기
                suppliers = H_matrix[i, :].nonzero()[1]
                purchase_amounts = H_matrix[i, suppliers].toarray()[0]
                
                if len(suppliers) == 0:
                    # 공급자 없으면 Prior (B 행렬) 사용
                    recipes[i, :] = self.B[i, :]
                    continue
                
                # 3) Key: 공급자들의 RAS 벡터 (B 행렬)
                keys = self.B[suppliers, :]  # (num_suppliers, 33)
                
                # 4) Attention 계산
                attention_scores = self._compute_attention(
                    query, keys, temperature
                )
                
                # 5) 거래 금액과 Attention 가중치 결합
                combined_weights = (
                    alpha * attention_scores +
                    (1 - alpha) * (purchase_amounts / purchase_amounts.sum())
                )
                
                # 6) Value: 공급자들의 레시피를 가중합
                recipe = (combined_weights[:, None] * keys).sum(axis=0)
                
                # 7) Prior와 블렌딩 (안정성 향상)
                recipes[i, :] = 0.8 * recipe + 0.2 * self.B[i, :]
            
            if (start // batch_size + 1) % 5 == 0:
                logger.info(f"   진행: {end}/{self.N} ({end/self.N*100:.1f}%)")
        
        # 정규화 (각 행의 합이 1)
        row_sums = recipes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
        recipes = recipes / row_sums
        
        logger.info(f"✅ 레시피 분리 완료")
        
        return recipes
    
    def _get_query(self, biz_id: str) -> np.ndarray:
        """
        구매자의 Query 벡터 생성 (33차원)
        
        Query = 기업이 생산하는 상품의 One-hot (또는 Multi-hot) 벡터
        
        Returns
        -------
        query : np.ndarray, shape (33,)
        """
        query = np.zeros(33, dtype=np.float32)
        
        if biz_id in self.firm_product_indices:
            indices, weights = self.firm_product_indices[biz_id]
            for idx, weight in zip(indices, weights):
                query[idx] = weight
        else:
            # 상품 정보 없으면 균등 분포
            query[:] = 1.0 / 33
        
        # 정규화
        query_sum = query.sum()
        if query_sum > 0:
            query = query / query_sum
        
        return query
    
    def _compute_attention(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """
        Attention 점수 계산
        
        Attention(Q, K) = softmax(Q · K^T / temperature)
        
        Parameters
        ----------
        query : np.ndarray, shape (33,)
        keys : np.ndarray, shape (num_suppliers, 33)
        temperature : float
        
        Returns
        -------
        attention_weights : np.ndarray, shape (num_suppliers,)
        """
        # 내적 계산 (코사인 유사도)
        scores = np.dot(keys, query)  # (num_suppliers,)
        
        # Temperature scaling
        scores = scores / temperature
        
        # Softmax
        exp_scores = np.exp(scores - scores.max())  # 수치 안정성
        attention_weights = exp_scores / exp_scores.sum()
        
        return attention_weights


def create_disentangled_recipes(
    H_matrix,
    B_matrix: np.ndarray,
    firm_products: Dict[str, List[Tuple[str, float]]],
    firm_ids: List[str],
    method: str = 'attention',
    **kwargs
) -> np.ndarray:
    """
    레시피 분리 헬퍼 함수
    
    Parameters
    ----------
    H_matrix : sparse matrix
        거래 네트워크
    B_matrix : np.ndarray
        기술계수 행렬
    firm_products : Dict
        기업별 상품 매핑
    firm_ids : List[str]
        기업 ID 리스트
    method : str
        'attention' 또는 'simple'
    **kwargs
        추가 파라미터 (temperature, alpha 등)
    
    Returns
    -------
    recipes : np.ndarray, shape (N, 33)
    """
    if method == 'attention':
        disentangler = AttentionRecipeDisentangler(
            B_matrix, firm_products, firm_ids
        )
        recipes = disentangler.disentangle_recipes(
            H_matrix,
            temperature=kwargs.get('temperature', 1.0),
            alpha=kwargs.get('alpha', 0.7)
        )
    else:
        # Simple weighted average (기존 방식)
        from .inventory_module import ZeroShotInventoryModule
        module = ZeroShotInventoryModule(H_matrix, B_matrix, firm_ids)
        recipes = module.estimate_recipes(method='weighted')
    
    return recipes


if __name__ == "__main__":
    # 테스트
    N = 1000
    B = np.random.dirichlet(np.ones(33), size=N).astype(np.float32)
    
    # 더미 데이터
    firm_ids = [f"BIZ{i:06d}" for i in range(N)]
    firm_products = {
        fid: [('06', 0.6), ('09', 0.4)] if i % 2 == 0 else [('11', 1.0)]
        for i, fid in enumerate(firm_ids)
    }
    
    # 더미 H 행렬
    from scipy.sparse import random
    H = random(N, N, density=0.01, format='csr')
    
    disentangler = AttentionRecipeDisentangler(B, firm_products, firm_ids)
    recipes = disentangler.disentangle_recipes(H, temperature=0.5, alpha=0.8)
    
    print(f"✅ 레시피 생성 완료: {recipes.shape}")
    print(f"평균 엔트로피: {-np.sum(recipes * np.log(recipes + 1e-9), axis=1).mean():.3f}")
