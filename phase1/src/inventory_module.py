"""
Zero-Shot Inventory Module
===========================
거래 네트워크(H)와 기술계수(B)를 결합하여 기업별 생산함수(레시피) 추정

핵심 아이디어:
- H 행렬: "이 기업이 누구에게서 얼마를 샀는가"
- B 행렬: "이 기업이 속한 산업은 어떤 중간재를 쓰는가"
- 결합: 거래 금액을 산업별로 분해(disentangle)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotInventoryModule:
    """
    기업별 생산함수(레시피) 추정 모듈
    
    Parameters
    ----------
    H_matrix : csr_matrix or np.ndarray, shape (N, N)
        거래 네트워크 행렬 (H[i,j] = 기업 i가 기업 j로부터 구매한 금액)
    B_matrix : np.ndarray, shape (N, 33)
        기업별 기술계수 행렬
    firm_ids : list, length N
        기업 ID 리스트
    """
    
    def __init__(
        self,
        H_matrix: csr_matrix,
        B_matrix: np.ndarray,
        firm_ids: list
    ):
        self.H = H_matrix
        self.B = B_matrix
        self.firm_ids = firm_ids
        self.N = len(firm_ids)
        
        assert self.B.shape[0] == self.N, "B matrix rows must match firm count"
        assert self.B.shape[1] == 33, "B matrix must have 33 columns"
        
        logger.info(f"✅ ZeroShotInventoryModule 초기화")
        logger.info(f"   - 기업 수: {self.N}")
        logger.info(f"   - H 행렬 밀도: {self.H.nnz / (self.N ** 2) * 100:.4f}%")
    
    def estimate_recipes(self, method: str = 'weighted') -> np.ndarray:
        """
        기업별 33차원 레시피 추정
        
        Parameters
        ----------
        method : str
            - 'weighted': 거래 금액으로 가중 평균 (기본)
            - 'simple': 단순 평균
            - 'bayesian': 베이지안 추정 (B를 Prior로)
        
        Returns
        -------
        recipes : np.ndarray, shape (N, 33)
            각 기업의 33개 산업별 중간재 사용 비율
        """
        logger.info(f"🔍 레시피 추정 시작 (method={method})...")
        
        if method == 'weighted':
            recipes = self._weighted_estimation()
        elif method == 'simple':
            recipes = self._simple_estimation()
        elif method == 'bayesian':
            recipes = self._bayesian_estimation()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 후처리
        recipes = self._postprocess(recipes)
        
        logger.info(f"✅ 레시피 추정 완료")
        self._print_statistics(recipes)
        
        return recipes
    
    def _weighted_estimation(self) -> np.ndarray:
        """
        가중 평균 방식: 거래 금액에 비례하여 공급자의 레시피를 가중합
        
        Recipe[i, k] = Σ_j (H[i,j] * B[j,k]) / Σ_j H[i,j]
        
        해석: "내가 100억을 A사에서 샀고, A사가 철강을 30% 쓴다면,
               내 철강 사용량에 30억이 반영된다"
        """
        recipes = np.zeros((self.N, 33), dtype=np.float32)
        
        # H 행렬을 dense로 변환 (메모리가 충분하면)
        # 대용량일 경우 배치 처리 필요
        if self.N < 50000:
            H_dense = self.H.toarray()
            
            for i in range(self.N):
                purchases = H_dense[i, :]  # 내가 각 기업으로부터 산 금액
                total_purchase = purchases.sum()
                
                if total_purchase > 0:
                    # 공급자들의 레시피를 거래 금액으로 가중 평균
                    weighted_recipe = (purchases[:, None] * self.B).sum(axis=0)
                    recipes[i, :] = weighted_recipe / total_purchase
                else:
                    # 구매 내역이 없으면 자기 산업의 기술계수 사용
                    recipes[i, :] = self.B[i, :]
        else:
            # 메모리 절약을 위한 배치 처리
            batch_size = 10000
            for start in range(0, self.N, batch_size):
                end = min(start + batch_size, self.N)
                H_batch = self.H[start:end, :].toarray()
                
                for i in range(H_batch.shape[0]):
                    purchases = H_batch[i, :]
                    total_purchase = purchases.sum()
                    
                    if total_purchase > 0:
                        weighted_recipe = (purchases[:, None] * self.B).sum(axis=0)
                        recipes[start + i, :] = weighted_recipe / total_purchase
                    else:
                        recipes[start + i, :] = self.B[start + i, :]
                
                if (start // batch_size + 1) % 10 == 0:
                    logger.info(f"   진행: {end}/{self.N} ({end/self.N*100:.1f}%)")
        
        return recipes
    
    def _simple_estimation(self) -> np.ndarray:
        """
        단순 평균: 공급자들의 레시피를 동일 가중치로 평균
        """
        recipes = np.zeros((self.N, 33), dtype=np.float32)
        
        for i in range(self.N):
            suppliers = self.H[i, :].nonzero()[1]  # 공급자 인덱스
            
            if len(suppliers) > 0:
                recipes[i, :] = self.B[suppliers, :].mean(axis=0)
            else:
                recipes[i, :] = self.B[i, :]
        
        return recipes
    
    def _bayesian_estimation(self, alpha: float = 0.3) -> np.ndarray:
        """
        베이지안 추정: B를 Prior로, H를 Likelihood로
        
        Recipe[i] = α * B[i] + (1-α) * WeightedRecipe[i]
        """
        weighted = self._weighted_estimation()
        recipes = alpha * self.B + (1 - alpha) * weighted
        return recipes
    
    def _postprocess(self, recipes: np.ndarray) -> np.ndarray:
        """
        후처리: 정규화, 이상치 제거
        """
        # 1. 음수 제거
        recipes = np.clip(recipes, 0, None)
        
        # 2. NaN/Inf 제거
        recipes = np.nan_to_num(recipes, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. 정규화 (행 합 = 1)
        row_sums = recipes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        recipes = recipes / row_sums
        
        # 4. 극단적 이상치 제거 (한 산업이 95% 이상인 경우)
        max_vals = recipes.max(axis=1)
        outlier_mask = max_vals > 0.95
        if outlier_mask.sum() > 0:
            logger.warning(f"⚠️ 극단적 레시피 {outlier_mask.sum()}개 발견 → 평활화")
            recipes[outlier_mask] = self.B[outlier_mask]
        
        return recipes
    
    def _print_statistics(self, recipes: np.ndarray):
        """통계 출력"""
        logger.info(f"   - NaN 개수: {np.isnan(recipes).sum()}")
        logger.info(f"   - 평균 사용 산업 수: {(recipes > 0.01).sum(axis=1).mean():.2f}")
        logger.info(f"   - 최대 집중도 평균: {recipes.max(axis=1).mean():.3f}")
        logger.info(f"   - 레시피 다양성 (엔트로피): {self._entropy(recipes).mean():.3f}")
    
    @staticmethod
    def _entropy(recipes: np.ndarray) -> np.ndarray:
        """Shannon entropy 계산"""
        eps = 1e-10
        p = recipes + eps
        return -(p * np.log(p)).sum(axis=1)
    
    def export_to_dataframe(self, recipes: np.ndarray) -> pd.DataFrame:
        """
        레시피를 DataFrame으로 변환
        
        Returns
        -------
        df : pd.DataFrame, shape (N, 34)
            첫 열: 사업자등록번호, 나머지 33열: 산업별 비율
        """
        sector_names = [f'sector_{i:02d}' for i in range(1, 34)]
        df = pd.DataFrame(recipes, columns=sector_names)
        df.insert(0, 'firm_id', self.firm_ids)
        return df
    
    def save_recipes(self, recipes: np.ndarray, output_path: str):
        """레시피를 pickle로 저장"""
        import pickle
        
        recipe_dict = {
            firm_id: recipes[i, :]
            for i, firm_id in enumerate(self.firm_ids)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(recipe_dict, f)
        
        logger.info(f"💾 레시피 저장: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("ZeroShotInventoryModule 테스트")
    print("=" * 60)
    
    # 더미 데이터 생성
    N = 1000
    firm_ids = [f'firm_{i:05d}' for i in range(N)]
    
    # 거래 네트워크 (Sparse)
    from scipy.sparse import random
    H = random(N, N, density=0.01, format='csr', random_state=42) * 1000
    
    # B 행렬
    B = np.random.rand(N, 33).astype(np.float32)
    B = B / B.sum(axis=1, keepdims=True)
    
    # 모듈 실행
    module = ZeroShotInventoryModule(H, B, firm_ids)
    recipes = module.estimate_recipes(method='weighted')
    
    print(f"\n✅ Recipes Shape: {recipes.shape}")
    print(f"✅ Sample Recipe (firm_00000):")
    print(recipes[0, :10])
    print(f"✅ Row Sum: {recipes[0, :].sum()}")
