"""
Recipe Validation Tool
======================
생성된 레시피의 품질 검증 및 통계 분석
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeValidator:
    """레시피 품질 검증 도구"""
    
    def __init__(self, recipe_dict: dict):
        """
        Parameters
        ----------
        recipe_dict : dict
            {firm_id: np.ndarray(33,)} 형태의 레시피 딕셔너리
        """
        self.recipes = recipe_dict
        self.firm_ids = list(recipe_dict.keys())
        self.N = len(self.firm_ids)
        
        # 행렬 형태로 변환
        self.recipe_matrix = np.array([recipe_dict[fid] for fid in self.firm_ids])
        
        logger.info(f"✅ RecipeValidator 초기화: {self.N}개 기업")
    
    def run_all_checks(self):
        """모든 검증 수행"""
        print("\n" + "=" * 70)
        print("📊 RECIPE VALIDATION REPORT")
        print("=" * 70)
        
        self.check_basic_properties()
        self.check_statistical_properties()
        self.check_diversity()
        self.check_outliers()
        self.check_sector_distribution()
        
        print("=" * 70)
    
    def check_basic_properties(self):
        """기본 속성 검증"""
        print("\n[1] 기본 속성 검증")
        print("-" * 70)
        
        # NaN 체크
        nan_count = np.isnan(self.recipe_matrix).sum()
        print(f"✓ NaN 개수: {nan_count} ({'PASS' if nan_count == 0 else 'FAIL'})")
        
        # Inf 체크
        inf_count = np.isinf(self.recipe_matrix).sum()
        print(f"✓ Inf 개수: {inf_count} ({'PASS' if inf_count == 0 else 'FAIL'})")
        
        # 음수 체크
        neg_count = (self.recipe_matrix < 0).sum()
        print(f"✓ 음수 개수: {neg_count} ({'PASS' if neg_count == 0 else 'FAIL'})")
        
        # 행 합 체크 (1에 가까운지)
        row_sums = self.recipe_matrix.sum(axis=1)
        sum_error = np.abs(row_sums - 1.0)
        max_error = sum_error.max()
        print(f"✓ 행 합 오차 (최대): {max_error:.6f} ({'PASS' if max_error < 0.01 else 'FAIL'})")
        
        # 0벡터 체크
        zero_rows = (self.recipe_matrix.sum(axis=1) == 0).sum()
        print(f"✓ Zero 벡터 개수: {zero_rows} ({'PASS' if zero_rows == 0 else 'FAIL'})")
    
    def check_statistical_properties(self):
        """통계적 속성"""
        print("\n[2] 통계적 속성")
        print("-" * 70)
        
        # 평균/표준편차
        mean = self.recipe_matrix.mean()
        std = self.recipe_matrix.std()
        print(f"✓ 전체 평균: {mean:.6f} (기대값: {1/33:.6f})")
        print(f"✓ 전체 표준편차: {std:.6f}")
        
        # 산업별 평균
        sector_means = self.recipe_matrix.mean(axis=0)
        print(f"✓ 산업별 평균 (Min~Max): {sector_means.min():.4f} ~ {sector_means.max():.4f}")
        
        # 가장 많이 사용되는 산업 Top 5
        top5_sectors = np.argsort(sector_means)[-5:][::-1]
        print(f"✓ 가장 많이 사용되는 산업 Top 5:")
        for rank, sector in enumerate(top5_sectors, 1):
            print(f"    {rank}. Sector {sector:02d}: {sector_means[sector]:.4f}")
    
    def check_diversity(self):
        """레시피 다양성 분석"""
        print("\n[3] 레시피 다양성")
        print("-" * 70)
        
        # 평균 사용 산업 수 (0.01 이상)
        active_sectors = (self.recipe_matrix > 0.01).sum(axis=1)
        print(f"✓ 평균 사용 산업 수: {active_sectors.mean():.2f}")
        print(f"✓ 중앙값 사용 산업 수: {np.median(active_sectors):.0f}")
        print(f"✓ 최소/최대: {active_sectors.min():.0f} ~ {active_sectors.max():.0f}")
        
        # Shannon Entropy (높을수록 다양)
        eps = 1e-10
        p = self.recipe_matrix + eps
        entropy = -(p * np.log(p)).sum(axis=1)
        print(f"✓ 평균 엔트로피: {entropy.mean():.3f} (Max: {np.log(33):.3f})")
        
        # Gini 계수 (낮을수록 균등)
        gini_scores = self._calculate_gini()
        print(f"✓ 평균 Gini 계수: {gini_scores.mean():.3f} (0=완전균등, 1=완전불균등)")
    
    def check_outliers(self):
        """이상치 탐지"""
        print("\n[4] 이상치 분석")
        print("-" * 70)
        
        # 극단적 집중도 (한 산업이 90% 이상)
        max_vals = self.recipe_matrix.max(axis=1)
        extreme_count = (max_vals > 0.9).sum()
        print(f"✓ 극단 집중 기업 (>90%): {extreme_count} ({extreme_count/self.N*100:.2f}%)")
        
        # 매우 균등 분포 (모든 산업이 3% 내외)
        uniform_mask = (self.recipe_matrix > 0.025) & (self.recipe_matrix < 0.035)
        very_uniform = uniform_mask.all(axis=1).sum()
        print(f"✓ 완전 균등 기업: {very_uniform} ({very_uniform/self.N*100:.2f}%)")
        
        # 가장 극단적인 케이스 출력
        extreme_idx = np.argmax(max_vals)
        extreme_firm = self.firm_ids[extreme_idx]
        extreme_sector = np.argmax(self.recipe_matrix[extreme_idx])
        print(f"✓ 최대 집중 케이스: {extreme_firm}")
        print(f"    → Sector {extreme_sector:02d}: {max_vals[extreme_idx]:.2%}")
    
    def check_sector_distribution(self):
        """산업별 분포 분석"""
        print("\n[5] 산업별 분포")
        print("-" * 70)
        
        sector_means = self.recipe_matrix.mean(axis=0)
        sector_stds = self.recipe_matrix.std(axis=0)
        
        print(f"{'Sector':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 50)
        
        for i in range(33):
            mean = sector_means[i]
            std = sector_stds[i]
            min_val = self.recipe_matrix[:, i].min()
            max_val = self.recipe_matrix[:, i].max()
            print(f"{i:02d}         {mean:.4f}     {std:.4f}     {min_val:.4f}     {max_val:.4f}")
    
    def _calculate_gini(self) -> np.ndarray:
        """Gini 계수 계산 (각 기업별)"""
        sorted_recipes = np.sort(self.recipe_matrix, axis=1)
        n = sorted_recipes.shape[1]
        index = np.arange(1, n + 1)
        
        gini = (2 * (sorted_recipes * index).sum(axis=1)) / (n * sorted_recipes.sum(axis=1)) - (n + 1) / n
        return gini
    
    def export_report(self, output_path: str):
        """검증 결과를 CSV로 저장"""
        report_data = []
        
        for i, firm_id in enumerate(self.firm_ids):
            recipe = self.recipe_matrix[i]
            active_sectors = (recipe > 0.01).sum()
            max_sector = np.argmax(recipe)
            max_val = recipe[max_sector]
            
            report_data.append({
                'firm_id': firm_id,
                'active_sectors': active_sectors,
                'max_sector': max_sector,
                'max_concentration': max_val,
                'entropy': -(recipe * np.log(recipe + 1e-10)).sum(),
            })
        
        df = pd.DataFrame(report_data)
        df.to_csv(output_path, index=False)
        logger.info(f"💾 검증 리포트 저장: {output_path}")


def load_and_validate(recipe_path: str, report_path: str = None):
    """
    레시피 파일을 로드하고 검증 수행
    
    Parameters
    ----------
    recipe_path : str
        disentangled_recipes.pkl 경로
    report_path : str, optional
        리포트 저장 경로
    """
    with open(recipe_path, 'rb') as f:
        recipe_dict = pickle.load(f)
    
    validator = RecipeValidator(recipe_dict)
    validator.run_all_checks()
    
    if report_path:
        validator.export_report(report_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        recipe_path = sys.argv[1]
        report_path = sys.argv[2] if len(sys.argv) > 2 else None
        load_and_validate(recipe_path, report_path)
    else:
        print("사용법: python check_recipe.py <recipe_path> [report_path]")
        print("예: python check_recipe.py data/processed/disentangled_recipes.pkl validation_report.csv")
