"""
Static Graph Builder for Phase 2
=================================
Phase 1 레시피 + 재무/TIS 데이터를 결합하여 PyG Data 객체 생성
"""

import numpy as np
import pandas as pd
import pickle
import torch
from scipy.sparse import load_npz, csr_matrix
from pathlib import Path
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StaticGraphBuilder:
    """정적 그래프 데이터 빌더"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def build_static_data(
        self,
        use_simple_features: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, list]:
        """
        정적 그래프 데이터 구축
        
        Parameters
        ----------
        use_simple_features : bool
            True: 73차원 (간소화), False: 197차원 (전체)
        
        Returns
        -------
        X : torch.Tensor, shape (N, D)
            노드 피처 행렬
        edge_index : torch.Tensor, shape (2, E)
            엣지 인덱스 (PyG 형식)
        edge_attr : np.ndarray, shape (E,)
            엣지 속성 (거래 금액)
        firm_ids : list
            기업 ID 리스트
        """
        logger.info("=" * 70)
        logger.info("🏗️  정적 그래프 데이터 구축 시작")
        logger.info("=" * 70)
        
        # 1. 기업 인덱스 로드
        firm_ids = self._load_firm_ids()
        N = len(firm_ids)
        logger.info(f"✓ 기업 수: {N:,}")
        
        # 2. 피처 생성
        if use_simple_features:
            X = self._build_simple_features(firm_ids)
            logger.info(f"✓ 피처 차원: {X.shape[1]} (간소화 버전)")
        else:
            X = self._build_full_features(firm_ids)
            logger.info(f"✓ 피처 차원: {X.shape[1]} (전체 버전)")
        
        # 3. 엣지 로드 (H 행렬)
        edge_index, edge_attr = self._load_edges(firm_ids)
        logger.info(f"✓ 엣지 수: {edge_index.shape[1]:,}")
        
        # 4. 인덱스 정렬 보장
        X = self._reindex_features(X, firm_ids)
        
        logger.info("=" * 70)
        
        return X, edge_index, edge_attr, firm_ids
    
    def _load_firm_ids(self) -> list:
        """기업 ID 로드"""
        firm_to_idx_path = self.raw_dir / "firm_to_idx_model2.csv"
        df = pd.read_csv(firm_to_idx_path)
        firm_ids = df['사업자등록번호'].astype(str).tolist()
        return firm_ids
    
    def _build_simple_features(self, firm_ids: list) -> torch.Tensor:
        """
        간소화 피처 (73차원)
        = 재무4 + 지리2 + TIS1 + 산업33 + 레시피33
        """
        N = len(firm_ids)
        features = []
        
        # 1. 재무 피처 (4차원)
        logger.info("  📊 재무 피처 생성...")
        financial = self._load_financial_features(firm_ids)  # (N, 4)
        features.append(financial)
        
        # 2. 지리 피처 (2차원)
        logger.info("  🌍 지리 피처 생성...")
        geo = self._load_geo_features(firm_ids)  # (N, 2)
        features.append(geo)
        
        # 3. TIS 피처 (1차원)
        logger.info("  ⚠️  TIS 피처 생성...")
        tis = self._load_tis_features(firm_ids)  # (N, 1)
        features.append(tis)
        
        # 4. 산업 One-Hot (33차원)
        logger.info("  🏭 산업 피처 생성...")
        industry = self._load_industry_features(firm_ids)  # (N, 33)
        features.append(industry)
        
        # 5. 레시피 피처 (33차원)
        logger.info("  🧪 레시피 피처 로드...")
        recipe = self._load_recipe_features(firm_ids)  # (N, 33)
        features.append(recipe)
        
        # 결합
        X = np.concatenate(features, axis=1).astype(np.float32)
        
        # NaN 제거
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.from_numpy(X)
    
    def _build_full_features(self, firm_ids: list) -> torch.Tensor:
        """전체 피처 (197차원) - 산업 임베딩 추가"""
        # 간소화 버전과 동일하게 구현 (나중에 확장 가능)
        return self._build_simple_features(firm_ids)
    
    def _load_financial_features(self, firm_ids: list) -> np.ndarray:
        """재무 피처 로드 (4차원)"""
        N = len(firm_ids)
        financial = np.zeros((N, 4), dtype=np.float32)
        
        # 매출
        revenue_path = self.raw_dir / "tg_2024_filtered.csv"
        if revenue_path.exists():
            df = pd.read_csv(revenue_path)
            df['사업자등록번호'] = df.get('업체번호', df.get('사업자등록번호', '')).astype(str)
            revenue_map = dict(zip(df['사업자등록번호'], df['tg_2024_final']))
            
            for i, fid in enumerate(firm_ids):
                rev = revenue_map.get(fid, 0)
                financial[i, 0] = np.log1p(rev)  # log(매출 + 1)
        
        # 수출액
        export_path = self.raw_dir / "export_estimation_value_final.csv"
        if export_path.exists():
            df = pd.read_csv(export_path)
            df['사업자등록번호'] = df.get('업체번호', '').astype(str)
            export_map = dict(zip(df['사업자등록번호'], df['export_value']))
            
            for i, fid in enumerate(firm_ids):
                exp = export_map.get(fid, 0)
                financial[i, 1] = np.log1p(exp)  # log(수출 + 1)
        
        # 자산
        asset_path = self.raw_dir / "asset_final_2024_6차.csv"
        if asset_path.exists():
            df = pd.read_csv(asset_path)
            df['사업자등록번호'] = df.get('업체번호', '').astype(str)
            asset_map = dict(zip(df['사업자등록번호'], df['asset']))
            
            for i, fid in enumerate(firm_ids):
                ast = asset_map.get(fid, 0)
                financial[i, 2] = np.log1p(ast)  # log(자산 + 1)
        
        # 수출/매출 비율
        financial[:, 3] = np.where(
            financial[:, 0] > 0,
            financial[:, 1] / financial[:, 0],
            0
        )
        
        return financial
    
    def _load_geo_features(self, firm_ids: list) -> np.ndarray:
        """지리 피처 로드 (2차원)"""
        N = len(firm_ids)
        geo = np.zeros((N, 2), dtype=np.float32)
        
        firm_info_path = self.raw_dir / "vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv"
        if firm_info_path.exists():
            df = pd.read_csv(firm_info_path)
            df['사업자등록번호'] = df['사업자등록번호'].astype(str)
            
            lat_map = dict(zip(df['사업자등록번호'], df.get('위도', df.get('latitude', 0))))
            lon_map = dict(zip(df['사업자등록번호'], df.get('경도', df.get('longitude', 0))))
            
            for i, fid in enumerate(firm_ids):
                geo[i, 0] = lat_map.get(fid, 37.5)  # 기본값: 서울
                geo[i, 1] = lon_map.get(fid, 127.0)
        
        # 정규화 (대한민국 범위)
        geo[:, 0] = (geo[:, 0] - 33) / 10  # 위도 33~43
        geo[:, 1] = (geo[:, 1] - 124) / 8  # 경도 124~132
        
        return geo
    
    def _load_tis_features(self, firm_ids: list) -> np.ndarray:
        """TIS 피처 로드 (1차원)"""
        N = len(firm_ids)
        tis = np.zeros((N, 1), dtype=np.float32)
        
        tis_path = self.raw_dir / "shock_after_P_v2.csv"
        if tis_path.exists():
            df = pd.read_csv(tis_path)
            df['사업자등록번호'] = df.get('업체번호', '').astype(str)
            tis_map = dict(zip(df['사업자등록번호'], df['TIS']))
            
            for i, fid in enumerate(firm_ids):
                tis[i, 0] = tis_map.get(fid, 0)
        
        # 정규화 (0~1)
        tis = np.clip(tis, 0, 1)
        
        # 별도 저장 (Phase 3에서 사용)
        tis_path = self.processed_dir / "tis_score_normalized.npy"
        np.save(tis_path, tis)
        logger.info(f"    💾 TIS 저장: {tis_path}")
        
        return tis
    
    def _load_industry_features(self, firm_ids: list) -> np.ndarray:
        """산업 One-Hot 피처 (33차원)"""
        N = len(firm_ids)
        industry = np.zeros((N, 33), dtype=np.float32)
        
        firm_info_path = self.raw_dir / "vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv"
        if firm_info_path.exists():
            df = pd.read_csv(firm_info_path)
            df['사업자등록번호'] = df['사업자등록번호'].astype(str)
            
            sector_map = {}
            for _, row in df.iterrows():
                fid = row['사업자등록번호']
                sector_code = row.get('산업코드', -1)
                try:
                    sector_idx = int(sector_code) - 1  # 1-based → 0-based
                    if 0 <= sector_idx < 33:
                        sector_map[fid] = sector_idx
                except:
                    pass
            
            for i, fid in enumerate(firm_ids):
                if fid in sector_map:
                    industry[i, sector_map[fid]] = 1.0
        
        return industry
    
    def _load_recipe_features(self, firm_ids: list) -> np.ndarray:
        """레시피 피처 로드 (33차원)"""
        N = len(firm_ids)
        recipe = np.zeros((N, 33), dtype=np.float32)
        
        recipe_path = self.processed_dir / "disentangled_recipes.pkl"
        if recipe_path.exists():
            with open(recipe_path, 'rb') as f:
                recipe_dict = pickle.load(f)
            
            for i, fid in enumerate(firm_ids):
                if fid in recipe_dict:
                    recipe[i, :] = recipe_dict[fid]
                else:
                    # 레시피 없으면 균등 분포
                    recipe[i, :] = 1.0 / 33
            
            logger.info(f"    ✓ 레시피 로드: {len(recipe_dict)} 기업")
        else:
            logger.warning(f"    ⚠️  레시피 파일 없음: {recipe_path}")
            # 균등 분포로 초기화
            recipe[:, :] = 1.0 / 33
        
        # 캐싱
        cache_path = self.processed_dir / "recipe_features_cache.npy"
        np.save(cache_path, recipe)
        logger.info(f"    💾 레시피 캐시 저장: {cache_path}")
        
        return recipe
    
    def _load_edges(self, firm_ids: list) -> Tuple[torch.Tensor, np.ndarray]:
        """엣지 로드 (H 행렬)"""
        H_path = self.raw_dir / "H_csr_model2.npz"
        H = load_npz(H_path)
        
        # COO 형식으로 변환
        H_coo = H.tocoo()
        
        # PyG 형식으로 변환
        edge_index = np.vstack([H_coo.row, H_coo.col])
        edge_index = torch.from_numpy(edge_index).long()
        
        # 엣지 속성 (거래 금액)
        edge_attr = H_coo.data.astype(np.float32)
        
        # Log 스케일링
        edge_attr = np.log1p(edge_attr)
        
        return edge_index, edge_attr
    
    def _reindex_features(self, X: torch.Tensor, firm_ids: list) -> torch.Tensor:
        """인덱스 정렬 보장"""
        # 이미 firm_to_idx_model2.csv 순서대로 로드했으므로 그대로 반환
        return X
    
    def save_features(self, X: torch.Tensor):
        """피처 행렬 저장"""
        X_path = self.processed_dir / "X_feature_matrix.npy"
        np.save(X_path, X.numpy())
        logger.info(f"💾 피처 행렬 저장: {X_path}")


if __name__ == "__main__":
    builder = StaticGraphBuilder()
    X, edge_index, edge_attr, firm_ids = builder.build_static_data(use_simple_features=True)
    
    print(f"\n✅ 그래프 데이터 생성 완료")
    print(f"   - 노드 수: {X.shape[0]:,}")
    print(f"   - 피처 차원: {X.shape[1]}")
    print(f"   - 엣지 수: {edge_index.shape[1]:,}")
    print(f"   - 엣지 속성 범위: {edge_attr.min():.2f} ~ {edge_attr.max():.2f}")
    
    builder.save_features(X)
