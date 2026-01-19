"""
Phase 2: Static Graph Builder
==============================
정적 그래프 데이터 구축 및 피처 생성

피처 구조 (73차원 - 간소화 버전):
- 재무 (4): 매출, 수출, 자산, 수출/매출 비율
- 지리 (2): X좌표, Y좌표
- 리스크 (1): TIS 점수
- 산업 (33): IO 대분류 원-핫 인코딩
- 레시피 (33): Phase 1 출력
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class StaticGraphBuilder:
    """
    정적 그래프 데이터 빌더
    
    Phase 1 레시피 + 재무/TIS 데이터를 결합하여
    GraphSAGE 학습용 피처 행렬 생성
    """
    
    def __init__(self, data_dir: str, use_cache: bool = True):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.use_cache = use_cache
        
        # 캐시 디렉토리
        self.cache_dir = self.processed_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # IO 산업 코드 (33개)
        self.io_sectors = ['A', 'B', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 
                           'C10', 'C11', 'C12', 'C13', 'C14', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
        self.io_sector_to_idx = {sec: i for i, sec in enumerate(self.io_sectors)}
        
    def build_static_data(self, use_simple_features: bool = True):
        """
        정적 그래프 데이터 구축
        
        Parameters
        ----------
        use_simple_features : bool
            True: 73차원 (간소화), False: 197차원 (전체)
        
        Returns
        -------
        X : np.ndarray [N, D]
            노드 피처 행렬
        edge_index : torch.Tensor [2, E]
            엣지 인덱스
        edge_attr : torch.Tensor [E, edge_dim]
            엣지 속성 (거래액)
        firm_ids : list
            기업 ID 리스트
        """
        # 캐시 파일 경로
        cache_suffix = "simple" if use_simple_features else "full"
        cache_files = {
            'X': self.cache_dir / f"static_X_{cache_suffix}.npy",
            'edge_index': self.cache_dir / "static_edge_index.pt",
            'edge_attr': self.cache_dir / "static_edge_attr.pt",
            'firm_ids': self.cache_dir / "static_firm_ids.pkl"
        }
        
        # 캐시 확인
        if self.use_cache and all(f.exists() for f in cache_files.values()):
            logger.info("=" * 70)
            logger.info("📦 캐시된 정적 그래프 데이터 로드")
            logger.info("=" * 70)
            
            X = np.load(cache_files['X'])
            edge_index = torch.load(cache_files['edge_index'])
            edge_attr = torch.load(cache_files['edge_attr'])
            with open(cache_files['firm_ids'], 'rb') as f:
                firm_ids = pickle.load(f)
            
            logger.info(f"   ✓ 노드 수: {len(firm_ids):,}")
            logger.info(f"   ✓ 피처 차원: {X.shape[1]}")
            logger.info(f"   ✓ 엣지 수: {edge_index.shape[1]:,}")
            logger.info("=" * 70)
            
            return X, edge_index, edge_attr, firm_ids
        
        # 캐시가 없으면 새로 생성
        logger.info("=" * 70)
        logger.info("📊 정적 그래프 데이터 구축")
        logger.info("=" * 70)
        
        # 1. Firm ID 매핑 로드
        logger.info("1️⃣ 기업 인덱스 매핑 로드...")
        firm_to_idx_path = self.raw_dir / "firm_to_idx_model2.csv"
        df_idx = pd.read_csv(firm_to_idx_path)
        
        # 컬럼명 확인
        if 'Unnamed: 0' in df_idx.columns:
            firm_ids = df_idx['Unnamed: 0'].astype(str).tolist()
        elif '사업자등록번호' in df_idx.columns:
            firm_ids = df_idx['사업자등록번호'].astype(str).tolist()
        else:
            firm_ids = df_idx.iloc[:, 0].astype(str).tolist()
        
        N = len(firm_ids)
        logger.info(f"   ✓ 기업 수: {N:,}")
        
        # 2. H 행렬 (거래 네트워크) 로드
        logger.info("2️⃣ H 행렬 (거래 네트워크) 로드...")
        H_path = self.raw_dir / "H_csr_model2.npz"
        H_sparse = load_npz(H_path)
        
        # Sparse → PyTorch edge_index
        edge_index, edge_attr = self._sparse_to_edge_index(H_sparse)
        logger.info(f"   ✓ 엣지 수: {edge_index.shape[1]:,}")
        logger.info(f"   ✓ 평균 거래액: {edge_attr.mean():.2e}")
        
        # 3. 피처 생성
        logger.info("3️⃣ 노드 피처 생성...")
        if use_simple_features:
            X = self._build_simple_features(firm_ids, N)
            logger.info(f"   ✓ 피처 차원: {X.shape[1]} (간소화 버전)")
        else:
            X = self._build_full_features(firm_ids, N)
            logger.info(f"   ✓ 피처 차원: {X.shape[1]} (전체 버전)")
        
        # 캐시 저장
        if self.use_cache:
            logger.info("4️⃣ 캐시 저장...")
            np.save(cache_files['X'], X)
            torch.save(edge_index, cache_files['edge_index'])
            torch.save(edge_attr, cache_files['edge_attr'])
            with open(cache_files['firm_ids'], 'wb') as f:
                pickle.dump(firm_ids, f)
            logger.info(f"   ✓ 캐시 저장: {self.cache_dir}")
        
        logger.info("=" * 70)
        
        return X, edge_index, edge_attr, firm_ids
    
    def _sparse_to_edge_index(self, H_sparse):
        """
        Sparse 행렬을 PyTorch edge_index로 변환
        
        Returns
        -------
        edge_index : torch.Tensor [2, E]
        edge_attr : torch.Tensor [E, 1]
        """
        # COO format으로 변환
        H_coo = H_sparse.tocoo()
        
        edge_index = torch.tensor(
            np.vstack([H_coo.row, H_coo.col]),
            dtype=torch.long
        )
        
        edge_attr = torch.tensor(
            H_coo.data,
            dtype=torch.float
        ).unsqueeze(1)  # [E] → [E, 1]
        
        return edge_index, edge_attr
    
    def _build_simple_features(self, firm_ids: list, N: int) -> np.ndarray:
        """
        간소화 피처 생성 (73차원)
        
        구조:
        - 재무 (4): 매출, 수출, 자산, 수출/매출 비율
        - 지리 (2): X좌표, Y좌표
        - 리스크 (1): TIS 점수
        - 산업 (33): IO 대분류 원-핫
        - 레시피 (33): Phase 1 출력
        
        Returns
        -------
        X : np.ndarray [N, 73]
        """
        logger.info("   - 간소화 피처 생성 중 (73차원)...")
        
        # 1. 재무 피처 (4차원)
        financial_features = self._load_financial_features(firm_ids, N)
        
        # 2. 지리 피처 (2차원)
        geo_features = self._load_geo_features(firm_ids, N)
        
        # 3. TIS 피처 (1차원)
        tis_features = self._load_tis_features(firm_ids, N)
        
        # 4. 산업 피처 (33차원)
        industry_features = self._load_industry_features(firm_ids, N)
        
        # 5. 레시피 피처 (33차원)
        recipe_features = self._load_recipe_features(firm_ids, N)
        
        # 결합
        X = np.hstack([
            financial_features,  # 4
            geo_features,        # 2
            tis_features,        # 1
            industry_features,   # 33
            recipe_features      # 33
        ])
        
        logger.info(f"      ✓ 재무: {financial_features.shape[1]}차원")
        logger.info(f"      ✓ 지리: {geo_features.shape[1]}차원")
        logger.info(f"      ✓ TIS: {tis_features.shape[1]}차원")
        logger.info(f"      ✓ 산업: {industry_features.shape[1]}차원")
        logger.info(f"      ✓ 레시피: {recipe_features.shape[1]}차원")
        logger.info(f"      ✓ 총합: {X.shape[1]}차원")
        
        return X
    
    def _build_full_features(self, firm_ids: list, N: int) -> np.ndarray:
        """
        전체 피처 생성 (197차원)
        
        TODO: 필요시 구현
        """
        logger.warning("   ⚠️  전체 피처는 아직 구현되지 않았습니다. 간소화 버전 사용")
        return self._build_simple_features(firm_ids, N)
    
    def _load_financial_features(self, firm_ids: list, N: int) -> np.ndarray:
        """
        재무 피처 로드 (4차원)
        - 매출 (tg_2024_final)
        - 수출 (export_value)
        - 자산 (asset)
        - 수출/매출 비율
        """
        features = np.zeros((N, 4))
        
        # 매출 데이터
        revenue_path = self.raw_dir / "final_tg_2024_estimation.csv"
        if revenue_path.exists():
            df_rev = pd.read_csv(revenue_path, dtype=str)
            df_rev['업체번호'] = df_rev['업체번호'].astype(str)
            
            # 매출 컬럼 찾기
            rev_col = None
            for col in ['tg_2024_final', 'tg_2024', 'revenue']:
                if col in df_rev.columns:
                    rev_col = col
                    break
            
            if rev_col:
                rev_dict = dict(zip(df_rev['업체번호'], pd.to_numeric(df_rev[rev_col], errors='coerce')))
                for i, fid in enumerate(firm_ids):
                    if fid in rev_dict:
                        features[i, 0] = rev_dict[fid]
        
        # 수출 데이터
        export_path = self.raw_dir / "export_estimation_value_final.csv"
        if export_path.exists():
            df_exp = pd.read_csv(export_path, dtype=str)
            df_exp['업체번호'] = df_exp['업체번호'].astype(str)
            
            exp_col = 'export_value' if 'export_value' in df_exp.columns else df_exp.columns[1]
            exp_dict = dict(zip(df_exp['업체번호'], pd.to_numeric(df_exp[exp_col], errors='coerce')))
            for i, fid in enumerate(firm_ids):
                if fid in exp_dict:
                    features[i, 1] = exp_dict[fid]
        
        # 자산 데이터
        asset_path = self.raw_dir / "asset_final_2024_6차.csv"
        if asset_path.exists():
            df_asset = pd.read_csv(asset_path, dtype=str)
            df_asset['업체번호'] = df_asset['업체번호'].astype(str)
            
            asset_col = None
            for col in ['자산추정_2024', 'asset', '자산']:
                if col in df_asset.columns:
                    asset_col = col
                    break
            
            if asset_col:
                asset_dict = dict(zip(df_asset['업체번호'], pd.to_numeric(df_asset[asset_col], errors='coerce')))
                for i, fid in enumerate(firm_ids):
                    if fid in asset_dict:
                        features[i, 2] = asset_dict[fid]
        
        # 수출/매출 비율
        with np.errstate(divide='ignore', invalid='ignore'):
            features[:, 3] = np.where(features[:, 0] > 0, features[:, 1] / features[:, 0], 0)
        
        # 정규화 (log1p)
        features[:, :3] = np.log1p(np.abs(features[:, :3]))
        
        # NaN 처리
        features = np.nan_to_num(features, 0.0)
        
        return features
    
    def _load_geo_features(self, firm_ids: list, N: int) -> np.ndarray:
        """
        지리 피처 로드 (2차원)
        - X축POI좌표값
        - Y축POI좌표값
        """
        features = np.zeros((N, 2))
        
        firm_info_path = self.raw_dir / "vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv"
        if firm_info_path.exists():
            df_firm = pd.read_csv(firm_info_path, dtype=str)
            
            # 사업자번호 컬럼 찾기
            biz_col = None
            for col in df_firm.columns:
                if '사업자' in col and '번호' in col:
                    biz_col = col
                    break
            
            if biz_col:
                df_firm[biz_col] = df_firm[biz_col].astype(str)
                
                # 좌표 컬럼
                x_col = 'X축POI좌표값' if 'X축POI좌표값' in df_firm.columns else None
                y_col = 'Y축POI좌표값' if 'Y축POI좌표값' in df_firm.columns else None
                
                if x_col and y_col:
                    coord_dict = dict(zip(
                        df_firm[biz_col],
                        zip(
                            pd.to_numeric(df_firm[x_col], errors='coerce'),
                            pd.to_numeric(df_firm[y_col], errors='coerce')
                        )
                    ))
                    
                    for i, fid in enumerate(firm_ids):
                        if fid in coord_dict:
                            x, y = coord_dict[fid]
                            if pd.notna(x) and pd.notna(y):
                                features[i, 0] = x
                                features[i, 1] = y
        
        # NaN 처리
        features = np.nan_to_num(features, 0.0)
        
        return features
    
    def _load_tis_features(self, firm_ids: list, N: int) -> np.ndarray:
        """
        TIS 리스크 피처 로드 (1차원)
        """
        features = np.zeros((N, 1))
        
        tis_path = self.raw_dir / "shock_after_P_v2.csv"
        if tis_path.exists():
            df_tis = pd.read_csv(tis_path, dtype=str)
            df_tis['업체번호'] = df_tis['업체번호'].astype(str) if '업체번호' in df_tis.columns else df_tis.iloc[:, 0].astype(str)
            
            # TIS 컬럼 찾기
            tis_col = None
            for col in ['tis_score', 'shock_score', 'TIS']:
                if col in df_tis.columns:
                    tis_col = col
                    break
            
            if tis_col is None:
                tis_col = df_tis.columns[1] if len(df_tis.columns) > 1 else df_tis.columns[0]
            
            tis_dict = dict(zip(df_tis.iloc[:, 0], pd.to_numeric(df_tis[tis_col], errors='coerce')))
            for i, fid in enumerate(firm_ids):
                if fid in tis_dict and pd.notna(tis_dict[fid]):
                    features[i, 0] = tis_dict[fid]
        
        # 정규화 (0-1)
        if features.max() > 0:
            features = features / features.max()
        
        # NaN 처리
        features = np.nan_to_num(features, 0.0)
        
        # TIS 저장
        tis_save_path = self.processed_dir / "tis_score_normalized.npy"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.save(tis_save_path, features)
        
        return features
    
    def _load_industry_features(self, firm_ids: list, N: int) -> np.ndarray:
        """
        산업 분류 피처 로드 (33차원 원-핫)
        - IO상품_단일_대분류_코드 기반
        """
        features = np.zeros((N, 33))
        
        firm_info_path = self.raw_dir / "vat_20-24_company_list_w_companyinfo_nocutoff_v3_hyundaisteel_hj.csv"
        if firm_info_path.exists():
            df_firm = pd.read_csv(firm_info_path, dtype=str)
            
            # 사업자번호 컬럼
            biz_col = None
            for col in df_firm.columns:
                if '사업자' in col and '번호' in col:
                    biz_col = col
                    break
            
            if biz_col:
                df_firm[biz_col] = df_firm[biz_col].astype(str)
                
                # IO 상품 코드 컬럼
                io_col = 'IO상품_단일_대분류_코드'
                if io_col in df_firm.columns:
                    io_dict = dict(zip(df_firm[biz_col], df_firm[io_col].astype(str).str.strip()))
                    
                    for i, fid in enumerate(firm_ids):
                        if fid in io_dict:
                            io_code = io_dict[fid]
                            if io_code in self.io_sector_to_idx:
                                sector_idx = self.io_sector_to_idx[io_code]
                                features[i, sector_idx] = 1.0
        
        return features
    
    def _load_recipe_features(self, firm_ids: list, N: int) -> np.ndarray:
        """
        Phase 1 레시피 로드 (33차원)
        """
        features = np.zeros((N, 33))
        
        recipe_path = self.processed_dir / "disentangled_recipes.pkl"
        if recipe_path.exists():
            with open(recipe_path, 'rb') as f:
                recipe_dict = pickle.load(f)
            
            for i, fid in enumerate(firm_ids):
                if fid in recipe_dict:
                    recipe = recipe_dict[fid]
                    if isinstance(recipe, np.ndarray) and len(recipe) == 33:
                        features[i] = recipe
        else:
            logger.warning(f"   ⚠️  Phase 1 레시피 파일 없음: {recipe_path}")
        
        return features
    
    def save_features(self, X: np.ndarray):
        """피처 행렬 저장"""
        save_path = self.processed_dir / "X_feature_matrix.npy"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_path, X)
        logger.info(f"💾 피처 행렬 저장: {save_path}")
