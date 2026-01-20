import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy import sparse
import os

class BMatrixGenerator:
    def __init__(self, io_path, h_path, firm_info_path, sales_path, alpha=0.5):
        print(f"[B-Gen] Initializing Generator (Alpha={alpha})...")
        self.alpha = alpha
        
        # 1. H Matrix 및 사업자번호 기반 인덱스 맵 로드
        print(f"   - Loading H Matrix and Business ID Index Map...")
        self.H_sparse = sp.load_npz(h_path)
        N_TARGET = self.H_sparse.shape[0] # 438,946
        
        idx_map_path = os.path.join(os.path.dirname(h_path), "firm_to_idx_model2.csv")
        df_idx = pd.read_csv(idx_map_path)
        # 첫 번째 컬럼(사업자번호)을 기준으로 인덱스 순서 정렬
        df_idx['clean_biz'] = self._normalize(df_idx.iloc[:, 0])
        df_idx['idx_val'] = pd.to_numeric(df_idx.iloc[:, 1])
        
        self.sorted_biz_ids = df_idx.sort_values('idx_val')['clean_biz'].tolist()
        self.biz_to_idx = {biz: i for i, biz in enumerate(self.sorted_biz_ids)}

        # 2. IO 테이블 및 표준 레시피 설정
        df_io = pd.read_csv(io_path, index_col=0)
        df_io.columns = [str(c).strip().replace('*', '') for c in df_io.columns]
        df_io.index = df_io.index.astype(str).str.strip()
        self.io_sectors = df_io.index.tolist()
        self.sector_to_idx = {sec: i for i, sec in enumerate(self.io_sectors)}
        self.standard_recipes = df_io.values.T
        
        print(f"   - IO 테이블: {len(self.io_sectors)}개 산업")

        # 3. 기업 정보(nocutoff) 및 매출 데이터 통합
        print("   - Mapping Industry Sectors & Shares to Business IDs...")
        df_firm = pd.read_csv(firm_info_path, dtype=str)
        
        # 사업자번호 컬럼 찾기
        col_biz = next((c for c in df_firm.columns if '사업자' in c and '번호' in c), df_firm.columns[0])
        col_id = next((c for c in df_firm.columns if '업체번호' in c), col_biz)
        
        # IO 상품 코드 찾기
        col_sec = 'IO상품_단일_대분류_코드'
        if col_sec not in df_firm.columns:
             # 폴백 검색
             for c in df_firm.columns:
                if 'IO상품' in c and '코드' in c:
                    col_sec = c
                    break
        
        df_firm['clean_biz'] = self._normalize(df_firm[col_biz])
        df_firm['clean_id'] = self._normalize(df_firm[col_id])

        df_sales = pd.read_csv(sales_path, dtype=str)
        
        # 매출 컬럼 찾기
        col_sales = next((c for c in df_sales.columns if 'tg_2024_final' in c or 'sales' in c.lower()), df_sales.columns[1])
        col_sales_id = next((c for c in df_sales.columns if '업체번호' in c or 'id' in c.lower()), df_sales.columns[0])
        
        df_sales['clean_id'] = self._normalize(df_sales[col_sales_id])
        df_sales['amt'] = pd.to_numeric(df_sales[col_sales], errors='coerce').fillna(0)
        
        # 병합
        df_merged = pd.merge(df_firm, df_sales[['clean_id', 'amt']], on='clean_id', how='inner')
        df_merged[col_sec] = df_merged[col_sec].astype(str).str.strip()
        
        # Share 계산
        sector_sums = df_merged.groupby(col_sec)['amt'].transform('sum')
        df_merged['share'] = df_merged['amt'] / sector_sums
        
        self.biz_sector_map = dict(zip(df_merged['clean_biz'], df_merged[col_sec]))
        self.biz_share_map = dict(zip(df_merged['clean_biz'], df_merged['share'].fillna(0)))

        # 4. H 행렬 인덱스별 산업 코드 매핑
        self.col_idx_to_sec_idx = np.full(N_TARGET, -1, dtype=int)
        for i, biz in enumerate(self.sorted_biz_ids):
            sec = self.biz_sector_map.get(biz)
            if sec in self.sector_to_idx:
                self.col_idx_to_sec_idx[i] = self.sector_to_idx[sec]

    def _normalize(self, series):
        return series.astype(str).str.replace(r'[^0-9]', '', regex=True).str.lstrip('0')

    def generate_all_vectors(self):
        """
        [최적화] 벡터화된 B 행렬 생성 (For Loop 제거)
        Returns: (N, 33) Matrix
        """
        print("   🚀 Generating B Matrix (Vectorized)...")
        N = len(self.sorted_biz_ids)
        
        # 1. R_vec 계산 (Standard Recipe * Share)
        valid_indices = self.col_idx_to_sec_idx != -1
        
        # (N, 33) 초기화
        R_full = np.zeros((N, 33), dtype=np.float32)
        
        # 유효한 기업들의 표준 레시피 매핑
        if np.any(valid_indices):
            # (N_valid, 33)
            mapped_recipes = self.standard_recipes[self.col_idx_to_sec_idx[valid_indices]]
            
            # Share 값 가져오기
            # biz_share_map은 dict이므로 순서대로 배열 생성
            shares = np.array([self.biz_share_map.get(self.sorted_biz_ids[i], 0) for i in range(N)])
            
            R_full[valid_indices] = mapped_recipes
            R_full = R_full * shares[:, None]
            
        # 2. H_vec 계산 (Transaction Based)
        # H_vec = H @ Sector_Matrix
        
        # Sector Matrix (S) 생성: (N, 33) Sparse
        # 기업 i가 산업 j에 속하면 1 (또는 share?)
        # 원본 로직에 따르면 H_vec 계산 시에는 단순히 산업군으로 집계 후 마지막에 본인의 share를 곱함
        
        S_rows = np.where(valid_indices)[0]
        S_cols = self.col_idx_to_sec_idx[S_rows]
        S_data = np.ones(len(S_rows), dtype=np.float32)
        
        S_mat = sp.csr_matrix((S_data, (S_rows, S_cols)), shape=(N, 33))
        
        # 희소 행렬 곱셈 (매우 빠름)
        # H(거래) @ S(산업) = 각 기업이 각 산업군으로부터 구매한 총액
        H_aggregated = self.H_sparse.dot(S_mat)
        
        if sp.issparse(H_aggregated):
            H_aggregated = H_aggregated.toarray()
            
        # 본인의 Share 적용
        H_full = H_aggregated * shares[:, None]
        
        # 3. 결합
        B_final = (self.alpha * H_full) + ((1 - self.alpha) * R_full)
        
        return B_final.astype(np.float32)