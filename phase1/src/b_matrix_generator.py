import pandas as pd
import numpy as np
import scipy.sparse as sp
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
        self.io_sectors = df_io.index.tolist()
        self.sector_to_idx = {sec: i for i, sec in enumerate(self.io_sectors)}
        self.standard_recipes = df_io.values.T 

        # 3. 기업 정보(nocutoff) 및 매출 데이터 통합
        print("   - Mapping Industry Sectors & Shares to Business IDs...")
        df_firm = pd.read_csv(firm_info_path, dtype=str)
        col_biz = next(c for c in df_firm.columns if '사업자' in c and '번호' in c)
        col_id = next(c for c in df_firm.columns if '업체번호' in c)
        col_sec = next(c for c in df_firm.columns if 'IO' in c and '대분류' in c)
        
        df_firm['clean_biz'] = self._normalize(df_firm[col_biz])
        df_firm['clean_id'] = self._normalize(df_firm[col_id])

        df_sales = pd.read_csv(sales_path, dtype=str)
        col_sales = next(c for c in df_sales.columns if 'tg_2024' in c or 'sales' in c)
        df_sales['clean_id'] = self._normalize(df_sales['업체번호'])
        df_sales['amt'] = pd.to_numeric(df_sales[col_sales], errors='coerce').fillna(0)
        
        # 사업자번호 기준으로 산업분류와 매출(Share) 연결
        df_merged = pd.merge(df_firm, df_sales[['clean_id', 'amt']], on='clean_id', how='inner')
        sector_sums = df_merged.groupby(col_sec)['amt'].transform('sum')
        df_merged['share'] = df_merged['amt'] / sector_sums
        
        self.biz_sector_map = dict(zip(df_merged['clean_biz'], df_merged[col_sec]))
        self.biz_share_map = dict(zip(df_merged['clean_biz'], df_merged['share'].fillna(0)))

        # 4. H 행렬 인덱스별 산업 코드 매핑 배열 생성 (IndexError 방지)
        self.col_idx_to_sec_idx = np.full(N_TARGET, -1, dtype=int)
        for i, biz in enumerate(self.sorted_biz_ids):
            sec = self.biz_sector_map.get(biz)
            if sec in self.sector_to_idx:
                self.col_idx_to_sec_idx[i] = self.sector_to_idx[sec]

    def _normalize(self, series):
        """숫자 외 문자 제거 및 앞자리 0 제거로 매칭률 극대화"""
        return series.astype(str).str.replace(r'[^0-9]', '', regex=True).str.lstrip('0')

    def get_vector(self, query_id):
        """사업자번호를 입력받아 최종 생산함수 벡터(33차원) 반환"""
        clean_query = "".join(filter(str.isdigit, str(query_id))).lstrip('0')
        
        if clean_query not in self.biz_to_idx:
            return None 

        idx = self.biz_to_idx[clean_query]
        share = self.biz_share_map.get(clean_query, 0)
        
        # R Vector: 산업 표준 레시피
        sec_code = self.biz_sector_map.get(clean_query)
        r_vec = np.zeros(33)
        if sec_code in self.sector_to_idx:
            r_vec = self.standard_recipes[self.sector_to_idx[sec_code]] * share
            
        # H Vector: 기업 간 거래 기반 데이터
        h_vec = np.zeros(33)
        start, end = self.H_sparse.indptr[idx], self.H_sparse.indptr[idx+1]
        
        if start < end:
            col_indices = self.H_sparse.indices[start:end]
            data_values = self.H_sparse.data[start:end]
            sec_indices = self.col_idx_to_sec_idx[col_indices]
            valid_mask = (sec_indices != -1)
            
            if np.any(valid_mask):
                np.add.at(h_vec, sec_indices[valid_mask], data_values[valid_mask])
            h_vec = h_vec * share

        return (self.alpha * h_vec) + ((1 - self.alpha) * r_vec)