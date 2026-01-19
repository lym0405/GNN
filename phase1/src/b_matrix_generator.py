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
        # IO 산업 코드를 문자열로 통일 (매칭 안정성)
        df_io.index = df_io.index.astype(str).str.strip()
        self.io_sectors = df_io.index.tolist()
        self.sector_to_idx = {sec: i for i, sec in enumerate(self.io_sectors)}
        self.standard_recipes = df_io.values.T
        
        print(f"   - IO 테이블: {len(self.io_sectors)}개 산업 (예: {self.io_sectors[:3]})")  

        # 3. 기업 정보(nocutoff) 및 매출 데이터 통합
        print("   - Mapping Industry Sectors & Shares to Business IDs...")
        df_firm = pd.read_csv(firm_info_path, dtype=str)
        
        # 사업자번호 컬럼 찾기
        col_biz = None
        for c in df_firm.columns:
            if '사업자' in c and '번호' in c:
                col_biz = c
                break
        if col_biz is None:
            col_biz = df_firm.columns[0]  # 폴백: 첫 번째 컬럼
            print(f"   ⚠️  사업자번호 컬럼을 찾을 수 없어 '{col_biz}' 사용")
        
        # 업체번호 컬럼 찾기
        col_id = None
        for c in df_firm.columns:
            if '업체번호' in c:
                col_id = c
                break
        if col_id is None:
            col_id = col_biz  # 폴백: 사업자번호 사용
            print(f"   ⚠️  업체번호 컬럼을 찾을 수 없어 '{col_id}' 사용")
        
        # IO 테이블(33개)과 매칭: IO상품_단일_대분류_코드 사용
        col_sec = 'IO상품_단일_대분류_코드'
        # 1순위: IO상품_단일_대분류_코드      
        # 2순위: 산업코드 (더미 데이터용)
        if col_sec is None:
            for c in df_firm.columns:
                if 'IO상품' in c and '단일' in c and '대분류' in c and '코드' in c:
                    col_sec = c
                    print(f"   ⚠️  IO상품 컬럼을 찾을 수 없어 '{col_sec}' 사용 (더미 데이터?)")
                    break

        # 4순위: 없으면 에러
        if col_sec is None:
                raise ValueError(f"IO 산업 매핑 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df_firm.columns[:10])}")
        
        df_firm['clean_biz'] = self._normalize(df_firm[col_biz])
        df_firm['clean_id'] = self._normalize(df_firm[col_id])

        df_sales = pd.read_csv(sales_path, dtype=str)
        
        # 매출 컬럼 찾기
        col_sales = None
        for c in df_sales.columns:
            if 'tg_2024_final' in c or 'sales' in c.lower() or '매출' in c:
                col_sales = c
                break
        if col_sales is None:
            # 폴백: 두 번째 컬럼 (첫 번째는 보통 ID)
            col_sales = df_sales.columns[1] if len(df_sales.columns) > 1 else df_sales.columns[0]
            print(f"   ⚠️  매출 컬럼을 찾을 수 없어 '{col_sales}' 사용")
        
        # 업체번호 컬럼 찾기
        col_sales_id = None
        for c in df_sales.columns:
            if '업체번호' in c or 'id' in c.lower():
                col_sales_id = c
                break
        if col_sales_id is None:
            col_sales_id = df_sales.columns[0]
            print(f"   ⚠️  매출 데이터의 ID 컬럼을 찾을 수 없어 '{col_sales_id}' 사용")
        
        df_sales['clean_id'] = self._normalize(df_sales[col_sales_id])
        df_sales['amt'] = pd.to_numeric(df_sales[col_sales], errors='coerce').fillna(0)
        
        # 사업자번호 기준으로 산업분류와 매출(Share) 연결
        df_merged = pd.merge(df_firm, df_sales[['clean_id', 'amt']], on='clean_id', how='inner')
        
        # 산업 코드를 문자열로 통일 (IO 테이블과 매칭)
        df_merged[col_sec] = df_merged[col_sec].astype(str).str.strip()
        
        # 매출 기반 Share 계산
        sector_sums = df_merged.groupby(col_sec)['amt'].transform('sum')
        df_merged['share'] = df_merged['amt'] / sector_sums
        
        self.biz_sector_map = dict(zip(df_merged['clean_biz'], df_merged[col_sec]))
        self.biz_share_map = dict(zip(df_merged['clean_biz'], df_merged['share'].fillna(0)))
        
        # 매핑 성공률 출력
        total_firms = len(self.biz_sector_map)
        matched_firms = sum(1 for sec in self.biz_sector_map.values() if sec in self.sector_to_idx)
        print(f"   - 기업-산업 매핑: {total_firms:,}개 기업")
        print(f"   - IO 테이블 매칭: {matched_firms:,}개 ({matched_firms/total_firms*100:.1f}%)")
        if matched_firms < total_firms * 0.5:
            print(f"   ⚠️  매칭률이 낮습니다! IO 코드 확인 필요")
            print(f"      샘플 기업 산업 코드: {list(self.biz_sector_map.values())[:5]}")
            print(f"      IO 테이블 산업 코드: {self.io_sectors[:5]}")

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