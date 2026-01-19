"""
Phase 5: KSIC 매칭 및 기업 선정
================================
KSIC 코드 기반으로 충격 시나리오 대상 기업 선정
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class KSICMatcher:
    """
    KSIC 코드 기반 기업 매칭
    
    Parameters
    ----------
    firm_info_path : str
        기업 정보 파일 경로 (업체번호, KSIC 코드 포함)
    firm_to_idx_path : str
        firm_to_idx 매핑 파일 경로
    """
    
    def __init__(
        self,
        firm_info_path: str,
        firm_to_idx_path: str
    ):
        self.firm_info_path = Path(firm_info_path)
        self.firm_to_idx_path = Path(firm_to_idx_path)
        
        # 데이터 로드
        self.firm_info_df = self._load_firm_info()
        self.firm_to_idx = self._load_firm_to_idx()
        
        logger.info(f"✅ KSICMatcher 초기화")
        logger.info(f"   - 기업 정보: {len(self.firm_info_df):,}개")
        logger.info(f"   - Firm-to-Index: {len(self.firm_to_idx):,}개")
    
    def _load_firm_info(self) -> pd.DataFrame:
        """기업 정보 로드"""
        if not self.firm_info_path.exists():
            logger.warning(f"⚠️  기업 정보 파일 없음: {self.firm_info_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.firm_info_path)
        logger.info(f"   ✓ 기업 정보 로드: {len(df):,}개 기업")
        
        # 컬럼명 확인
        if 'KSIC' in df.columns or 'ksic' in df.columns:
            ksic_col = 'KSIC' if 'KSIC' in df.columns else 'ksic'
            logger.info(f"   ✓ KSIC 컬럼 확인: {ksic_col}")
        else:
            logger.warning("   ⚠️  KSIC 컬럼 없음")
        
        return df
    
    def _load_firm_to_idx(self) -> Dict[str, int]:
        """firm_to_idx 매핑 로드"""
        if not self.firm_to_idx_path.exists():
            logger.warning(f"⚠️  firm_to_idx 파일 없음: {self.firm_to_idx_path}")
            return {}
        
        df = pd.read_csv(self.firm_to_idx_path)
        
        # 컬럼명 확인
        if '사업자등록번호' in df.columns and 'idx' in df.columns:
            mapping = dict(zip(
                df['사업자등록번호'].astype(str),
                df['idx']
            ))
        else:
            logger.warning("   ⚠️  컬럼명 불일치")
            return {}
        
        logger.info(f"   ✓ Firm-to-Index 매핑: {len(mapping):,}개")
        return mapping
    
    def get_firms_by_ksic(
        self,
        ksic_codes: List[str],
        exact_match: bool = False
    ) -> List[Dict]:
        """
        KSIC 코드로 기업 검색
        
        Parameters
        ----------
        ksic_codes : List[str]
            검색할 KSIC 코드 리스트 (예: ['C26111', 'C26112'])
        exact_match : bool
            True: 정확히 일치하는 코드만
            False: 앞자리가 일치하면 포함 (예: C261로 시작하는 모든 코드)
        
        Returns
        -------
        firms : List[Dict]
            매칭된 기업 리스트
        """
        if self.firm_info_df.empty:
            logger.warning("⚠️  기업 정보 없음")
            return []
        
        # KSIC 컬럼 확인
        ksic_col = 'KSIC' if 'KSIC' in self.firm_info_df.columns else 'ksic'
        if ksic_col not in self.firm_info_df.columns:
            logger.warning("⚠️  KSIC 컬럼 없음")
            return []
        
        matched_firms = []
        
        for ksic_code in ksic_codes:
            if exact_match:
                # 정확히 일치
                mask = self.firm_info_df[ksic_col] == ksic_code
            else:
                # 앞자리 일치 (startswith)
                mask = self.firm_info_df[ksic_col].astype(str).str.startswith(ksic_code)
            
            matched = self.firm_info_df[mask]
            
            for _, row in matched.iterrows():
                firm_dict = row.to_dict()
                matched_firms.append(firm_dict)
        
        logger.info(f"   ✓ KSIC 매칭: {len(matched_firms):,}개 기업")
        return matched_firms
    
    def get_firm_indices_by_ksic(
        self,
        ksic_codes: List[str],
        exact_match: bool = False
    ) -> List[int]:
        """
        KSIC 코드로 기업의 그래프 인덱스 검색
        
        Returns
        -------
        indices : List[int]
            그래프 상의 노드 인덱스 리스트
        """
        firms = self.get_firms_by_ksic(ksic_codes, exact_match)
        
        indices = []
        missing_count = 0
        
        for firm in firms:
            # 사업자등록번호 추출
            firm_id = None
            for key in ['사업자등록번호', 'firm_id', 'business_id']:
                if key in firm:
                    firm_id = str(firm[key])
                    break
            
            if firm_id and firm_id in self.firm_to_idx:
                idx = self.firm_to_idx[firm_id]
                indices.append(idx)
            else:
                missing_count += 1
        
        if missing_count > 0:
            logger.warning(f"   ⚠️  매핑 실패: {missing_count}개 기업")
        
        logger.info(f"   ✓ 인덱스 변환: {len(indices):,}개")
        return indices
    
    def get_known_firms_indices(
        self,
        firm_ids: List[str]
    ) -> Dict[str, int]:
        """
        알려진 기업 ID 리스트를 인덱스로 변환
        
        Parameters
        ----------
        firm_ids : List[str]
            사업자등록번호 리스트 (예: ['LI5265', '240623'])
        
        Returns
        -------
        mapping : Dict[str, int]
            {firm_id: node_index}
        """
        mapping = {}
        missing = []
        
        for firm_id in firm_ids:
            if firm_id in self.firm_to_idx:
                mapping[firm_id] = self.firm_to_idx[firm_id]
            else:
                missing.append(firm_id)
        
        if missing:
            logger.warning(f"   ⚠️  매핑 실패 기업: {missing}")
        
        logger.info(f"   ✓ Known firms: {len(mapping)}/{len(firm_ids)}개 매핑 성공")
        return mapping


# ============================================================
# 2019 일본 수출규제 시나리오 정의
# ============================================================

class JapanExportRestriction2019:
    """
    2019년 일본 수출규제 시나리오
    
    반도체 핵심 소재 (불화수소, 포토레지스트, 불화폴리이미드)
    """
    
    # 공급자: 소재 생산 기업 KSIC
    SUPPLIER_KSIC_CODES = [
        'C20129',  # 기타 기초 무기화학 물질 제조업
        'C20119',  # 기타 기초 유기 화학물질 제조업
        'C20499',  # 그 외 기타 분류 안된 화학제품 제조업
        'C20122',  # 산소, 질소 및 기타 산업용 가스 제조업
        'C20501',  # 합성섬유 제조업
    ]
    
    # 필수 공급자 (알려진 기업)
    KNOWN_SUPPLIERS = [
        'LI5265',   # 솔브레인
        '240623',   # 램테크놀러지
        '215316',   # 이엔에프테크놀로지
        '355950',   # 동진쎄미켐
        'F03302',   # 코오롱인더스트리
        '350885',   # SKC
        '093025',   # SK머티리얼즈
    ]
    
    # 수요자: 반도체/디스플레이 기업 KSIC
    BUYER_KSIC_CODES = [
        'C26110',  # 전자집적회로 제조업
        'C26111',  # 메모리용 전자집적회로 제조
        'C26112',  # 비메모리용 및 기타 전자집적회로 제조
        'C26120',  # 다이오드, 트랜지스터 및 유사 반도체소자 제조업
        'C26121',  # 발광 다이오드 제조업
        'C26129',  # 기타 반도체 소자 제조업
    ]
    
    # 필수 수요자 (알려진 기업)
    KNOWN_BUYERS = [
        '380725',  # 삼성전자
        '383511',  # SK하이닉스
        '452556',  # MEMC코리아
        '092819',  # 서울반도체
        '360651',  # SK실트론
    ]
    
    @staticmethod
    def get_scenario_config() -> Dict:
        """시나리오 설정 반환"""
        return {
            'name': 'japan_export_restriction_2019',
            'description': '2019년 일본 수출규제 (반도체 핵심 소재)',
            'shock_date': '2019-07-04',
            'target_materials': ['불화수소', '포토레지스트', '불화폴리이미드'],
            'pre_shock_year': 2018,
            'shock_year': 2019,
            'post_shock_year': 2020,
            'supplier_ksic': JapanExportRestriction2019.SUPPLIER_KSIC_CODES,
            'buyer_ksic': JapanExportRestriction2019.BUYER_KSIC_CODES,
            'known_suppliers': JapanExportRestriction2019.KNOWN_SUPPLIERS,
            'known_buyers': JapanExportRestriction2019.KNOWN_BUYERS,
        }
